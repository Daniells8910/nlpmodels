# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
from datasets.dataset import TrainParam, DatasetIterator
import logging
import time


class ModelParam(object):
    def __init__(self):
        # 词向量是否不更新
        self.is_static_word2vec = False
        # 初始词向量权重
        self.weights = None
        # 分类类型
        self.class_num = 2
        # 词汇表大小
        self.vocab_size = 0
        # 词向量维度
        self.embed_dim = 0
        # 卷积核数量
        self.kernel_num = 0
        # 卷积核步长列表
        self.kernel_size_list = []
        self.dropout = 0.5

    @classmethod
    def from_dict(cls, param: dict):
        obj = ModelParam()
        for k, v in param.items():
            obj.__setattr__(k, v)
        return obj


class TextcnnModel(nn.Module):
    def __init__(self, args: ModelParam):
        super(TextcnnModel, self).__init__()
        self.args = args

        weights = None
        if self.args.weights is not None:
            if isinstance(self.args.weights, np.ndarray):
                weights = torch.from_numpy(self.args.weights).float()
            elif isinstance(self.args.weights, list):
                weights = torch.FloatTensor(self.args.weights)

            # 这个参数比较大，而一次性使用，置空可以压缩模型保存大小
            self.args.weights = None

        self.embed = nn.Embedding(self.args.vocab_size, self.args.embed_dim, _weight=weights)

        in_channel = 1
        # 输出通道数是卷积核的数量
        out_channel = self.args.kernel_num
        embed_dim = self.args.embed_dim
        KS = self.args.kernel_size_list
        self.convslist = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, embed_dim)) for K in KS])
        self.dropout = nn.Dropout(self.args.dropout, inplace=True)
        self.fc = nn.Linear(len(KS) * out_channel, self.args.class_num)

    def get_predict_args(self) -> dict:
        args = self.args
        args.weights = None
        args.dropout = 1.0
        return args.__dict__

    # N: 样本批数（batch_size)
    # W: 一条文本中词会的数量
    # D: 词向量的维度
    # Ci: 输入通道数
    # Co: 输出通道数
    # K: 卷积核尺寸（步长）
    # KS: 卷积核尺寸列表
    # C: 分类类别数
    def forward(self, input_x):
        # x: (N, W)
        x = self.embed(input_x)
        # x: (N, W, D)

        x = x.unsqueeze(1)
        # x: (N, 1, W, D)

        if self.args.is_static_word2vec:
            x = Variable(x, reqires_grad=False)

        # 卷积维度输出:
        #   [       batch_size,
        #           out_channel, # 输出通道数等于卷积核数
        #           (weight - kernel_size[0] + 2 * padding) / stride[0] + 1,
        #           (hegith - kernel_size[1] + 2 * padding) / stride[1] + 1
        #   ]
        #
        # x尺寸: (N, Ci, W, D), 卷积核的输入：input_channel:Ci, output_channel: Co， kernel_size: (K, D)
        # conv(x) 输出尺寸: (N, Co, W - K + 1, D - D + 1) == (N, Co, W - K + 1, 1)
        # relu不改变输入输出尺寸
        # （N, Co, W-K+1, 1).squeeze(3) : (N, Co, W-K+1)

        x = [F.relu(conv(x), inplace=True).squeeze(3) for conv in self.convslist]
        # x: [(N, Co, W-K+1), ...] * len(KS)

        # i: (N, Co, W-K+1)
        # i.size(2)=W-K+1
        # max_pool1d(i, W-K+1): (N, Co, 1)
        # (N, Co, 1).squeeze(2) : (N, Co)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # x: [(N, Co), ...] * len(KS)

        # x: [(N, Co), ...] * len(Ks)
        # cat(x,1): (N, Co * len(Ks))
        x = torch.cat(x, 1)
        # x: (N, len(KS) * Co)
        x = self.dropout(x)
        logit = self.fc(x)
        # logit: (N, C)
        return logit


class ModelTrain(object):
    def __init__(self, args: TrainParam, model_args: ModelParam):
        self.args = args
        self.model = TextcnnModel(model_args)
        if not self._load():
            logging.info("Created model with fresh parameters.")
        if self.args.cuda:
            self.model.cuda()

    def _eval(self, data_iter: DatasetIterator):
        with torch.no_grad():
            # 中间计算不需要梯度，节省显存
            x_batch, y_batch = data_iter.rand_testdata(self.args.test_batch_size)
            x_batch = torch.from_numpy(np.array(x_batch)).long()
            y_batch = torch.from_numpy(np.array(y_batch)).long()
            batch_size = y_batch.size()[0]

            if self.args.cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            logit = self.model(x_batch)
            loss = F.cross_entropy(logit, y_batch)
            correct_num = (torch.max(logit, 1)[1].view(y_batch.size()).long() == y_batch).sum()
            accurary = 100.0 * correct_num / batch_size
            logging.info("\t Test - loss: %.4f  acc:%.4f%% %d/%d" % (loss.item(), accurary, correct_num, batch_size))

    def _load(self) -> bool:
        if not self.args.continue_train:
            return False

        snapshot = os.path.join(self.args.model_save_dir, "snapshot")
        if not os.path.exists(snapshot):
            return False

        checkpoint = os.path.join(self.args.model_save_dir, "checkpoint")
        with open(checkpoint, "r", encoding="utf-8") as rf:
            steps = int(rf.readline().strip())
            fpath = os.path.join(snapshot, "%d.pt" % steps)
            if os.path.exists(fpath):
                self.model.load_state_dict(torch.load(fpath))
                logging.info("Reading model parameters from %s" % fpath)
                return True
        return False

    def _save(self, steps=-1):
        if not os.path.exists(self.args.model_save_dir):
            os.makedirs(self.args.model_save_dir)

        if steps > 0:
            snapshot = os.path.join(self.args.model_save_dir, "snapshot")
            if not os.path.exists(snapshot):
                os.makedirs(snapshot)
            save_path = os.path.join(snapshot, "%d.pt" % steps)
            torch.save(self.model.state_dict(), save_path)

            checkpoint = os.path.join(self.args.model_save_dir, "checkpoint")
            with open(checkpoint, "w", encoding="utf-8") as wf:
                wf.write(str(steps) + "\n")
        else:
            ext = os.path.splitext(self.args.model_name)[1]
            if len(ext) == 0:
                save_path = os.path.join(self.args.model_save_dir, self.args.model_name + ".pt")
            else:
                save_path = os.path.join(self.args.model_save_dir, self.args.model_name)
            # torch.save(self.model, save_path)
            state = {
                "model_args": self.model.get_predict_args(),
                "state_dict": self.model.state_dict()
            }
            torch.save(state, save_path)

    def valid(self, data_iter: DatasetIterator):
        self.model.eval()
        with torch.no_grad():
            # 中间计算不需要梯度，节省显存
            b = time.time()
            correct_num = 0
            batch_num = data_iter.test_num // self.args.test_batch_size
            if data_iter.test_num % self.args.test_batch_size != 0:
                batch_num += 1
            count = 0
            for x_batch, y_batch in data_iter.next_testdata(self.args.test_batch_size):
                count += 1
                logging.info("%.f%%, %d/%d" % (count * 100.0 / batch_num, count, batch_num))
                x_batch = torch.from_numpy(np.array(x_batch)).long()
                y_batch = torch.from_numpy(np.array(y_batch)).long()
                if self.args.cuda:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                logit = self.model(x_batch)
                correct_num += (torch.max(logit, 1)[1].view(y_batch.size()).long() == y_batch).sum()
            acc = 100.0 * correct_num / data_iter.test_num
            logging.info("-----> acc:%.f%%, cost:%f" % (acc, time.time() - b))

    def train(self, data_iter: DatasetIterator):
        b = time.time()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        steps = 0
        self.model.train()
        for epoch in range(1, self.args.epoches + 1):
            batch_count = 0
            for x_batches, y_batches in data_iter.next_batch(self.args.train_batch_size):
                batch_count += 1
                fetures = torch.from_numpy(np.array(x_batches)).long()
                labels = torch.from_numpy(np.array(y_batches)).long()
                if self.args.cuda:
                    fetures, labels = fetures.cuda(), labels.cuda()
                batch_size = labels.size()[0]

                optimizer.zero_grad()
                logit = self.model(fetures)

                loss = F.cross_entropy(logit, labels)
                loss.backward()
                optimizer.step()

                steps += 1
                if steps % self.args.log_interval == 0:
                    # labels是（batch_size, 1)的矩阵
                    correct_num = (torch.max(logit, 1)[1].view(labels.size()).long() == labels).sum()
                    accuracy = 100.0 * correct_num / batch_size
                    print("epoch-%d step-%d batch-%d - loss: %.4f  acc: %.2f%% %d/%d" % (
                        epoch, steps, batch_count, loss.item(), accuracy, correct_num, batch_size))
                    logging.info("epoch-%d step-%d batch-%d - loss: %.4f  acc: %.2f%% %d/%d" % (
                        epoch, steps, batch_count, loss.item(), accuracy, correct_num, batch_size))

                if steps % self.args.test_interval == 0:
                    self.model.eval()
                    self._eval(data_iter)
                    self.model.train()

                if steps % self.args.save_interval == 0:
                    self._save(steps)
        self._save()
        logging.info("train finished, cost:%f" % (time.time() - b))

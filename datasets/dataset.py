import numpy as np
import random
import pickle
import logging
import time
import os
from tools import regcommon as common
import codecs

UNK = '<UNK>'
UNK_ID = 0


def _clean_text(text: str):
    text = common.normalize(text.strip())
    return text


class TrainParam(object):
    def __init__(self):
        self.learning_rate = 0.001
        self.epoches = 0
        self.cuda = True
        self.log_interval = 3
        self.test_interval = 50
        self.save_interval = 100
        self.train_batch_size = 64
        self.test_rate = 0.1
        self.test_batch_size = 100
        self.model_save_dir = "model_bin"
        self.model_name = "modelname"
        self.continue_train = True


class DatasetParam(object):
    """
        该类用于配置数据集的参数
        最终要传入Dataset中作为参数
    """

    def __init__(self):
        self.output_dir = ""  # 模型词汇表及log输出目录
        self.embed_dim = 0  # 词向量维度
        self.max_sentence_len = 100  # 句子最大词数
        self.min_word_freq = 1  # 词频最小值，大于该值才进入词汇表
        self.max_vocab_size = np.inf  # 词汇表最大词数
        self.test_rate = 0.1  # 测试集比例
        self.word2vec_iterator = None  # 训练好的word2vec模型
        self.batch_size = 5  # 批次大小
        # 分词函数
        self.tokenizer = None
        # 训练数据目录
        self.data_dir = ""
        # 提取词汇表目录
        self.data_vocab_dir = ""
        # 分类类别列表
        self.cate_list = []

    def check(self):
        assert self.tokenizer
        assert self.data_dir and os.path.exists(self.data_dir)
        assert self.output_dir
        assert len(self.cate_list) >= 2


class DataTransform(object):
    """
            数据处理类
            把原始数据（比如一段话） 清洗，分词得到分词后映射到词汇表中对应的索引
    """

    def __init__(self, vocab_file, tokenizer):
        self._tokenizer = tokenizer
        self.max_sent_len, self.vocab_dict = DataTransform.load_vocab(vocab_file)
        logging.info("max_sent_len:%d, vocab_size:%d" % (self.max_sent_len, len(self.vocab_dict)))

    @classmethod
    def load_vocab(cls, vocab_file):
        with open(vocab_file, "rb") as rf:
            vocab = pickle.loads(rf.read())
            max_sent_len = vocab["max_sent_len"]
            vocab_list = vocab["vocab_list"]
            vocab_dict = dict([(x, y) for (y, x) in enumerate(vocab_list)])
            return max_sent_len, vocab_dict

    def text2ids(self, text, pad_empty=True):
        """
        1.清洗数据后切词 得到 words
        2.判断words中每一个词是否在词汇表vocab_dict中，words（分词list）->  word_ids（分词对应id的list）
        3.补全数据，为word_ids补0，补充到词汇表的长度
        :param text:
        :param pad_empty:
        :return:
        """
        text = _clean_text(text)
        words = self._tokenizer.tokenize(text)
        if len(words) == 0:
            word_ids = []
        else:
            word_ids = [self.vocab_dict.get(word, UNK_ID) for i, word in enumerate(words) if i < self.max_sent_len]

        word_ids_len = len(word_ids)
        assert word_ids_len <= self.max_sent_len
        if 0 < word_ids_len < self.max_sent_len:
            npad = (0, self.max_sent_len - word_ids_len)
            word_ids = np.pad(word_ids, pad_width=npad, mode="constant", constant_values=UNK_ID)
        elif word_ids_len == 0 and pad_empty:
            word_ids = np.pad(word_ids, pad_width=(0, self.max_sent_len), mode="constant", constant_values=UNK_ID)

        return word_ids


class Dataset(object):
    """
        传入DatasetParam参数，执行build方法 生成 data_iter(数据集), vocab_dict（词汇表）, weights（词汇表的词向量字典）
    """

    def __init__(self, args: DatasetParam):
        self.args = args
        self.args.check()

    def _create_vocab(self, vocab_file):
        logging.info("\tcreate vocab ...")
        vocab = {}
        b = time.time()
        count = 0
        with codecs.open(self.args.data_file, 'r', encoding='utf8') as rf:
            text = rf.readline().strip()
            while text:
                if count % 1000 == 0:
                    logging.info("\t\tprocessing %d" % count)
                tag = text[0]
                if tag in self.args.data_vocab_tag:
                    text = _clean_text(text[2:])
                    words = self.args.tokenizer.tokenize(text)
                    for word in words:
                        word = word.strip()
                        assert word
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1
                count += 1
                text = rf.readline().strip()
        new_vocab = {}
        for k, v in vocab.items():
            if v >= self.args.min_word_freq:
                new_vocab[k] = v
        vocab_list = [UNK] + sorted(new_vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > self.args.max_vocab_size:
            vocab_list = vocab_list[:self.args.max_vocab_size]

        with open(vocab_file, "wb") as wf:
            wf.write(pickle.dumps({
                "max_sent_len": self.args.max_sentence_len,
                "vocab_list": vocab_list
            }))
        logging.info("\t create vocab finished! cost:%f" % (time.time() - b))
        return dict([(x, y) for (y, x) in enumerate(vocab_list)])

    def _load_trainset(self, trainset_file):
        with open(trainset_file, "rb") as rf:
            trainset = pickle.loads(rf.read())
        return trainset

    def _create_trainset(self, trainset_file, vocab_file, seg_sentence: bool) -> list:
        logging.info("\tcreate trainset...")
        transform = DataTransform(vocab_file, self.args.tokenizer)

        b = time.time()
        train_set = []
        test_set = []

        with codecs.open(self.args.data_file, 'r', encoding='utf8') as rf:
            texts = rf.readlines()
            for i, cate in enumerate(self.args.cate_list):
                count = 0
                cate_set = []
                for text in texts:
                    text = text.strip()
                    tag = text[0]
                    if tag == cate:
                        count += 1
                        if count % 1000 == 0:
                            print('数据标签分类{0}，处理条数:{1}'.format(cate, count))
                        if seg_sentence:
                            text_size = len(text[2:])
                            batch_size = text_size // self.args.max_sentence_len
                            if text_size % self.args.max_sentence_len != 0:
                                batch_size += 1
                            for idx in range(0, batch_size):
                                offset = idx * self.args.max_sentence_len
                                subtext = text[offset: offset + self.args.max_sentence_len]
                                word_ids = transform.text2ids(subtext, False)
                                if len(word_ids) > 0:
                                    assert len(word_ids) == self.args.max_sentence_len
                                    cate_set.append([i, word_ids])
                        else:
                            word_ids = transform.text2ids(text[2:], False)
                            if len(word_ids) > 0:
                                assert len(word_ids) == self.args.max_sentence_len
                                cate_set.append([i, word_ids])
                if len(cate_set) > 0:
                    n = int(len(cate_set) * self.args.test_rate)
                    test_set.extend(cate_set[0:n])
                    train_set.extend(cate_set[n:])

        np.random.shuffle(test_set)
        np.random.shuffle(train_set)
        ds = [len(test_set)] + test_set + train_set
        with open(trainset_file, "wb") as wf:
            wf.write(pickle.dumps(ds))
        logging.info("\tcreate trainset finished! cost:%f" % (time.time() - b))
        return ds

    def buildWithAllData(self, seg_sentence=False):
        """
        配合简单模型，直接提供所有训练集使用，不提供minibatch
        :return: 训练数据集
        """
        logging.info("build dataset...")
        b = time.time()
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        trainset_file = os.path.join(self.args.output_dir, "trainset")
        vocab_file = os.path.join(self.args.output_dir, "vocab")

        # 创建词汇表
        if not os.path.exists(vocab_file):
            vocab_dict = self._create_vocab(vocab_file)
        else:
            _, vocab_dict = DataTransform.load_vocab(vocab_file)

        # 创建训练集文件
        if os.path.exists(trainset_file):
            dataset = self._load_trainset(trainset_file)
        else:
            dataset = self._create_trainset(trainset_file, vocab_file, seg_sentence)

        # 序列化测试集和训练集到内存
        labels = []
        docids = []
        test_size = dataset[0]
        data_list = dataset[1:]
        for data in data_list:
            # label 用单数字表示，不需要转出one_hot
            labels.append(int(data[0]))
            docids.append(np.array(data[1]))
        labels = np.array(labels)
        docids = np.array(docids)
        batches = list(zip(docids, labels))
        logging.info("build cost:%f" % (time.time() - b))

        test_set = batches[0: test_size]
        train_set = batches[test_size:]

        return train_set, test_set

    def build(self, seg_sentence=False):
        """
        data_dir:  训练数据目录, 目录结构如：
                            类别1
                            类别2
                            类别3
                            ...
        cate_list: 需要训练的类别名称（对应data_dir目录下部分或者全部子目录名称)
        """
        logging.info("build dataset...")
        b = time.time()
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        trainset_file = os.path.join(self.args.output_dir, "trainset")
        vocab_file = os.path.join(self.args.output_dir, "vocab")

        # 创建词汇表
        if not os.path.exists(vocab_file):
            vocab_dict = self._create_vocab(vocab_file)
        else:
            _, vocab_dict = DataTransform.load_vocab(vocab_file)

        # 创建训练集文件
        if os.path.exists(trainset_file):
            dataset = self._load_trainset(trainset_file)
        else:
            dataset = self._create_trainset(trainset_file, vocab_file, seg_sentence)

        # 加载词向量
        weights = None
        if self.args.word2vec_iterator:
            weights = np.random.uniform(-0.25, 0.25, (len(vocab_dict), self.args.embed_dim))
            for word, vec in self.args.word2vec_iterator:
                if word in vocab_dict:
                    word_id = vocab_dict[word]
                    weights[word_id] = np.array(vec)

        # 序列化测试集和训练集到内存
        labels = []
        docids = []
        test_size = dataset[0]
        data_list = dataset[1:]
        for data in data_list:
            # label 用单数字表示，不需要转出one_hot
            labels.append(int(data[0]))
            docids.append(np.array(data[1]))
        labels = np.array(labels)
        docids = np.array(docids)
        batches = list(zip(docids, labels))
        logging.info("build cost:%f" % (time.time() - b))
        return DatasetIterator(test_size, batches), vocab_dict, weights


class DatasetIterator(object):
    def __init__(self, test_size, dataset_list: list):
        self.test_set = dataset_list[0: test_size]
        self.train_set = dataset_list[test_size:]
        self.test_num = len(self.test_set)
        self.train_num = len(self.train_set)
        logging.info("train num:%d, test num:%d" % (self.train_num, self.test_num))

    def next_batch(self, batch_size):
        if batch_size > self.train_num:
            batch_size = self.train_num
        assert batch_size <= self.train_num
        np.random.shuffle(self.train_set)
        for i in range(0, self.train_num, batch_size):
            batches = self.train_set[i: i + batch_size]
            x_batches, y_batches = zip(*batches)
            yield x_batches, y_batches

    def rand_testdata(self, size):
        batches = random.sample(self.test_set, size)
        x_batches, y_batches = zip(*batches)
        return x_batches, y_batches

    def next_testdata(self, batch_size):
        if batch_size >= self.test_num:
            batch_size = self.test_num
        assert batch_size <= self.test_num
        for i in range(0, self.test_num, batch_size):
            batches = self.test_set[i: i + batch_size]
            x_batches, y_batches = zip(*batches)
            yield x_batches, y_batches

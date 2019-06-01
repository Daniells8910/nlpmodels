# -*- coding: utf-8 -*-
import os
import numpy as np

"分类模型"


def classmodels(configs, word2vec, tokenizer):
    model_type = configs['modelType']

    # 机器学习模型
    if model_type == 'naivebayes':
        model = NaiveBayesModel()
        model.train(configs, None, tokenizer)
    elif model_type == 'lr':
        model = LogiticRegressionModel()
        model.train(configs, None, tokenizer)
    elif model_type == 'xgboost':
        model = XgboostModel()
        model.train(configs, None, tokenizer)
    # 深度学习模型
    elif model_type == 'textcnn':
        model = TextCnnModel()
        model.train(configs, word2vec, tokenizer)


# TextCnn
class TextCnnModel(object):
    def train(self, config, word2vec, tokenizer):
        from datasets.dataset import DatasetParam, Dataset, TrainParam
        from models.TextCnn import ModelParam, ModelTrain
        import logging
        if not os.path.exists(config['data_params']['output_dir']):
            os.makedirs(config['data_params']['output_dir'])
        fmt = "%(asctime)s:%(filename)s:%(funcName)s:%(lineno)s: %(levelname)s - %(message)s"
        logging.basicConfig(filename=os.path.join(config['data_params']['output_dir'], "train.log"), format=fmt,
                            level=logging.DEBUG)
        # 数据预处理
        dataset_args = DatasetParam()
        dataset_args.output_dir = config['data_params']['output_dir']
        dataset_args.embed_dim = config['data_params']['embed_dim']
        dataset_args.max_sentence_len = config['data_params']['max_sentence_len']
        dataset_args.min_word_freq = config['data_params']['min_word_freq']
        dataset_args.max_vocab_size = config['data_params']['max_vocab_size']
        dataset_args.test_rate = config['data_params']['test_rate']
        dataset_args.tokenizer = tokenizer
        dataset_args.data_dir = config['data_params']['data_dir']
        dataset_args.cate_list = config['model_params']['cate_list']
        dataset_args.word2vec_iterator = word2vec
        dataset_args.data_vocab_dir = config['data_params']['data_vocab_dir']
        dataset_args.data_vocab_tag = str(config['data_params']['data_vocab_tag'])
        dataset_args.data_file = config['data_params']['data_file']
        dataset = Dataset(dataset_args)
        data_iter, vocab_dict, weights = dataset.build(config['data_params']['seg_sentence'])

        # 初始化模型参数
        model_args = ModelParam()
        model_args.is_static_word2vec = config['model_params']['is_static_word2vec']
        model_args.weights = weights
        model_args.class_num = len(config['model_params']['cate_list'])
        model_args.vocab_size = len(vocab_dict)
        model_args.embed_dim = config['data_params']['embed_dim']
        model_args.kernel_num = config['model_params']['kernel_num']
        model_args.kernel_size_list = config['model_params']['kernel_size_list']
        model_args.dropout = config['model_params']['dropout']

        # 初始化训练参数
        train_args = TrainParam()
        train_args.learning_rate = config['train_params']['learning_rate']
        train_args.epoches = config['train_params']['epoches']
        train_args.cuda = config['train_params']['cuda']
        train_args.log_interval = config['train_params']['log_interval']
        train_args.test_interval = config['train_params']['test_interval']
        train_args.save_interval = config['train_params']['save_interval']
        train_args.train_batch_size = config['train_params']['train_batch_size']
        train_args.test_batch_size = config['train_params']['test_batch_size']
        train_args.model_save_dir = config['data_params']['output_dir']
        train_args.model_name = config['train_params']['model_name']
        train_args.continue_train = config['train_params']['continue_train']

        model_train = ModelTrain(train_args, model_args)
        # 训练
        is_valid_only = config['train_params']['is_valid_only']
        if not is_valid_only:
            model_train.train(data_iter)
        model_train.valid(data_iter)


# NaiveBayes
class NaiveBayesModel(object):
    def train(self, config, word2vec, tokenizer):
        from datasets.dataset import Dataset, DatasetParam
        dataset_args = DatasetParam()
        dataset_args.output_dir = config['data_params']['output_dir']
        dataset_args.embed_dim = config['data_params']['embed_dim']
        dataset_args.max_sentence_len = config['data_params']['max_sentence_len']
        dataset_args.min_word_freq = config['data_params']['min_word_freq']
        dataset_args.max_vocab_size = config['data_params']['max_vocab_size']
        dataset_args.test_rate = config['data_params']['test_rate']
        dataset_args.tokenizer = tokenizer
        dataset_args.data_dir = config['data_params']['data_dir']
        dataset_args.cate_list = config['model_params']['cate_list']
        dataset_args.word2vec_iterator = word2vec
        dataset_args.data_vocab_dir = config['data_params']['data_vocab_dir']
        dataset_args.data_vocab_tag = str(config['data_params']['data_vocab_tag'])
        dataset_args.data_file = config['data_params']['data_file']
        dataset = Dataset(dataset_args)
        train_set, test_set = dataset.buildWithAllData(False)
        x_train, y_train = zip(*train_set)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test, y_test = zip(*test_set)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # 加载贝叶斯模型
        from sklearn.naive_bayes import BernoulliNB
        from sklearn.externals import joblib
        classifier = BernoulliNB()

        # 训练模型并保存
        classifier.fit(x_train, y_train)
        joblib.dump(classifier, os.path.join(dataset_args.output_dir, 'bayes_model.m'))

        # 验证并计算acc
        y_ = classifier.predict(x_test)
        acc = np.mean([1 if y_[i] == y_test[i] else 0 for i in range(y_test.shape[0])], axis=0)
        print("eval acc: %f" % acc)


# LogiticRegression
class LogiticRegressionModel(object):
    def train(self, config, word2vec, tokenizer):
        from datasets.dataset import Dataset, DatasetParam
        dataset_args = DatasetParam()
        dataset_args.output_dir = config['data_params']['output_dir']
        dataset_args.embed_dim = config['data_params']['embed_dim']
        dataset_args.max_sentence_len = config['data_params']['max_sentence_len']
        dataset_args.min_word_freq = config['data_params']['min_word_freq']
        dataset_args.max_vocab_size = config['data_params']['max_vocab_size']
        dataset_args.test_rate = config['data_params']['test_rate']
        dataset_args.tokenizer = tokenizer
        dataset_args.data_dir = config['data_params']['data_dir']
        dataset_args.cate_list = config['model_params']['cate_list']
        dataset_args.word2vec_iterator = word2vec
        dataset_args.data_vocab_dir = config['data_params']['data_vocab_dir']
        dataset_args.data_vocab_tag = str(config['data_params']['data_vocab_tag'])
        dataset_args.data_file = config['data_params']['data_file']
        dataset = Dataset(dataset_args)
        train_set, test_set = dataset.buildWithAllData(False)
        x_train, y_train = zip(*train_set)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test, y_test = zip(*test_set)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # 加载LogistRegression模型
        from sklearn.linear_model import LogisticRegression
        from sklearn.externals import joblib
        classifier = LogisticRegression()

        # 训练模型并保存
        classifier.fit(x_train, y_train)
        joblib.dump(classifier, os.path.join(dataset_args.output_dir, 'lr_model.m'))

        # 验证并计算acc
        y_ = classifier.predict(x_test)
        acc = np.mean([1 if y_[i] == y_test[i] else 0 for i in range(y_test.shape[0])], axis=0)
        print("eval acc: %f" % acc)


# xgboost
class XgboostModel(object):
    def train(self, config, word2vec, tokenizer):
        from datasets.dataset import Dataset, DatasetParam
        dataset_args = DatasetParam()
        dataset_args.output_dir = config['data_params']['output_dir']
        dataset_args.embed_dim = config['data_params']['embed_dim']
        dataset_args.max_sentence_len = config['data_params']['max_sentence_len']
        dataset_args.min_word_freq = config['data_params']['min_word_freq']
        dataset_args.max_vocab_size = config['data_params']['max_vocab_size']
        dataset_args.test_rate = config['data_params']['test_rate']
        dataset_args.tokenizer = tokenizer
        dataset_args.data_dir = config['data_params']['data_dir']
        dataset_args.cate_list = config['model_params']['cate_list']
        dataset_args.word2vec_iterator = word2vec
        dataset_args.data_vocab_dir = config['data_params']['data_vocab_dir']
        dataset_args.data_vocab_tag = str(config['data_params']['data_vocab_tag'])
        dataset_args.data_file = config['data_params']['data_file']
        dataset = Dataset(dataset_args)

        # 加载xgboost参数
        xgboost_args = dict()
        xgboost_args['learning_rate'] = config['xgboost_params']['learning_rate']
        xgboost_args['n_estimators'] = config['xgboost_params']['n_estimators']  # 树的个数--100棵树建立xgboost 总共迭代的次数
        xgboost_args['max_depth'] = config['xgboost_params']['max_depth']  # 树的深度
        xgboost_args['min_child_weight'] = config['xgboost_params']['min_child_weight']  # 叶子节点最小权重
        xgboost_args['gamma'] = config['xgboost_params']['gamma']  # 惩罚项中叶子结点个数前的参数
        xgboost_args['subsample'] = config['xgboost_params']['subsample']  # 随机选择80%样本建立决策树
        xgboost_args['colsample_btree'] = config['xgboost_params']['colsample_btree']  # 随机选择80%特征建立决策树
        xgboost_args['objective'] = config['xgboost_params']['objective']  # 指定损失函数
        xgboost_args['scale_pos_weight'] = config['xgboost_params']['scale_pos_weight']  # 解决样本个数不平衡的问题
        xgboost_args['nthread'] = config['xgboost_params']['nthread']  # 使用全部CPU进行并行运算
        xgboost_args['random_state'] = config['xgboost_params']['random_state']  # 随机数
        xgboost_args['num_class'] = config['xgboost_params']['num_class']  # 分类数目
        xgboost_args['eval_metric'] = config['xgboost_params']['eval_metric']
        xgboost_args['early_stopping_rounds'] = config['xgboost_params']['early_stopping_rounds']
        xgboost_args['verbose'] = config['xgboost_params']['verbose']

        # 生成数据集
        train_set, test_set = dataset.buildWithAllData(False)
        x_train, y_train = zip(*train_set)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test, y_test = zip(*test_set)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # 加载xgboost模型
        from xgboost import XGBClassifier
        from sklearn.externals import joblib
        classifier = XGBClassifier(learning_rate=xgboost_args['learning_rate'],
                                   n_estimators=xgboost_args['n_estimators'],
                                   max_depth=xgboost_args['max_depth'],
                                   min_child_weight=xgboost_args['min_child_weight'],
                                   gamma=xgboost_args['gamma'],
                                   subsample=xgboost_args['subsample'],
                                   colsample_btree=xgboost_args['colsample_btree'],
                                   objective=xgboost_args['objective'],
                                   scale_pos_weight=xgboost_args['scale_pos_weight'],
                                   nthread=xgboost_args['nthread'],
                                   random_state=xgboost_args['random_state'],
                                   num_class=xgboost_args['num_class']
                                   )

        # 训练模型并保存
        classifier.fit(x_train, y_train,
                       eval_set=[(x_test, y_test)],
                       eval_metric=xgboost_args['eval_metric'],
                       early_stopping_rounds=xgboost_args['early_stopping_rounds'],
                       verbose=xgboost_args['verbose'])
        joblib.dump(classifier, os.path.join(dataset_args.output_dir, 'xgboost_model.m'))

        # 验证并计算acc
        y_ = classifier.predict(x_test)
        acc = np.mean([1 if y_[i] == y_test[i] else 0 for i in range(y_test.shape[0])], axis=0)
        print("eval acc: %f" % acc)


# FastText
class FastTextModel(object):
    def train(self, configs):
        print('hello fasttext')

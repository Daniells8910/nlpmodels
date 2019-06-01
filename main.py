# -*- coding: utf-8 -*-
import yaml
import sys
import codecs
from optparse import OptionParser
import tasks.classification as classify
import tasks.ner as ner
import tools.regcommon as common
from tools.segment import Segment
import os


# 读取命令行参数
def parse_opts():
    op = OptionParser()
    op.add_option(
        '-c', '--config', dest='config', type='str', help='配置文件路径', default='examples/classify/textcnn.yml')
    argv = [] if not hasattr(sys.modules['__main__'], '__file__') else sys.argv[1:]
    (opts, args) = op.parse_args(argv)
    if not opts.config:
        op.print_help()
        exit()
    return opts


class Tokenizer(object):
    def __init__(self, segment):
        self.segment = segment

    def tokenize(self, text: str):
        text = common.normalize(text)
        return self.segment.cut(text)


# 读取预存向量文件
class Word2vecIterator(object):
    def __init__(self, emdding_path):
        self.embedding_path = emdding_path

    def __iter__(self):
        with open(self.embedding_path, "r", encoding="utf-8") as rf:
            row = rf.readline()
            row_size = int(row.strip().split(" ")[0])
            for i, row in enumerate(rf):
                if i % 10000 == 0:
                    print("%.f%% === %d/%d\r" % ((i + 1) * 100.0 / row_size, i + 1, row_size), end="")
                ls = row.strip().split(" ")
                word = ls[0]
                vec = ls[1:]
                yield word, vec
            print()


def main():
    opts = parse_opts()
    config_path = opts.config
    if not os.path.exists(config_path):
        print('配置文件不存在！')
        exit()

    configs = yaml.load(codecs.open(opts.config, encoding='utf8'))

    # 加载词向量
    use_embedding = configs['data_params']['use_embedding']
    if use_embedding:
        embedding_path = configs['data_params']['embedding_path']
        word2vec = Word2vecIterator(embedding_path)

    # 加载停用词
    seg = Segment(
        stopword_files=configs['data_params']['stopwords'],
        jieba_tmp_dir="my_tmp"
    )
    # 加载jieba
    tokenizer = Tokenizer(seg)

    task = configs['task']
    if task == 'classification':
        # 分类任务
        classify.classmodels(configs, word2vec, tokenizer)
    elif task == 'ner':
        # 命名识别识别任务
        ner.nermodels(configs)


if __name__ == '__main__':
    main()

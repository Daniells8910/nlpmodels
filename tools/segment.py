import jieba
import logging
import os

jieba.setLogLevel(log_level=logging.WARN)


class Segment(object):
    """
    去停用词和切词逻辑
    """

    def __init__(self, stopword_files=[], userdict_files=[], jieba_tmp_dir=None):
        if jieba_tmp_dir:
            jieba.dt.tmp_dir = jieba_tmp_dir
            if not os.path.exists(jieba_tmp_dir):
                os.makedirs(jieba_tmp_dir)

        self.stopwords = set()
        for stopword_file in stopword_files:
            with open(stopword_file, "r", encoding="utf-8") as rf:
                for row in rf.readlines():
                    word = row.strip()
                    if len(word) > 0:
                        self.stopwords.add(word)
        for userdict in userdict_files:
            jieba.load_userdict(userdict)

    def cut(self, text):
        words_list = []
        words = jieba.cut(text)
        for word in words:
            word = word.strip()
            if word in self.stopwords or len(word) == 0:
                continue
            words_list.append(word)
        return words_list

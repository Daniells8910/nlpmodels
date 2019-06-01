# -*- coding: utf-8 -*-
from tools.langconv import *
import codecs
import re

regEx_html = '<[^>]+'  # 定义HTML标签的正则表达式


def normalize(tag):
    tag = re.sub(regEx_html, '', tag)
    tag = re.sub("((\r\n)|\n)[\\s\t ]*(\\1)+", '', tag)
    tag = re.sub("^((\r\n)|\n)+", '', tag)
    tag = re.sub("    +| +|　+", '', tag)
    tag = re.sub("<br>", '', tag)
    tag = Converter('zh-hans').convert(tag)
    return tag


def save_txt(data, save_path, tag):
    with codecs.open(save_path, 'wb', encoding='utf8') as out:
        for line in data:
            out.write(tag + ',' + normalize(str(line)) + '\n')

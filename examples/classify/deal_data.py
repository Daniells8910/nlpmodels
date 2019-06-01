# encoding=utf-8
import numpy as np
import codecs

if __name__ == '__main__':

    with codecs.open('data/验证集.txt', 'r', 'utf-8') as f:
        line = f.readline().rstrip()
        index = 1
        while line:
            tag = line[0]
            line = line[2:]
            my_path = 'normal'
            if tag == '1':
                my_path = 'politics'
            with codecs.open('politics_dataset/{0}/{1}.txt'.format(my_path, index), 'wb', 'utf-8') as out:
                out.write(line)
            if index % 1000 == 0:
                print(index)
            index += 1
            line = f.readline().rstrip()

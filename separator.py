import numpy as np
import random
import sys
import codecs

# 前処理を行ったデータを，訓練用，開発用，評価用に分割するためのスクリプト
# 訓練用：開発用：評価用 = 8 : 1 : 1 


def separate_data(name):
    with open('./train_data/all.{}_japanese.jsonlines'.format(name), 'r') as f:
        data = f.read().split('\n')
        data.remove('')
        print('all {} documents'.format(len(data)))
        split_size = round(len(data) / 100)
        random.shuffle(data)
        if split_size == 0:
            test = data[:1]
            dev = data[1:2]
            train = data[2:]
        else:
            test = data[:split_size*5]
            dev = data[split_size*5:split_size*15]
            train = data[split_size*15:]
        print('separate to train:{} dev:{} test:{}'.format(len(train), len(dev), len(test)))
    with codecs.open('./train_data/train.{}_japanese.jsonlines'.format(name), 'w', 'utf-8') as train_f, codecs.open('./train_data/dev.{}_japanese.jsonlines'.format(name), 'w', 'utf-8') as dev_f, codecs.open('./train_data/test.{}_japanese.jsonlines'.format(name), 'w', 'utf-8') as test_f:
        train_f.write('\n'.join(train))
        dev_f.write('\n'.join(dev))
        test_f.write('\n'.join(test))
    return train, dev, test


if __name__ == '__main__':
    assert len(sys.argv) > 1, 'Usage: python separator.py <corpus name(kyoto, ntc, bccwj)>'
    names = sys.argv[1:]
    if names[0] == 'all': names = ['kyoto', 'kyotxt', 'ntc', 'bccwj']

    with codecs.open('./train_data/train.japanese.jsonlines', 'w', 'utf-8') as train_all, codecs.open('./train_data/dev.japanese.jsonlines', 'w', 'utf-8') as dev_all, codecs.open('./train_data/test.japanese.jsonlines', 'w', 'utf-8') as test_all:
        for ftype in names:
            train, dev, test = separate_data(ftype)
            train_all.write('\n'.join(train) + '\n')
            dev_all.write('\n'.join(dev) + '\n')
            test_all.write('\n'.join(test) + '\n')

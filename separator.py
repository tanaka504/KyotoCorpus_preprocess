import numpy as np
import random

# 前処理を行ったデータを，訓練用，開発用，評価用に分割するためのスクリプト
# 訓練用：開発用：評価用 = 8 : 1 : 1 


def separate_data():
    with open('./train_data/all.japanese.jsonlines', 'r') as f:
        data = f.read().split('\n')
        data.remove('')
        print('all {} documents'.format(len(data)))
        split_size = round(len(data)/10)
        random.shuffle(data)
        test = data[:split_size]
        dev = data[split_size:split_size*2]
        train = data[split_size*2:]
        print('separate to train:{} dev:{} test:{}'.format(len(train), len(dev), len(test)))
    with open('./train_data/train.japanese.jsonlines', 'w') as train_f, open('./train_data/dev.japanese.jsonlines', 'w') as dev_f, open('./train_data/test.japanese.jsonlines', 'w') as test_f:
        train_f.write('\n'.join(train))
        dev_f.write('\n'.join(dev))
        test_f.write('\n'.join(test))

if __name__ == '__main__':
    separate_data()

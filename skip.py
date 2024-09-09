import numpy as np

np.random.seed(13)
import csv
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import time
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from datetime import datetime
import matplotlib.pyplot as plt
# from keras.utils import plot_model
# from keras.utils.visualize_util import plot
# import pydot
# from pylab import *
from keras.models import Sequential, Model
from keras.layers.core import Dense

from keras.layers import Input
from keras.utils.data_utils import get_file
from keras.optimizer_v2.nadam import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np

eventlog = 'helpdesk_extend'

window_size = 2


def readcsv(eventlog):
    # csvfile = open('data2/%s' % eventlog, 'r', encoding='utf-8')  # BPIC_2012A\O\W  需要加encoding='utf-8'，其他的删除
    csvfile = open('data/' + eventlog + '.csv')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    sequence = []
    next(spamreader, None)  # skip the headers
    for line in spamreader:
        sequence.append(line)
    ###print(sequence)
    return sequence


data = readcsv(eventlog)
data[:10]


def makeVocabulary(data, eventlog):
    temp = list()
    for line in data:
        temp.append(line[1])
    # ##print(temp)
    temp_temp = set(temp)
    # ##print(temp_temp)

    vocabulary = {sorted(list(temp_temp))[i]: i + 1 for i in range(len(temp_temp))}
    # vocabulary = {sorted(list(temp_temp))[i]:i for i in range(len(temp_temp))}
    vocabulary['0'] = 0
    vocabulary['end'] = len(vocabulary)
    f = open('%s' % eventlog + '_2skipmodel_noTime_noEnd_vocabulary' + '.txt', 'w', encoding='utf-8')
    for k in vocabulary:
        f.write(str(k) + '\t' + str(vocabulary[k]) + '\n')
    return vocabulary


vocabulary = makeVocabulary(data, eventlog)


def processData(data, vocabulary, eventlog):
    front = data[0]
    data_new = []
    time_code_temp = {}
    time_code = {}
    # vocabulary_temp = [data[0][1]]
    for line in data[1:]:
        temp = 0
        # vocabulary_temp.append(line[1])
        if line[0] == front[0]:
            temp1 = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
            temp2 = time.strptime(front[2], "%Y-%m-%d %H:%M:%S")
            temp = datetime.fromtimestamp(time.mktime(temp1)) - datetime.fromtimestamp(time.mktime(temp2))
        else:
            temp = 0
        t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
        week = datetime.fromtimestamp(time.mktime(t)).weekday()
        midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = datetime.fromtimestamp(time.mktime(t)) - midnight
        data_new.append([line[0], vocabulary[str(line[1])], line[2], timesincemidnight, week])
        front = line
    front = data_new[0]
    for row in range(1, len(data_new)):
        line = data_new[row]
        ###print(line)
        if line[0] == front[0]:
            key = str(line[1]) + '-' + str(front[1])
            if key not in time_code_temp:
                ##print(key)
                time_code_temp[key] = []
                time_code_temp[key].append(line[3].seconds)
            else:
                time_code_temp[key].append(line[3].seconds)
        front = data_new[row]
    for key in time_code_temp:
        ##print(key)
        time_code_temp[key] = sorted(time_code_temp[key])
    ##print('**************')
    data_merge = []
    data_temp = [data_new[0]]
    for line in data_new[1:]:
        if line[0] != data_temp[-1][0]:
            data_merge.append(data_temp)
            data_temp = [line]
        else:
            data_temp.append(line)
    data_merge.append(data_temp)
    # data_new[:-1].append(vocabulary['end'])
    ##print(data_new)
    vocabulary_num = len(vocabulary)

    vocabulary_temp = vocabulary
    ##print(vocabulary_temp)
    #     f = open('vector/%s' % eventlog+'CBoW_noTime_noEnd_Vocabulary+'.txt', 'wb')
    #     for k in time_code_temp:
    #         for value in time_code_temp[k]:
    #             f.write(str(k)+'\t'+str(value)+'\n')
    return data_merge, data_new, time_code_temp, vocabulary_num, vocabulary_temp


data_merge, data_new, time_code_temp, vocabulary_num, vocabulary = processData(data, vocabulary, eventlog)


# 采样分 种
# 1.活动1，时间1，活动2，活动3，时间3
# 2.活动1，时间1，时间2，活动3，时间3
# 3.活动1，活动2，活动3，活动4，活动5

def generate_data(corpus, window_size, V):
    maxlen = window_size * 2
    flag = 0
    skip_grams = []
    for words in corpus:
        L = len(words)
        if L == 1:
            continue
        for index, word in enumerate(words):
            contexts = []
            labels = []

            if index > window_size:
                for i in range(1, window_size):
                    context = words[index - i][1]  # 前i个词
                    contexts.append(context)
            elif 0 < index and index < window_size:
                for i in range(1, index):
                    context = words[index - i][1]  # 前i个词
                    contexts.append(context)
            if index < len(words) - window_size:
                for i in range(1, window_size):
                    context = words[index + i][1]  # 后i个词
                    contexts.append(context)
            elif index > len(words) - window_size:
                for i in range(1, len(words) - index - 1):
                    context = words[index + i][1]  # 后i个词
                    contexts.append(context)

            if index == 0 and index < len(words) - window_size:
                for i in range(1, window_size):
                    context = words[index + i][1]
                    contexts.append(context)
                # context = [event[index + 1][1]]
            elif index == 0 and index > len(words) - window_size:
                for i in range(1, len(words) - index - 1):
                    context = words[index + i][1]  # 后i个词
                    contexts.append(context)
            contexts_activity = []
            for w in contexts:
                skip_grams.append([w, word[1]])
                contexts_activity.append([w, word[1]])
            labels.append(word)
            labels_activity = word[1]
            for this_activity in contexts_activity:
                x_activity = this_activity
                y_activity = np_utils.to_categorical([this_activity[1]], V)
                yield (x_activity, y_activity)
# generate_data(data_merge, window_size, vocabulary_num)


X = []
Y = []
for x_activity, y_activity in generate_data(data_merge, window_size, vocabulary_num):
    X.append(x_activity)
    Y.append(y_activity.reshape(-1).tolist())

X = np.array(X)
Y = np.array(Y)

dim = 300
skip_model = Sequential()

skip_model.add(Embedding(input_dim=vocabulary_num, output_dim=dim, input_length=2))

skip_model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(dim,)))

skip_model.add(Dense(vocabulary_num, activation='softmax'))

opt = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.0004, clipvalue=3)

skip_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])  # 设置优化器，损失函数，准确率标准

early_stopping = EarlyStopping(monitor='val_loss', patience=100)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100,verbose=0, mode='auto', cooldown=0, min_lr=0)
start = time.clock()
skip_model.fit(X, Y, validation_split=0.2, verbose=2,callbacks=[lr_reducer], batch_size=40, epochs=300)

end = time.clock()
# print("RunningTime:"%(end-start))


f = open('%s' % eventlog + '_vectors_2skipmodel_noTime_noEnd_Vector_vLoss_v1' + '.txt', 'w')
f.write('{} {}\n'.format(vocabulary_num, dim))
vectors = skip_model.get_weights()[0]
for word in range(vocabulary_num):
    str_vec = ' '.join(map(str, list(vectors[int(word), :])))

    f.write('{} {}\n'.format(word, str_vec))
f.close()
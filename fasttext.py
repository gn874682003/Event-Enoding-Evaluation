# -*- coding:utf-8 -*-
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
from keras.utils import plot_model
# from keras.utils.visualize_util import plot
# import pydot
# from pylab import *
from keras.models import Sequential, Model
from keras.layers.core import Dense
# from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input
from keras.utils.data_utils import get_file
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
# from keras.layers.normalization import BatchNormalization
import numpy as np
from tqdm import tqdm
import os
import gc
import time

tqdm.pandas()
eventlog = 'hd'
# dim = 3
n_gram = 2  # 效果不好 增大


def readcsv(eventlog):
    # csvfile = open('data4/%s' % eventlog, 'r',encoding='utf-8')#BPIC_2012A\O\W  需要加encoding='utf-8'，其他的删除
    csvfile = open('../Dataset/' + eventlog + '.csv')

    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    sequence = []
    next(spamreader, None)  # skip the headers
    for line in spamreader:
        sequence.append(line)
    ###print(sequence)
    return sequence


# data = readcsv(eventlog+'.csv')
data = readcsv(eventlog)
data[:10]


# 事件类型字典
def makeVocabulary(data, eventlog):
    temp = list()
    for line in data:
        # print(line[1])
        temp.append(line[1])
    # ##print(temp)
    temp_temp = set(temp)
    # ##print(temp_temp)

    vocabulary = {sorted(list(temp_temp))[i]: i + 1 for i in range(len(temp_temp))}
    # print(vocabulary)
    # vocabulary = {sorted(list(temp_temp))[i]:i for i in range(len(temp_temp))}
    vocabulary['0'] = 0
    vocabulary['end'] = len(vocabulary)
    # f = open('vector8/%s' % eventlog + '_2CBoW_noTime_noEnd_vocabulary' + '.txt', 'w', encoding='utf-8')
    # for k in vocabulary:
    #     f.write(str(k) + '\t' + str(vocabulary[k]) + '\n')
    return vocabulary


vocabulary = makeVocabulary(data, eventlog)

print(vocabulary)
# vocabulary


def processData(data, vocabulary, eventlog):
    front = data[0]
    data_new = []
    time_code_temp = {}  # 活动类型字典序号-事件序号：[时间s]
    time_code = {}
    # vocabulary_temp = [data[0][1]]
    for line in data[1:]:
        temp = 0

        # vocabulary_temp.append(line[1])
        if line[0] == front[0]:  # 相同事件
            # "%Y/%m/%d %H:%M:%S"
            temp1 = time.strptime(line[2], '%Y/%m/%d %H:%M')  # 将字符串时间转换为时间元组【
            temp2 = time.strptime(front[2], '%Y/%m/%d %H:%M')
            # temp1 = time.strptime(line[2], '%Y/%m/%d %H:%M')#\"%Y/%m/%d %H:%M:%S.%f\"  转变时间
            # temp2 = time.strptime(front[2], '%Y/%m/%d %H:%M')
            temp = datetime.fromtimestamp(time.mktime(temp1)) - datetime.fromtimestamp(
                time.mktime(temp2))  # 活动消耗时间 没有地方用？    mktime将时间转换为秒
        else:
            temp = 0
        t = time.strptime(line[2], '%Y/%m/%d %H:%M')
        week = datetime.fromtimestamp(time.mktime(t)).weekday()  # 返回星期数，星期一为0
        midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = datetime.fromtimestamp(time.mktime(t)) - midnight
        data_new.append([line[0], vocabulary[str(line[1])], line[2], timesincemidnight,
                         week])  # 事件序号、活动类型字典序号、活动时间(str)、计算后的时间、weekday
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
    for line in data_new[1:]:  # 将data_new 里的活动按照事件分组得到 data_merge
        if line[0] != data_temp[-1][0]:  # 当前活动是否和前一个活动属于同一个事件
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

def generate_data(corpus, n_gram, V):
    maxlen = 300
    flag = 0
    # 生成 skip_grams
    skip_grams = []

    for event in corpus:  # 遍历每个事件
        # print(words)
        # print("+++++++++++++++++++++")
        L = len(event)  # 活动个数
        # print(L)
        # print("===================")
        if len(event) == 1:
            continue
        for index, acts in enumerate(event):  # 遍历活动
            # #print(word)
            contexts = []
            labels = []
            # print(index)
            # s = index - n_gram
            # e = index + n_gram + 1

            if index > 0 and index < len(event) - 1:
                context = [event[index - 1][1], event[index + 1][1]]
            elif index == 0:
                context = [event[index + 1][1]]
            else:
                context = [event[index - 1][1]]
            contexts_activity = []
            for w in context:
                skip_grams.append([acts[1], w])
                print(skip_grams)
                contexts_activity.append([acts[1], w])
            #               if acts[1]==40:
            #                 print(acts[1], w)

            # print(contexts)

            labels.append(acts)
            labels_activity = acts[1]  # 当前活动类型序号

            for this_activity in contexts_activity:
                x_activity = this_activity  # sequence.pad_sequences([this_activity], maxlen=maxlen) # 序列填充

                y_activity = np_utils.to_categorical([labels_activity], V)  # V 活动类型个数 热读编码
                # #print(labels_time)
                # y_time = np_utils.to_categorical([labels_time], V)
                # yield (x, y)
                yield (x_activity, y_activity)

            # for x_activity,x_mix,y_activity,y_time in  generate_data(data_merge,window_size,len(vocabulary)):
            #     #print(x_mix)


X = []
Y = []
# print(data_merge)
# print(n_gram)


for x_activity, y_activity in generate_data(data_merge, n_gram, vocabulary_num):
    X.append(x_activity)
    Y.append(y_activity.reshape(-1).tolist())
    # print(Y)

X = np.array(X)
Y = np.array(Y)

max_features = vocabulary_num
maxlen = 300
embed_size = 300
tokenizer = Tokenizer(num_words=3, lower=True)
# tokenizer.fit_on_sequences(voc_list)

tokenizer.fit_on_sequences(vocabulary)

def build_matrix(embeddings_index, word_index):
    embedding_matrix = np.zeros((max_features, embed_size))
    for word, i in tqdm(word_index.items()):
        if i >= max_features: continue
        try:
            # word对应的vector
            embedding_vector = embeddings_index[word]
        except:
            # word不存在则使用unknown的vector
            embedding_vector = embeddings_index["unknown"]
        if embedding_vector is not None:
            # 保证embedding_matrix行的向量与word_index中序号一致
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


fasttext_embedding_matrix = build_matrix(embeddings_index, vocabulary)


def build_model(embedding_matrix=True):
    inp = Input(shape=(2,))
    # if embedding_matrix is None:
    #     x = Embedding(max_features, embed_size)(inp)
    # else:
    #     # x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    # x = Lambda(lambda x: K.mean(x, axis=1), output_shape=(64,))
    # x = Dense(vocabulary_num, activation='softmax')(x)
    # x = Dense(128, activation="relu")(x)
    # x = Embedding(32, return_sequences=True)(x)
    # x = GRU(512, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    # x = Dropout(0.5)(x)
    x = Dense(vocabulary_num, activation="softmax")(x)

    model = Model(inputs=inp, outputs=x)
    opt = Nadam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


fasttext = build_model(fasttext_embedding_matrix)

early_stopping = EarlyStopping(monitor='val_loss', patience=100)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100,
                               verbose=0, mode='auto', cooldown=0, min_lr=0)

model_checkpoint = ModelCheckpoint(
    'vector8/%s' % eventlog + '_2CBoW_noTime_noEnd_Vector_vLoss_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_acc',
    verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

# fasttext.fit(X, Y, validation_split=0.2, verbose=2,
#           callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=16, epochs=200)
start = time.clock()
# batch_size=30
fasttext.fit(X, Y, batch_size=30, epochs=400, validation_split=0.2, verbose=2,
             callbacks=[early_stopping, model_checkpoint, lr_reducer])
end = time.clock()

print('Running time:%ss' % (end - start))
f = open('vector8/%s' % eventlog + '_vectors_2CBoW_noTime_noEnd_Vector_vLoss_v1' + '.txt', 'w')
f.write('{} {}\n'.format(vocabulary_num, embed_size))
vectors = fasttext.get_weights()[0]
for word in range(vocabulary_num):
    str_vec = ' '.join(map(str, list(vectors[int(word), :])))

    f.write('{} {}\n'.format(word, str_vec))
f.close()

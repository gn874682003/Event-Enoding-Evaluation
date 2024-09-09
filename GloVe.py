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

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.preprocessing.text import Tokenizer
import warnings
# import BatchNormalization
from keras.preprocessing.text import Tokenizer
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from datetime import datetime
import matplotlib.pyplot as plt
# from keras.utils import plot_model
#from keras.utils.visualize_util import plot
#import pydot
#from pylab import *
from keras.models import Sequential, Model
from keras.layers.core import Dense
# from tensorflow.python.keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input
from keras.utils.data_utils import get_file
from keras.optimizer_v2.nadam import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.layers.normalization import BatchNormalization
import numpy as np
from tqdm import tqdm
from keras.layers import Dense,Input,GlobalMaxPooling1D
from keras.layers import Conv1D,MaxPooling1D,Embedding,GlobalAveragePooling1D


eventlog = 'helpdesk_extend'
# dim = 3
window_size = 2
def readcsv(eventlog):
    # csvfile = open('data2/%s' % eventlog, 'r', encoding='utf-8')#BPIC_2012A\O\W  需要加encoding='utf-8'，其他的删除
    csvfile = open('data/' + eventlog + '.csv')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    sequence = []
    next(spamreader, None)  # skip the headers
    for line in spamreader:
        sequence.append(line)
    ###print(sequence)
    return sequence
data = readcsv(eventlog)




# 事件类型字典
def makeVocabulary(data, eventlog):
    temp = list()
    for line in data:
        temp.append(line[1])
    temp_temp = set(temp)
    vocabulary = {sorted(list(temp_temp))[i]: i + 1 for i in range(len(temp_temp))}
    vocabulary['0'] = 0
    vocabulary['end'] = len(vocabulary)
    # f = open('vector2/%s' % eventlog + '_2CBoW_noTime_noEnd_vocabulary' + '.txt', 'w', encoding='utf-8')
    # for k in vocabulary:

    #     f.write(str(k) + '\t' + str(vocabulary[k]) + '\n')
    return vocabulary

vocabulary = makeVocabulary(data, eventlog)


def processData(data,vocabulary,eventlog):
    front = data[0]
    data_new = []
    time_code_temp = {} #活动类型字典序号-事件序号：[时间s]
    time_code = {}
    #vocabulary_temp = [data[0][1]]
    for line in data[1:]:
        temp = 0

        #vocabulary_temp.append(line[1])
        if line[0] == front[0]: # 相同事件
            temp1 = time.strptime(line[2], '%Y-%m-%d %H:%M:%S')#\"%Y/%m/%d %H:%M:%S.%f\"  转变时间
            temp2 = time.strptime(front[2], '%Y-%m-%d %H:%M:%S')
            temp = datetime.fromtimestamp(time.mktime(temp1))-datetime.fromtimestamp(time.mktime(temp2)) # 活动消耗时间 没有地方用？
        else:
            temp = 0
        t = time.strptime(line[2], '%Y-%m-%d %H:%M:%S')
        week = datetime.fromtimestamp(time.mktime(t)).weekday()
        midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = datetime.fromtimestamp(time.mktime(t))-midnight
        data_new.append([line[0],vocabulary[str(line[1])],line[2],timesincemidnight,week]) # 事件序号、活动类型字典序号、活动时间(str)、计算后的时间、weekday
        front = line
    front = data_new[0]
    for row in range(1,len(data_new)):
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
        time_code_temp[key] =  sorted(time_code_temp[key])
    ##print('**************')
    data_merge = []
    data_temp = [data_new[0]]
    for line in data_new[1:]:  # 将data_new 里的活动按照事件分组得到 data_merge
        if line[0] != data_temp[-1][0]:  #当前活动是否和前一个活动属于同一个事件
            data_merge.append(data_temp)
            data_temp = [line]
        else:
            data_temp.append(line)
    data_merge.append(data_temp)
    vocabulary_num = len(vocabulary)

    vocabulary_temp = vocabulary
    return data_merge,data_new,time_code_temp,vocabulary_num,vocabulary_temp
data_merge,data_new,time_code_tmep,vocabulary_num,vocabulary = processData(data,vocabulary,eventlog)
print(data_new)
fit_data = []
for d in data_merge:
    this_d = []
    # print(1)
    for i in d:


        this_d.append(float(i[1]))
    fit_data.append(this_d)


# 初始化矩阵
def build_matirx(set_word):
    edge = len(set_word) + 1  # 建立矩阵，矩阵的高度和宽度为关键词集合的长度+1
    '''matrix = np.zeros((edge, edge), dtype=str)'''  # 另一种初始化方法
    matrix = [['' for j in range(edge)] for i in range(edge)]  # 初始化矩阵
    matrix[0][1:] = np.array(set_word)
    matrix = list(map(list, zip(*matrix)))
    print("*****")
    matrix[0][1:] = np.array(set_word)  # 赋值矩阵的第一行与第一列
    return matrix


matirx = build_matirx(list(vocabulary.values()))

# print(11)
print(len(matirx))
# 计算各个活动的共现次数
def count_matrix(matrix, formated_data):
    for row in range(1, len(matrix)):
        # 遍历矩阵第一行

        for col in range(1, len(matrix)):
            # 遍历矩阵第一列
            # print(row, col)
            if matrix[0][row] == matrix[col][0]:
                matrix[col][row] = 0
            else:
                counter = 0  # 初始化计数器
                for ech in formated_data:
                    if matrix[0][row] in ech and matrix[col][0] in ech:
                        counter += 1
                    else:
                        continue
                matrix[col][row] = counter
    return matrix


matirx = count_matrix(matirx, fit_data)
print("^^^^^^")
print(len(matirx))
fit_matirx = []
for i in range(1, len(matirx)):
    # print(i)
    fit_matirx.append(matirx[i][1:])

print(1)
from mittens import GloVe
import numpy as np
embed_size = 5
glove_model = GloVe(n=embed_size, max_iter=300,test_mode=True,learning_rate=0.1)

print(fit_matirx)
embeddings = glove_model.fit(np.array(fit_matirx))


corr = np.corrcoef(embeddings.dot(embeddings.T).ravel(), np.array(fit_matirx).ravel())[0][1]
print(corr)

f = open(eventlog + '.txt', 'w')
f.write('{} {}\n'.format(vocabulary_num, embed_size))
vectors = embeddings
for word in range(vocabulary_num):
    str_vec = ' '.join(map(str, list(vectors[int(word), :])))

    f.write('{} {}\n'.format(word, str_vec))
f.close()

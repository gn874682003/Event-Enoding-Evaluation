import numpy as np
import torch
import torch.nn as nn
import gensim
import torch.optim as optim
from torch.autograd import Variable
from torchqrnn import QRNN
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt



class qrnn5(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,out_size,batch_size=20,n_layer = 1, dropout = 0,
                 embedding = None):
        super(qrnn5, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout
        # self.weight_W = nn.Parameter(torch.Tensor(batch_size, hidden_dim * 2, hidden_dim * 2).cuda()).cuda()
        # self.weight_Mu = nn.Parameter(torch.Tensor(hidden_dim * 2, n_layer).cuda()).cuda()
        # self.rnn = QRNN(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout,
        #                    num_layers=self.n_layer, bidirectional=True)
        self.rnn1 = QRNN(input_size=embedding_dim, hidden_size=hidden_dim, dropout=self.dropout,
                         num_layers=self.n_layer)

        self.out = nn.Linear(hidden_dim, 1)
    ####加入注意力机制
    # def attention_net(self, rnn_output):
    #
    #     attn_weights = torch.matmul(rnn_output,self.weight_Mu).cuda()
    #     print(attn_weights.size())
    #     soft_attn_weights = F.softmax(attn_weights, 1)
    #     context = torch.bmm(rnn_output.transpose(1, 2), soft_attn_weights).squeeze(2).cuda()
    #     return context, soft_attn_weights.data.cpu().numpy()  # context : [batch_size, hidden_dim * num_directions(=2)]
    def forward(self, X):
        input = self.embedding(X)
        # print(input.shape)
        # print(input)
        input = input.permute(1, 0, 2)
        # print(input)
        # exit()
        hidden_state = Variable(
            torch.randn(self.n_layer, self.batch_size, self.hidden_dim))
        cell_state = Variable(
            torch.randn(self.n_layer, self.batch_size, self.hidden_dim))
        # print(type(input))
        # input.cpu()
        # print(type(input))
        # output, _ = self.rnn(input, (hidden_state, cell_state))
        #
        output1, _ = self.rnn1(input, (hidden_state, cell_state))

        hn = output1[-1]
        # print(hn.shape)
        # temp=input()
        # exit()
        output = self.out(hn)
        # print(output.shape)
        # exit()
        return  output # model : [batch_size, num_classes], attention : [batch_size, n_step]
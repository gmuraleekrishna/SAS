from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import logging
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

def from_numpy(states):
    return [Variable(torch.from_numpy(state)).cuda() for state in states]


def get_GloVe(text, size, vectors, aggregation='mean'):
    vec = np.zeros(size).reshape((1, size))  # create for size of glove embedding and assign all values 0
    count = 0
    for word in text.split():
        print('\n word:   ', word)
        try:
            print('glove vect', vectors[word])
            vec += vectors[word].reshape((1, size))  # update vector with new word
            count += 1  # counts every word in sentence
        except KeyError:
            continue
    if aggregation == 'mean':
        if count != 0:
            vec /= count  # get average of vector to create embedding for sentence
        return vec
    elif aggregation == 'sum':
        return vec


class CNNRewardModel(nn.Module):
    def __init__(self, vocab_size, feat_size, dropout, activation):
        super(CNNRewardModel, self).__init__()
        self.vocab_size = vocab_size
        self.word_embed_dim = 300
        self.feat_size = feat_size
        self.kernel_num = 512
        self.kernels = [2, 3, 4, 5]
        self.out_dim = len(self.kernels) * self.kernel_num + self.word_embed_dim

        self.emb = nn.Embedding(self.vocab_size, self.word_embed_dim)
        self.emb.weight.data.copy_(torch.from_numpy(np.load('./data/pkls/glove_vocab_embedding.npy')))

        self.proj = nn.Linear(36*feat_size, self.word_embed_dim)

        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (k, self.word_embed_dim)) for k in self.kernels])

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(self.out_dim, 1, bias=True)

        if activation.lower() == "linear":
            self.activation = None
        elif activation.lower() == "sign":
            self.activation = nn.Softsign()
        elif activation.lower() == "tahn":
            self.activation = nn.Tanh()
        elif activation.lower() == "sm":
            self.activation = nn.Softmax()
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, instruction, img_feats, instruction_lens=None):

        embedding = Variable(self.emb(instruction).data)  # (batch_size, seq_length, embed_dim)
        # batch x seq_len x emb_dim -> batch x 1 x seq_len x emb_dim
        # embedding = embedding.unsqueeze(1)
        x = [F.relu(conv(embedding)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        # combine with image feature
        img = self.proj(img_feats)
        combined = torch.cat([x, img], 1)
        combined = self.dropout(combined)

        prob = self.fc(combined).view(-1)
        return self.activation(prob)


class GRURewardModel(nn.Module):

    def __init__(self, vocab_size, feat_size, dropout, activation):

        super(GRURewardModel, self).__init__()

        self.vocab_size = vocab_size
        self.word_embed_dim = 300
        self.gru_units = 2
        self.feat_size = feat_size
        self.kernel_num = 512
        self.kernels = [2, 3, 4, 5]
        self.out_dim = len(self.kernels) * self.kernel_num + self.word_embed_dim

        self.emb = nn.Embedding(self.vocab_size, self.word_embed_dim)
        self.emb.weight.data.copy_(torch.from_numpy(np.load('./data/pkls/glove_vocab_embedding.npy')))

        self.gru = nn.GRU(input_size=self.word_embed_dim, hidden_size=self.gru_units, batch_first=True)

        self.fc1 = nn.Linear(36 * feat_size, self.word_embed_dim)
        self.fc2 = nn.Linear(self.gru_units, 1)
        self.dropout = nn.Dropout(dropout)
        if activation.lower() == "linear":
            self.activation = None
        elif activation.lower() == "sign":
            self.activation = nn.Softsign()
        elif activation.lower() == "tahn":
            self.activation = nn.Tanh()
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, instruction, img_feats, instruction_lens=None):
        img_feats = self.fc1(img_feats)
        embeddings = Variable(self.emb(instruction).data)
        inputs = torch.cat((img_feats.unsqueeze(1), embeddings.squeeze(1)), 1)
        outputs, _ = self.gru(inputs)
        outputs = self.dropout(outputs)
        pred = self.activation(self.fc2(outputs))
        return pred.squeeze(-1)

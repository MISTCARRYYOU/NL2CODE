import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import nltk


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def prepareData(self, lang1, lang2):

        input_lang, output_lang, pairs = self.readLangs(lang1, lang2)
        print("Read %s sentence pairs" % len(pairs))
        print("Trimmed to %s sentence pairs" % len(pairs))

        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])

        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)

        return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)  # good size for gru input
        output = embedded

        output, hidden = self.gru(output, hidden)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)  # init hidden state for first word


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device='cpu')


def readLangs(lang1, lang2):
    print("Reading lines...")

    line_1 = open('../dataset/data_conala/train/%s' % (lang1), encoding='utf-8'). \
        read().strip().split('\n')

    line_2 = open('../dataset/data_conala/train/%s' % (lang2), encoding='utf-8'). \
        read().strip().split('\n')

    pairs = [*zip(line_1, line_2)]

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


if __name__ == '__main__':
    input_lang, output_lang, pairs = readLangs('conala-train.intent', 'conala-train.snippet')
    # nl = Lang('natural language')
    # fl = Lang('formal language')

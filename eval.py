import random

import re

import torch
import torch.nn as nn
from torch import optim
from torchtext.data.metrics import bleu_score

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset.json_to_seq2seq import *
from model.seq2seq_withoutattention import EncoderRNN, DecoderRNN

from nltk import bleu


def evaluate(encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size(0)

        encoder_hidden = encoder.init_hidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(100):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)

            topv, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:
                #decoded_words.append('<EOS>')
                break

            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


input_lang, output_lang, training_pairs = data_tensor('train')

hidden_size = 256

encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words)

encoder1.load_state_dict(torch.load('SAVED_MODEL_ENCODER.pt'))
decoder1.load_state_dict(torch.load('SAVED_MODEL_DECODER.pt'))

sentences = open('./dataset/data_conala/train/conala-train.intent', encoding='utf-8'). \
    read().strip().split('\n')

prediction = []

for sentence in sentences:
    output_words = evaluate(encoder1,
                            decoder1,
                            sentence)
    prediction.append(output_words)

with open('./conala.prediction', 'w') as predicted:
    for code in prediction:
        predicted.write(' '.join(code) + '\n')

prediction = open('./conala.prediction', encoding='utf-8'). \
    read().strip().split('\n')

reference = open('./dataset/data_conala/train/conala-train.snippet', encoding='utf-8'). \
    read().strip().split('\n')

reference = [[x.split(' ')] for x in reference]
prediction = [x.split(' ') for x in prediction]

print(bleu_score(prediction, reference))

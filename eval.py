import random

import torch
import torch.nn as nn
from torch import optim

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset.json_to_seq2seq import *
from model.seq2seq import EncoderRNN, DecoderRNN


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
                decoded_words.append('<EOS>')
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

output_words = evaluate(encoder1,
                        decoder1,
                        'prepend string \' hello \' to all items in list \' a \'')

print('input =', 'prepend string \' hello \' to all items in list \' a \'')
print('output =', ' '.join(output_words))

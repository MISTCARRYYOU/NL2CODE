import random

import torch
import torch.nn as nn
from torch import optim

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset.json_to_seq2seq import *
from model.seq2seq import EncoderRNN, DecoderRNN


def train(input_tensor, target_tensor,
          encoder, decoder,
          encoder_optimizer, decoder_optimizer,
          criterion):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]])

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


input_lang, output_lang, training_pairs = data_tensor('train')

teacher_forcing_ratio = 0.7

plot_losses = []
print_loss_total = 0
plot_loss_total = 0

hidden_size = 256
epochs = 15

encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words)

encoder_optimizer = optim.SGD(encoder1.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder1.parameters(), lr=0.01)

criterion = nn.NLLLoss()

for epoch in range(1, epochs):
    print('epoch - %d over - %d ' % (epoch, epochs))
    for iter in range(1, len(training_pairs)):

        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor,
                     encoder1, decoder1,
                     encoder_optimizer, decoder_optimizer,
                     criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % 100 == 0:
            print_loss_avg = print_loss_total / 100
            print_loss_total = 0
            print('iteration - %d loss - %.4f' % (iter, print_loss_avg))

torch.save(encoder1.state_dict(), 'SAVED_MODEL_ENCODER.pt')
torch.save(decoder1.state_dict(), 'SAVED_MODEL_DECODER.pt')

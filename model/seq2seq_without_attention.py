import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    """
    Basic encoder with GRU
    """
    def __init__(self, input_size, embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size  # size of encoder output

        self.embedding = nn.Embedding(input_size, embedding_size)  # input size of vocabulary

        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1, 1, -1)  # good size for gru input

        output = embedded

        output, hidden = self.gru(output, hidden)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)  # init hidden state for first word


class DecoderRNN(nn.Module):
    """
    Basic decoder with gru
    """
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device='cpu')


if __name__ == '__main__':
    pass

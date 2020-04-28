class EncoderRNN(nn.Module):
    """
    Encoder with bi-GRU and dropout
    """

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.0):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size  # size of encoder output

        self.embedding = nn.Embedding(input_size, hidden_size)

        self.num_layers = num_layers

        self.dropout = dropout

        # Bidirectional GRU
        self.gru = nn.GRU(hidden_size, hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)  # matrice d'embedding

        # Pack the sequence of embeddings
        # packed_embeddings = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)  # pour les batchs

        # Run the packed embeddings through the GRU, and then unpack the sequences
        outputs, hidden = self.gru(embedded)
        # outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # The ouput of a GRU has shape (seq_len, batch, hidden_size * num_directions)
        # Because the Encoder is bidirectional, combine the results from the
        # forward and reversed sequence by simply adding them together.
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs

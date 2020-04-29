class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "OOV"}
        self.n_words = 3  # Count SOS and EOS

    def add_sentence(self, sentence):
        """
        Creation of vocabulary for each input
        :param sentence: NL input of NN
        :return:
        """
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """
        Vocabulary creation for each NL token
        :param word: NL token
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


if __name__ == '__main__':
    pass

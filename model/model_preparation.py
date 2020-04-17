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



from __future__ import print_function

import json
import sys

from model.vocabulary import Lang
import torch

path_train = './data_conala/train/'
path_test = './data_conala/test/'
path_mined = './data_conala/mined/'

SOS_token = 0
EOS_token = 1

def read_langs(lang1, lang2, mode):
    print("Reading lines...")

    line_1 = open('../dataset/data_conala/%s/%s' % (mode, lang1), encoding='utf-8'). \
        read().strip().split('\n')

    line_2 = open('../dataset/data_conala/%s/%s' % (mode, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    pairs = [*zip(line_1, line_2)]

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs


def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]

    indexes.append(EOS_token)

    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])

    return (input_tensor, target_tensor)

def main():
    argument = [('./data_conala/conala-corpus/conala-train.json.seq2seq', path_train + 'conala-train.intent',
                 path_train + 'conala-train.snippet'),
                ('./data_conala/conala-corpus/conala-test.json.seq2seq', path_test + 'conala-test.intent',
                 path_test + 'conala-test.snippet'),
                ('./data_conala/conala-corpus/conala-mined.jsonl.seq2seq', path_mined + 'conala-mined.intent',
                 path_mined + 'conala-mined.snippet')]

    for arg in argument:
        json_file = arg[0]
        seq_input = arg[1]
        seq_output = arg[2]

        dataset = json.load(open(json_file))
        with open(seq_input, 'w') as f_inp, open(seq_output, 'w') as f_out:
            for example in dataset:
                f_inp.write(' '.join(example['intent_tokens']) + '\n')
                f_out.write(' '.join(example['snippet_tokens']) + '\n')

    input_lang, output_lang, pairs = read_langs('conala-train.intent', 'conala-train.snippet', 'train')

    training_pairs = [tensorsFromPair(input_lang, output_lang, x) for x in pairs]

    print(training_pairs)

if __name__ == '__main__':
    main()
from __future__ import print_function

import json

from dataset.vocabulary import Lang
import torch


SOS_token = 0
EOS_token = 1


def path_mode(mode):
    if mode is 'mined':
        path_seq2seq = './data_conala/conala-corpus/conala-%s.jsonl.seq2seq' % mode
    else:
        path_seq2seq = './data_conala/conala-corpus/conala-%s.json.seq2seq' % mode
    path_input = './data_conala/{0}/conala-{0}.intent'.format(mode)
    path_output = './data_conala/{0}/conala-{0}.snippet'.format(mode)
    return path_seq2seq, path_input, path_output


def read_langs(lang1, lang2, mode):
    print("Reading lines...")

    line_1 = open('./dataset/data_conala/%s/%s' % (mode, lang1), encoding='utf-8'). \
        read().strip().split('\n')

    line_2 = open('./dataset/data_conala/%s/%s' % (mode, lang2), encoding='utf-8'). \
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


def json_merge(json_train, number_of_examples=20000):
    json_mined, _, _ = path_mode('mined')

    data_mined = json.load(open(json_mined))
    data_train = json.load(open(json_train))
    for i in range(number_of_examples):
        data_train.append(data_mined[i])

    return data_train


def data_creation(mode, merge=False):
    json_file, seq_input, seq_output = path_mode(mode)
    if mode is 'train' and merge is True:  # merge mined and train examples
        dataset = json_merge(json_file)
    else:
        dataset = json.load(open(json_file))

    with open(seq_input, 'w') as f_inp, open(seq_output, 'w') as f_out:
        for example in dataset:
            f_inp.write(' '.join(example['intent_tokens']) + '\n')
            f_out.write(' '.join(example['snippet_tokens']) + '\n')


def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] if word in lang.word2index.keys()
               else 2 for word in sentence.split(' ')]

    indexes.append(EOS_token)

    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)  # transform index into tensor


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])

    return (input_tensor, target_tensor)



def data_tensor(mode):
    input_lang, output_lang, pairs = read_langs('conala-%s.intent' % mode, 'conala-%s.snippet' % mode , mode)

    training_pairs = [tensorsFromPair(input_lang, output_lang, x) for x in pairs]

    return input_lang, output_lang, training_pairs


if __name__ == '__main__':
    data_creation('train', True)
    #  data_creation('test')

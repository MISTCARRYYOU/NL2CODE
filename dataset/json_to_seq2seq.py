from __future__ import print_function

import json
import sys

path_train = './data_conala/train/'
path_test = './data_conala/test/'
path_mined = './data_conala/mined/'

def main():
    argument = [('./data_conala/conala-train.json.seq2seq', path_train + 'conala-train.intent',
                 path_train + 'conala-train.snippet'),
                ('./data_conala/conala-test.json.seq2seq', path_test + 'conala-test.intent',
                 path_test + 'conala-test.snippet'),
                ('./data_conala/conala-mined.jsonl.seq2seq', path_mined + 'conala-mined.intent',
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


if __name__ == '__main__':
    main()
from __future__ import print_function

import json
import sys

def main():
    argument = [('./data_conala/conala-train.json.seq2seq', './data_conala/conala-train.intent', './data_conala/conala-train.snippet'),
                ('./data_conala/conala-test.json.seq2seq', './data_conala/conala-test.intent', './data_conala/conala-test.snippet')]

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
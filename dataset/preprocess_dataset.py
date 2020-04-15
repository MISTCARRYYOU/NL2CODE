import json
import sys
import traceback

import nltk
# nltk.download('punkt')

from dataset.util import get_encoded_code_tokens

if __name__ == '__main__':
    path = './data_conala/'

    for file_path in [(path + 'conala-train.json'), (path + 'conala-test.json')]:
        print('file {}'.format(file_path), file=sys.stderr)

        dataset = json.load(open(file_path))

        for i, example in enumerate(dataset):
            intent = example['intent']
            rewritten_intent = example['rewritten_intent']
            snippet = example['snippet']

            if rewritten_intent:
                try:
                    intent_tokens = nltk.word_tokenize(rewritten_intent)
                    encoded_reconstr_code = get_encoded_code_tokens(snippet)
                except:
                    print('*' * 20, file=sys.stderr)
                    print(i, file=sys.stderr)
                    print(intent, file=sys.stderr)
                    print(snippet, file=sys.stderr)
                    traceback.print_exc()

                    failed = True
            if not intent_tokens:
                intent_tokens = nltk.word_tokenize(intent)

            example['intent_tokens'] = intent_tokens
            example['snippet_tokens'] = encoded_reconstr_code

        json.dump(dataset, open(file_path + '.seq2seq', 'w'), indent=2)

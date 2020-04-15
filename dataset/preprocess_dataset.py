import json
import sys
import traceback

import nltk
# nltk.download('punkt')

from dataset.canonicalize import *
from dataset.util import get_encoded_code_tokens

if __name__ == '__main__':
    file_path = './data_conala/'

    for file_path in [(file_path + 'conala-train.json'), (file_path + 'conala-test.json')]:
        print('file {}'.format(file_path), file=sys.stderr)

        dataset = json.load(open(file_path))

        for i, example in enumerate(dataset):
            intent = example['intent']
            rewritten_intent = example['rewritten_intent']
            snippet = example['snippet']
            intent_tokens = []

            if rewritten_intent:
                try:
                    canonical_intent, slot_map = canonicalize_intent(rewritten_intent)
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
            example['snippet_tokens'] = snippet

        json.dump(dataset, open(file_path + '.seq2seq', 'w'), indent=2)

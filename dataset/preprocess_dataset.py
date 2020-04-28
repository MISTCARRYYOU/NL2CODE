import json
import sys
import traceback

import nltk

nltk.download('punkt')

from dataset.util import get_encoded_code_tokens

path = './dataset/data_conala/conala-corpus/'

for file_path, file_type in [(path + 'conala-train.json', 'annotated'),
                             (path + 'conala-test.json', 'annotated'),
                             (path + 'conala-mined.jsonl', 'mined')]:
    print('file {}'.format(file_path), file=sys.stderr)

    if file_type == 'annotated':
        dataset = json.load(open(file_path))
    elif file_type == 'mined':
        dataset = []
        with open(file_path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line.strip()))

    for i, example in enumerate(dataset):
        intent = example['intent']
        snippet = example['snippet']

        if file_type == 'annotated':
            rewritten_intent = example['rewritten_intent']
        elif file_type == 'mined':
            rewritten_intent = example['intent']

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

        if rewritten_intent is None:
            encoded_reconstr_code = get_encoded_code_tokens(snippet.strip())

        example['intent_tokens'] = intent_tokens
        example['snippet_tokens'] = encoded_reconstr_code

    json.dump(dataset, open(file_path + '.seq2seq', 'w'), indent=2)

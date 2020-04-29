from __future__ import print_function

import token as tk
from tokenize import generate_tokens

from io import StringIO


def get_encoded_code_tokens(code):
    """
    Transform code string into a tokenized form
    :param code: snippet of code from CoNaLa
    :return: list of tokenized code
    """
    code = code.strip() #remove space at the beginning/end
    token_stream = generate_tokens(StringIO(code).readline) #tokenize code between OP/NAME/NEWLINE/INDENT
    tokens = []
    indent_level = 0
    new_line = False

    for toknum, tokval, _, _, _ in token_stream:
        if toknum == tk.NEWLINE:
            tokens.append('#NEWLINE#')
            new_line = True
        elif toknum == tk.INDENT:
            indent_level += 1
        elif toknum == tk.STRING:
            tokens.append(tokval.replace(' ', '#SPACE#').replace('\t', '#TAB#').replace('\r\n', '#NEWLINE#').replace('\n', '#NEWLINE#'))
        elif toknum == tk.DEDENT:
            indent_level -= 1
        else:
            tokval = tokval.replace('\n', '#NEWLINE#')
            if new_line:
                for i in range(indent_level):
                    tokens.append('#INDENT#')

            new_line = False
            tokens.append(tokval)

    # remove ending None
    if len(tokens[-1]) == 0:
        tokens = tokens[:-1]

    if '\n' in tokval:
        pass

    return tokens





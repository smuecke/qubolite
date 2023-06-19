import re

import numpy as np
from numpy.typing import ArrayLike


def all_bitvectors(n: int):
    x = np.zeros(n)
    b = 2**np.arange(n)
    for k in range(1<<n):
        x[:] = (k & b) > 0
        yield x
   
    
def all_bitvectors_array(n):
    return np.arange(1<<n)[:, np.newaxis] & (1<<np.arange(n)) > 0


def from_string(string: str):
    return np.fromiter(string, dtype=np.float64)


def to_string(bitvec: ArrayLike):
    if bitvec.ndim <= 1:
        return ''.join(str(int(x)) for x in bitvec)
    else:
        return np.apply_along_axis(to_string, -1, bitvec)
    

_BITVEC_EXPR = re.compile(r'[01*]|\[!?\d+\]')

def _expr_normal_form(tokens):
    # re-arrange tokens so that all references
    # point to previous indices
    i = 0
    while i < len(tokens):
        if not tokens[i].startswith('['):
            i += 1
            continue
        j = int(tokens[i].strip('[!]'))
        if tokens[j].startswith('['):
            raise RuntimeError('No references to references allowed!')
        if j > i:
            if tokens[i].startswith('[!'):
                tokens[i] = '[!' + str(i) + ']'
                tokens[j] = '10*'['01*'.index(tokens[j])]
            else:
                tokens[i] = '[' + str(i) + ']'
            # swap tokens
            tokens[i], tokens[j] = tokens[j], tokens[i]
            # check if constant needs to be inverted
        i += 1
    return tokens

def from_expression(expr: str):
    # '01**1****0[2][!3]**'
    tokens = _expr_normal_form(list(_BITVEC_EXPR.findall(expr)))
    n = len(tokens)
    n_free = expr.count('*')
    x = np.empty((1<<n_free, n))
    power = 0
    for i, token in enumerate(tokens):
        if token in ['0', '1']:
            x[:, i] = float(token)
        elif token == '*':
            x[:, i] = np.arange(1<<n_free) & (1<<power) > 0
            power += 1
        else:
            ix = int(token.strip('[!]'))
            if token[1] == '!': # inverse reference
                x[:,i] = 1-x[:,ix]
            else:
                x[:,i] = x[:,ix]
    return x

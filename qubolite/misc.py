from hashlib  import md5
from warnings import warn

import numpy as np
from numpy.random import RandomState


def is_symmetrical(arr):
    return NotImplemented


def is_triu(arr):
    return np.all(np.isclose(arr, np.triu(arr)))


def min_max(it):
    min_ = float('inf')
    max_ = float('-inf')
    for x in it:
        if x < min_: min_ = x
        if x > max_: max_ = x
    return min_, max_


def warn_size(n: int, limit: int=20):
    if n > limit:
        warn('This operation may take a very long time for n>{limit}.')


def get_random_state(state=None):
    if isinstance(state, RandomState):
        return state
    if state is None:
        return RandomState()
    try:
        seed = int(state)
    except ValueError:
        # use hash digest when seed is a (non-numerical) string
        seed = int(md5(state.encode('utf-8')).hexdigest(), 16) & 0xffffffff
    return RandomState(seed)


def set_suffix(filename, suffix):
    s = suffix.strip(' .')
    if filename.lower().endswith('.'+s.lower()):
        return filename
    else:
        return f'{filename}.{s}'


def all_bitvectors(n: int, read_only=True):
    x = np.zeros(n, dtype=np.float64)
    while True:
        yield x if read_only else x.copy()
        pointer = 0
        while x[pointer] > 0:
            x[pointer] = 0
            pointer += 1
            if pointer >= n:
                return
        x[pointer] = 1


def bitvector_from_string(string):
    return np.fromiter(string, dtype=np.float64)


def bitvector_to_string(bitvec):
    return ''.join(str(int(x)) for x in bitvec)
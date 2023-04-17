import numpy as np
from numpy.typing import ArrayLike


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


def all_bitvectors_fast(n, as_int=False):
    B = np.arange(1<<n)[:, np.newaxis] & (1<<np.arange(n)) > 0
    if as_int:
        B = B.astype(int)
    return B


def from_string(string: str):
    return np.fromiter(string, dtype=np.float64)


def to_string(bitvec: ArrayLike):
    if bitvec.ndim <= 1:
        return ''.join(str(int(x)) for x in bitvec)
    else:
        return np.apply_along_axis(to_string, -1, bitvec)
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
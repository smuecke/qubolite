from itertools import combinations
from warnings  import warn

import bitvec
import numpy as np
from seedpy import get_random_state

from .misc import is_triu, min_max


def is_qubo_like(arr):
    if arr.ndim >= 2:
        u, v = arr.shape[-2:]
        return u == v
    else:
        return False


def to_triu_form(arr):
    if is_triu(arr):
        return arr
    else:
        # add lower to upper triangle
        return np.triu(arr + np.tril(arr, -1).T)


class qubo:

    def __init__(self, m: np.ndarray):
        assert is_qubo_like(m)
        self.m = to_triu_form(m)
        self.n = m.shape[-1]

    def __repr__(self):
        return 'qubo'+self.m.__repr__().lstrip('array')

    def __call__(self, x: np.ndarray):
        return np.sum(np.dot(x, self.m)*x, axis=-1)

    def copy(self):
        return qubo(self.m.copy())


    @classmethod
    def random(cls, n: int, distr='normal', density=1.0, random_state=None, **kwargs):
        npr = get_random_state(random_state)
        if distr == 'normal':
            arr = npr.normal(
                kwargs.get('loc', 0.0),
                kwargs.get('scale', 1.0)/2,
                size=(n, n))
        elif distr == 'uniform':
            arr = npr.uniform(
                kwargs.get('low', -1.0),
                kwargs.get('high', 1.0),
                size=(n, n))/2
        else:
            raise ValueError(f'Unknown distribution "{distr}"')
        m = np.triu(arr+arr.T)
        if density < 1.0:
            m *= npr.binomial(1, density, size=m.shape)
        return cls(m)


    def to_file(self, path: str):
        return NotImplemented

    @classmethod
    def from_file(cls, path: str):
        return NotImplemented


    def to_dict(self, names=None, double_indices=True):
        if names is None:
            names = { i: i for i in range(self.n) }
        qubo_dict = dict()
        for i, j in zip(*np.triu_indices_from(self.m)):
            if not np.isclose(self.m[i, j], 0):
                if (i == j) and (not double_indices):
                    qubo_dict[(names[i],)] = self.m[i, i]
                else:
                    qubo_dict[(names[i], names[j])] = self.m[i, j]
        return qubo_dict

    @classmethod
    def from_dict(cls, qubo_dict):
        names = { name: i for i, name in enumerate(sorted(qubo_dict.keys())) }
        n = max(names.values())+1
        m = np.zeros((n, n))
        for k, v in qubo_dict.items():
            match k:
                case i, j: m[i, j] += v
                case i,  : m[i, i] += v
                case _   : pass
        m = np.triu(m + np.tril(m, -1).T)
        return cls(m)


    def brute_force(self):
        if self.n >= 20: warn('Brute-forcing QUBOs with n>=20 might take a very long time')
        return min(bitvec.all(self.n, read_only=False), key=self)


    def clamp(self, partial_assignment=None):
        if partial_assignment is None:
            return self.copy(), 0, set(range(self.n))
        ones = list(sorted({i for i, b in partial_assignment.items() if b == 1}))
        free = list(sorted(set(range(self.n)).difference(partial_assignment.keys())))
        R = self.m.copy()
        const = R[ones, :][:, ones].sum()
        for i in free:
            R[i, i] += sum(R[l, i] if l<i else R[i, l] for l in ones)
        return qubo(R[free,:][:,free]), const, free


    def dx(self, x: np.ndarray):
        # 1st discrete derivatice
        m_  = np.triu(self.m, 1)
        m_ += m_.T
        sign = 1-2*x
        return sign*(np.diag(self.m)+(m_*x).sum(1))


    def dx2(self, x: np.ndarray):
        # 2nd discrete derivative
        return NotImplemented


    def dynamic_range(self, bits=False):
        min_, max_ = min_max(
            abs(u-v) for u, v in combinations(
                np.r_[self.m[np.triu_indices_from(self.m)], 0], r=2) if not np.isclose(u, v))
        r = max_/min_
        return np.log2(r) if bits else 20*np.log10(r)


    def absmax(self):
        return np.max(np.abs(self.m))

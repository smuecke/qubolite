from collections import Counter, defaultdict
from functools   import cached_property

import bitvec
import numpy as np
from seedpy import get_random_state


class QUBOSample:

    def __init__(self, *, counts: dict[str, int]=None, raw: np.ndarray=None):
        if counts is not None:
            self.counts = counts
        elif raw is not None:
            C = Counter([bitvec.to_string(x) for x in raw])
            self.counts = dict(C)
        else:
            raise ValueError('Provide counts or raw sample data!')

    @cached_property
    def n(self):
        n = len(next(iter(self.counts)))
        assert all(len(x)==n for x in self.counts.keys())
        return n

    @cached_property
    def shots(self):
        return sum(self.counts.values())

    @cached_property
    def raw(self):
        X = np.empty((self.shots, self.n))
        pointer = 0
        for x, k in self.counts.items():
            X[pointer:pointer+k, :] = np.tile(bitvec.from_string(x), (k, 1))
            pointer += k
        return X

    @cached_property
    def suff_stat(self):
        X = self.raw
        return np.triu(X.T @ X)

    def hellinger_distance(self, other):
        assert self.n == other.n
        xs = set(self.counts.keys())
        xs.update(other.counts.keys())
        xs = list(xs)
        p1 = np.asarray([self.counts.get(x, 0) for x in xs], dtype=np.float64)/self.shots
        p2 = np.asarray([other.counts.get(x, 0) for x in xs], dtype=np.float64)/other.shots
        return np.linalg.norm(np.sqrt(p1)-np.sqrt(p2))/np.sqrt(2.0)


def generate_num_flips(λ=1.0, random_state=None):
    npr = get_random_state(random_state)
    flip_list = []
    while True:
        if flip_list:
            yield flip_list.pop()
        else:
            ks = npr.poisson(1, size=100)
            flip_list.extend(ks[ks>0])


def full(qubo, samples: int=1, temp=1.0, random_state=None):
    npr = get_random_state(random_state)
    X = np.vstack(list(bitvec.all(qubo.n, read_only=False)))
    p = np.exp(-qubo(X)/temp)
    p = p / p.sum()
    vals = npr.choice(2**qubo.n, p=p, size=samples)
    C = Counter(vals)
    fmt = f'0{qubo.n}b'
    counts = { format(i, fmt)[::-1]: k for i, k in C.items() }
    return QUBOSample(counts=counts)

def gibbs(qubo, samples: int=1, burn_in=1000, initial=None, temp=1.0, random_state=None):
    npr = get_random_state(random_state)
    counts = defaultdict(int)
    x = initial if initial is not None else npr.binomial(1, 0.5, size=qubo.n)
    for t in range(burn_in+samples):
        exp_dx = np.exp(qubo.dx(x)*(2*x-1)/temp)
        p = exp_dx/(exp_dx+1)
        x = npr.binomial(1, p=p)
        if t >= burn_in:
            counts[bitvec.to_string(x)] += 1
    return QUBOSample(counts=counts)

        
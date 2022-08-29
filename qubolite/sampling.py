import struct
from collections import Counter, defaultdict
from functools   import cached_property

import numpy as np
from numba  import njit

from .misc import all_bitvectors, bitvector_from_string, bitvector_to_string, get_random_state, set_suffix


class BinarySample:

    def __init__(self, *, counts=None, raw: np.ndarray=None):
        if counts is not None:
            self.counts = counts
        elif raw is not None:
            C = Counter([bitvector_to_string(x) for x in raw])
            self.counts = dict(C)
        else:
            raise ValueError('Provide counts or raw sample data!')

    def save(self, filename):
        f = open(set_suffix(filename, 'sample'), 'wb')
        f.write(struct.pack('<I', self.n))
        max_count = max(self.counts.values())
        fmt = 'B' if max_count<(1<<8) else 'H' if max_count<(1<<16) else 'I'
        f.write(struct.pack('c', fmt.encode()))
        b = int(np.ceil(self.n/8))
        for x, k in self.counts.items():
            f.write(int.to_bytes(int(x[::-1], base=2), length=b, byteorder='little', signed=False))
            f.write(struct.pack(f'<{fmt}', k))
        f.close()

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = f.read()
        n, = struct.unpack('<I', data[:4])
        fmt, = struct.unpack('c', data[4:5])
        fmt = fmt.decode()
        b = int(np.ceil(n/8))
        l = struct.calcsize(fmt)
        counts = dict()
        for offset in range(5, len(data), b+l):
            i = int.from_bytes(data[offset:offset+b], byteorder='little', signed=False)
            x = format(i, f'0{n}b')[::-1]
            k, = struct.unpack(f'<{fmt}', data[offset+b:offset+b+l])
            counts[x] = k
        return cls(counts=counts)

    @cached_property
    def n(self):
        n = len(next(iter(self.counts)))
        assert all(len(x)==n for x in self.counts.keys())
        return n

    @cached_property
    def size(self):
        return sum(self.counts.values())

    @cached_property
    def raw(self):
        X = np.empty((self.size, self.n))
        pointer = 0
        for x, k in self.counts.items():
            X[pointer:pointer+k, :] = np.tile(bitvector_from_string(x), (k, 1))
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
        p1 = np.asarray([self.counts.get(x, 0) for x in xs], dtype=np.float64)/self.size
        p2 = np.asarray([other.counts.get(x, 0) for x in xs], dtype=np.float64)/other.size
        return np.linalg.norm(np.sqrt(p1)-np.sqrt(p2))/np.sqrt(2.0)

    def subsample(self, size: int, random_state=None):
        npr = get_random_state(random_state)
        xs = list(sorted(self.counts.keys())) # sort for reproducibility
        cumcs = np.cumsum(np.asarray([self.counts[x] for x in xs]))
        mask = npr.permutation(self.size) < size
        counts = dict()
        for u, v, x in zip(np.r_[0, cumcs], cumcs, xs):
            c = mask[u:v].sum()
            if c > 0:
                counts[x] = c
        return BinarySample(counts=counts)

    def most_frequent(self):
        return max(self.counts, key=self.counts.get)

    def empirical_prob(self, x=None):
        if x is None:
            return self.__emp_prob
        c = self.counts.get(x, 0)
        return c/self.size

    @cached_property
    def __emp_prob(self):
        P = np.zeros(1<<self.n)
        for x, c in self.counts.items():
            i = int(x[::-1], base=2)
            P[i] += c/self.size
        assert np.isclose(P.sum(), 1.0)
        return P


def full(qubo, samples: int=1, temp=1.0, random_state=None):
    npr = get_random_state(random_state)
    X = np.vstack(list(all_bitvectors(qubo.n, read_only=False)))
    p = np.exp(-qubo(X)/temp)
    p = p / p.sum()
    vals = npr.choice(2**qubo.n, p=p, size=samples)
    C = Counter(vals)
    fmt = f'0{qubo.n}b'
    counts = { format(i, fmt)[::-1]: k for i, k in C.items() }
    return BinarySample(counts=counts)


def mcmc(qubo, samples: int=1, burn_in=1000, initial=None, temp=1.0, random_state=None):
    npr = get_random_state(random_state)
    counts = defaultdict(int)
    x = initial if initial is not None else (npr.random(qubo.n)<0.5).astype(np.float64)
    for t in range(burn_in+samples):
        exp_dx = np.exp(qubo.dx(x)*(2*x-1)/temp)
        p = exp_dx/(exp_dx+1)
        x = npr.binomial(1, p=p)
        if t >= burn_in:
            counts[bitvector_to_string(x)] += 1
    return BinarySample(counts=dict(counts))


@njit
def _marginal(qm, x, i, temp=1.0):
    dxi = qm[i,i]+(x[:i]*qm[:i,i]).sum()+(x[i+1:]*qm[i,i+1:]).sum()
    if x[i] == 0:
        e0 = x @ qm @ x
        e1 = e0+dxi
    else:
        e1 = x @ qm @ x
        e0 = e1-dxi
    p0 = np.exp(-e0/temp)
    p1 = np.exp(-e1/temp)
    return p1/(p0+p1)


def gibbs(qubo, samples: int=1, burn_in: int=1000, keep_interval: int=10, initial=None, temp=1.0, random_state=None):
    npr = get_random_state(random_state)
    counts = defaultdict(int)
    x = initial if initial is not None else (npr.random(qubo.n)<0.5).astype(np.float64)
    sampled = -burn_in
    skip = 0
    while sampled < samples:
        # iterate over all indices in random order
        for i in npr.permutation(qubo.n):
            # get marginal Bernoulli probability for component i
            p = _marginal(qubo.m, x, i, temp)
            # sample with given probability
            x[i] = 1 if npr.random() < p else 0
        if sampled >= 0:
            if skip <= 0:
                counts[bitvector_to_string(x)] += 1
                sampled += 1
                skip = keep_interval-1
            else:
                skip -= 1
        else:
            # still in burn-in phase -> discard sample
            sampled += 1
    return BinarySample(counts=dict(counts))

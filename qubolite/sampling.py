import struct
from collections import Counter, defaultdict
from functools   import cached_property

import numpy as np

from .bitvec import all_bitvectors_array, from_string, to_string
from ._misc   import get_random_state, set_suffix


class BinarySample:
    """Class for representing samples of binary vectors,
    e.g., measurements of a qubit system.

    Args:
        counts (dict, optional): Dictionary mapping binary strings to integer counts.
            Defaults to None.
        raw (numpy.ndarray, optional): Array of shape ``(m, n)`` containing ``m``
            samples of bit vectors of size ``n``. If a ``counts`` dictionary is
            provided, this argument is ignored.

    Raises:
        ValueError: Raised if neither ``counts`` nor ``raw`` are set.

    Attributes:
        counts (dict): Dictionary mapping binary strings to integer counts.
        n (int): Size of the bit vectors recorded in this sample.
        size (int): Sample size.
        raw (numpy.ndarray): Raw sample, i.e., array containing bit vectors
            in the correct multiplicity.
        suff_stat (numpy.ndarray): Sufficient statistic of this sample, i.e.,
            an ``(n, n)`` matrix ``mat`` such that ``mat[i, j]`` is the number
            of samples ``x`` where ``x[i]==1`` and ``x[j]==1``, for all ``i<=j``.
    """

    def __init__(self, *, counts=None, raw: np.ndarray=None):
        if counts is not None:
            self.counts = counts
        elif raw is not None:
            C = Counter([to_string(x) for x in raw])
            self.counts = dict(C)
        else:
            raise ValueError('Provide counts or raw sample data!')

    def save(self, path: str):
        """Save the sample to disk.
        If the file exists, it will be overwritten.
        
        Args:
            path (str): Target file path.
        """
        f = open(set_suffix(path, 'sample'), 'wb')
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
    def load(cls, path: str):
        """Load sample from disk.

        Args:
            path (str): Sample file path.

        Returns:
            BinarySample: Sample instance loaded from disk.
        """
        with open(path, 'rb') as f:
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
            X[pointer:pointer+k, :] = np.tile(from_string(x), (k, 1))
            pointer += k
        return X

    @cached_property
    def suff_stat(self):
        X = self.raw
        return np.triu(X.T @ X)

    def hellinger_distance(self, other):
        """`Hellinger distance <https://en.wikipedia.org/wiki/Hellinger_distance#Discrete_distributions>`__ between this sample and another.
        The Hellinger distance between two discrete probability
        distributions :math:`p` and :math:`q` is defined as
        
        :math:`\\frac{1}{\\sqrt{2}}\\sqrt{\\sum_{x\\in\\lbrace 0,1\\rbrace^n}(\\sqrt{p(x)}-\\sqrt{q(x)})^2}`.
        
        In contrast to KL divergence, Hellinger distance is an
        actual distance, in that it is symmetrical.

        Args:
            other (BinarySample): Binary sample to compare against.

        Returns:
            float: Hellinger distance between the two samples.
        """
        assert self.n == other.n
        xs = set(self.counts.keys())
        xs.update(other.counts.keys())
        xs = list(xs)
        p1 = np.asarray([self.counts.get(x, 0) for x in xs], dtype=np.float64)/self.size
        p2 = np.asarray([other.counts.get(x, 0) for x in xs], dtype=np.float64)/other.size
        return np.linalg.norm(np.sqrt(p1)-np.sqrt(p2))/np.sqrt(2.0)

    def subsample(self, size: int, random_state=None):
        """Return a subsample of this sample instance of given size,
        i.e., a subset of the observed raw bit vectors.

        Args:
            size (int): Size of the subsample.
            random_state (optional): A numerical or lexical seed, or a NumPy random generator. Defaults to None.

        Returns:
            BinarySample: Subsample.
        """
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
        """Return the binary string that was observed most often.

        Returns:
            str: Most frequent binary string.
        """
        return max(self.counts, key=self.counts.get)

    def empirical_prob(self, x=None):
        """Return the empirical probability of observing a given
        binary string w.r.t. this sample. If no argument is
        provided, return a vector containing all empirical 
        probabilities of all ``n``-bit vectors in lexicographical
        order.

        Args:
            x (str, optional): Binary string to get the empirical probability for. Defaults to None.

        Returns:
            Empirical probability (float), or vector of shape ``(2**n,)`` containing all probabilities.
        """
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
    """Given a QUBO instance, sample from its Gibbs distribution
    by computing the full probability vector. This method yields
    faithful samples, but the memory requirement becomes
    infeasibly large for large QUBO sizes.

    Args:
        qubo (qubo): QUBO instance.
        samples (int, optional): Number of samples to draw. Defaults to 1.
        temp (float, optional): Temperature parameter of the Gibbs distribution. Defaults to 1.0.
        random_state (optional): A numerical or lexical seed, or a NumPy random generator. Defaults to None.

    Returns:
        BinarySample: Random sample.
    """
    npr = get_random_state(random_state)
    X = all_bitvectors_array(qubo.n)
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
            counts[to_string(x)] += 1
    return BinarySample(counts=dict(counts))


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


def gibbs(qubo,
          samples: int=1,
          burn_in: int=1000,
          keep_interval: int=10,
          initial=None,
          temp=1.0,
          random_state=None):
    """Perform Gibbs sampling on the Gibbs distribution induced by
    the given QUBO instance. This method builds upon a Markov chain
    that converges to the true distribution after a certain number
    of iterations (*burn-in* phase). The longer the initial burn-in
    phase, the higher the sample quality. A caveat of this method
    is that subsequent samples are not independent, which is why
    most samples are discarded (see ``keep_interval`` below).

    Args:
        qubo (qubo): QUBO instance.
        samples (int, optional): Sample size. Defaults to 1.
        burn_in (int, optional): Number of initial iterations that are discarded,
            the so-called *burn-in* phase. Defaults to 1000.
        keep_interval (int, optional): Number of samples out of which
            only one is kept, and the others discarded. Choosing a high
            value makes the samples more independent, but slows down 
            the sampling procedure. Defaults to 10.
        initial (numpy.ndarray, optional): Bit vector to use as a starting
            point for the Markov chain. Using a representative sample from
            the Gibbs distribution can make the burn-in phase obsolete.
            Defaults to None, which samples a random bit vector.
        temp (float, optional): Temperature parameter of the Gibbs distribution. Defaults to 1.0.
        random_state (optional): A numerical or lexical seed, or a NumPy random generator. Defaults to None.

    Returns:
        BinarySample: Random sample.
    """
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
                counts[to_string(x)] += 1
                sampled += 1
                skip = keep_interval-1
            else:
                skip -= 1
        else:
            # still in burn-in phase -> discard sample
            sampled += 1
    return BinarySample(counts=dict(counts))

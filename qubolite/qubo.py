import numpy as np

from .misc   import all_bitvectors, get_random_state, is_triu, set_suffix, warn_size


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

    def __getitem__(self, k):
        try:
            i, j = sorted(k)
            return self.m.__getitem__((i, j))
        except TypeError:
            return self.m.__getitem__((k, k))

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
            m *= npr.random(size=m.shape)<density
        return cls(m)

    def save(self, path: str):
        with open(set_suffix(path, 'qubo'), 'wb') as f:
            np.save(f, self.m)

    @classmethod
    def load(cls, path: str):
        return cls(np.load(path))

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
            try:
                i, j = k
                m[i, j] += v
            except ValueError:
                try:
                    i, = k
                    m[i, i] += v
                except ValueError:
                    pass
        m = np.triu(m + np.tril(m, -1).T)
        return cls(m)

    def brute_force(self):
        warn_size(self.n, limit=20)
        return min(all_bitvectors(self.n, read_only=False), key=self)

    def spectral_gap(self, return_optimum=False):
        warn_size(self.n, limit=20)
        o1, o2 = float('inf'), float('inf')
        for x in all_bitvectors(self.n):
            v = self(x)
            if v < o1:
                o1, o2 = v, o1
            elif v < o2:
                o2 = v
        sgap = o2-o1
        return (sgap, o1) if return_optimum else sgap

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
        params = np.sort(np.unique(np.r_[self.m[np.triu_indices_from(self.m)], 0]))
        max_diff = params[-1]-params[0]
        min_diff = np.min(params[1:]-params[:-1])
        r = max_diff/min_diff
        return np.log2(r) if bits else 20*np.log10(r)

    def absmax(self):
        return np.max(np.abs(self.m))

    def round(self, *args):
        return qubo(self.m.round(*args))
    
    def scale(self, factor):
        return qubo(self.m*factor)

    def as_int(self, bits=32):
        factor = ((1<<(bits-1))-1)/self.absmax()
        return qubo((self.m*factor).round())

    def partition_function(self, log=False, temp=1.0, fast=True):
        Z = self.probabilities(temp=temp, unnormalized=True, fast=fast).sum()
        return np.log(Z) if log else Z

    def probabilities(self, temp=1.0, out=None, unnormalized=False, fast=True):
        if out is None:
            out = np.empty(1<<self.n)
        else:
            assert out.shape == (1<<self.n,), f'out array has wrong shape, ({1<<self.n},) expected'
        if fast:
            # builds the entire (2**n, n)-array of n-bit vectors
            X = np.vstack(list(all_bitvectors(self.n, read_only=False)))
            out[...] = np.exp(-self(X)/temp)
        else:
            # uses less memory, but much slower
            warn_size(self.n, limit=20)
            for i, x in enumerate(all_bitvectors(self.n)):
                out[i] = np.exp(-self(x)/temp)
        if unnormalized:
            return out
        return out/out.sum()

    def pairwise_marginals(self, temp=1.0, fast=True):
        warn_size(self.n, limit=20)
        probs = self.probabilities(temp=temp, fast=fast)
        marginals = np.zeros((self.n, self.n))
        for x, p in zip(all_bitvectors(self.n), probs):
            suff_stat = np.outer(x, x)
            marginals += p*suff_stat
        return np.triu(marginals)
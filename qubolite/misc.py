import warnings
from hashlib   import md5
from importlib import import_module
from operator  import itemgetter
from sys       import stderr

import numpy as np

from .bitvec import all_bitvectors_array


class Intervals:
    def __init__(self, ivs=None):
        if ivs is None:
            self.__ivs = np.empty((0, 2))
        self.__ivs = self.__normalize(ivs)

    @classmethod
    def from_points(cls, x, radius):
        return cls(np.vstack([x-radius, x+radius]).T)

    def copy(self):
        return Intervals(self)

    def __repr__(self):
        return f'Intervals({list(map(tuple, self.__ivs)).__repr__()})'

    def __iter__(self):
        return iter(self.__ivs)

    def __normalize(self, ivs):
        iv_iter = iter(sorted(ivs, key=itemgetter(0)))
        ivs_ = [*next(iv_iter)]
        for u, v in iv_iter:
            assert u < v, 'left interval boundary must be less than right boundary'
            if u <= ivs_[-1]:
                if ivs_[-1] < v:
                    ivs_[-1] = v
            else:
                ivs_.extend([u, v])
        return np.asarray(ivs_, dtype=np.float64).reshape((-1, 2))

    def union(self, *others):
        if len(others) == 0:
            return self.copy()
        return Intervals(np.vstack([obj.__ivs for obj in [self, *others]]))

    def intersect(self, *others):
        ivs = []
        m = len(others)+1 # number of Intervals objects
        if m == 1:
            return self.copy()
        l = max([obj.__ivs.shape[0] for obj in [self, *others]]) # max. num. of intervals
        uvs = np.full((m, l, 2), np.infty)
        for i, obj in enumerate([self, *others]):
            uvs[i, :obj.__ivs.shape[0], :] = obj.__ivs
        states = np.ones(len(others)+1)
        prev_state_sum = states.sum()
        print(uvs)
        for i, j, k in zip(*np.unravel_index(np.argsort(uvs, axis=None), uvs.shape)):
            states[i] = k
            value = uvs[i, j, k]
            if value == np.infty:
                break
            state_sum = states.sum()
            if state_sum == 0 or (prev_state_sum == 0 and state_sum == 1):
                # all intervals are open (0), or
                # one interval just closed (0->1)
                ivs.append(uvs[i, j, k])
            prev_state_sum = state_sum
        return Intervals(np.asarray(ivs).reshape((-1, 2)))
        
    def difference(self, *others):
        raise NotImplementedError()


# make warning message more minialistic
def _custom_showwarning(message, *args, file=None, **kwargs):
    (file or stderr).write(f'Warning: {str(message)}\n')
warnings.showwarning = _custom_showwarning


def warn(*args, **kwargs):
    warnings.warn(*args, **kwargs)


def is_symmetrical(arr, rtol=1e-05, atol=1e-08):
    return np.allclose(arr, arr.T, rtol=rtol, atol=atol)


def is_triu(arr):
    return np.all(np.isclose(arr, np.triu(arr)))


def min_max(it):
    min_ = float('inf')
    max_ = float('-inf')
    for x in it:
        if x < min_: min_ = x
        if x > max_: max_ = x
    return min_, max_


def warn_size(n: int, limit: int=30):
    if n > limit:
        warn(f'This operation may take a very long time for n>{limit}.')


def get_random_state(state=None):
    if state is None:
        return np.random.default_rng()
    if isinstance(state, np.random._generator.Generator):
        return state
    if isinstance(state, np.random.RandomState):
        # for compatibility
        seed = state.randint(0xffffffff)
        return np.random.default_rng(seed)
    try:
        seed = int(state)
    except ValueError:
        # use hash digest when seed is a (non-numerical) string
        seed = int(md5(state.encode('utf-8')).hexdigest(), 16) & 0xffffffff
    return np.random.default_rng(seed)


def set_suffix(filename, suffix):
    s = suffix.strip(' .')
    if filename.lower().endswith('.'+s.lower()):
        return filename
    else:
        return f'{filename}.{s}'
    

def try_import(*libs):
    libdict = dict()
    for lib in libs:
        try:
            module = import_module(lib)
        except ModuleNotFoundError:
            continue
        libdict[lib] = module


def ordering_distance(Q1, Q2, X=None):
    try:
        from scipy.stats import kendalltau
    except ImportError as e:
        raise ImportError(
            "scipy needs to be installed prior to running qubolite.ordering_distance(). You "
            "can install scipy with:\n'pip install scipy'"
        ) from e
    assert Q1.n == Q2.n, 'QUBO instances must have the same dimension'
    warn_size(Q1.n, limit=22)
    if X is None:
        X = all_bitvectors_array(Q1.n)
    rnk1 = np.argsort(np.argsort(Q1(X)))
    rnk2 = np.argsort(np.argsort(Q2(X)))
    tau, _ = kendalltau(rnk1, rnk2)
    return (1-tau)/2

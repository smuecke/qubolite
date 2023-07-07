import warnings
from hashlib   import md5
from importlib import import_module
from sys       import stderr

import numpy as np

from .bitvec import all_bitvectors_array


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
        seed = state.randint(1<<31)
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

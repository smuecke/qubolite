import warnings
from functools import wraps
from hashlib   import md5
from importlib import import_module
from sys       import stderr

import numpy as np


# make warning message more minialistic
def _custom_showwarning(message, *args, file=None, **kwargs):
    (file or stderr).write(f'Warning: {str(message)}\n')
warnings.showwarning = _custom_showwarning


def warn(*args, **kwargs):
    warnings.warn(*args, **kwargs)


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn(f'{func.__name__} is deprecated',
                      category=DeprecationWarning,
                      stacklevel=2)
        return func(*args, **kwargs)
    return new_func


def is_symmetrical(arr, rtol=1e-05, atol=1e-08):
    return np.allclose(arr, arr.T, rtol=rtol, atol=atol)


def make_upper_triangle(arr):
    return np.triu(arr) + np.tril(arr, -1).T


def is_triu(arr):
    return np.all(np.isclose(arr, np.triu(arr)))


def min_max(it):
    min_ = float('inf')
    max_ = float('-inf')
    for x in it:
        if x < min_: min_ = x
        if x > max_: max_ = x
    return min_, max_


def to_shape(size_or_shape):
    if isinstance(size_or_shape, int):
        if size_or_shape == 1:
            return ()
        return (size_or_shape,)
    return (*size_or_shape,)


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
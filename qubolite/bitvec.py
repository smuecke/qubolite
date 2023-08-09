import re

import numpy as np
from numpy.typing import ArrayLike

from ._misc import get_random_state


def all_bitvectors(n: int):
    """Generate all bit vectors of size ``n`` in lexicographical order, starting from all zeros.
    The least significant bit is at index 0. Note that always the same array object is yielded,
    so to persist the bit vectors you need to make copies.

    Args:
        n (int): Number of bits.

    Yields:
        numpy.ndarray: Array of shape ``(n,)`` containing a bit vector.

    Exapmles:
        This method can be used to obtain all possible energy values for a given QUBO instance:

        >>> Q = qubo.random(3)
        >>> for x in all_bitvectors(3):
        ...     print(to_string(x), '->', Q(x))
        ... 
        000 -> 0.0
        100 -> 0.6294629779101759
        010 -> 0.1566040993504083
        110 -> 0.5098350500036248
        001 -> 1.5430218546339793
        101 -> 3.9359808951564057
        011 -> 2.1052824965032304
        111 -> 4.222009509768697
    """
    x = np.zeros(n)
    b = 2**np.arange(n)
    for k in range(1<<n):
        x[:] = (k & b) > 0
        yield x
   
    
def all_bitvectors_array(n: int):
    """Create an array containing all bit vectors of size ``n`` in
    lexicographical order.

    Args:
        n (int): Size of bit vectors.

    Returns:
        numpy.ndarray: Array of shape ``(2**n, n)``
    """
    return np.arange(1<<n)[:, np.newaxis] & (1<<np.arange(n)) > 0


def random(n: int, size=None, random_state=None):
    """Create an array containing random bit vectors.

    Args:
        n (int): Number of bits per bit vector.
        size (int | tuple, optional): Number of bit vectors to sample, or tuple
            representing a shape. Defaults to None, which returns a single bit
            vector (shape ``(n,)``).
        random_state (optional): A numerical or lexical seed, or a NumPy random
            generator. Defaults to None.

    Returns:
        numpy.ndarray: Random bit vector(s).
    """
    size = () if size is None else size
    try:
        shape = (*size, n)
    except TypeError:
        # `size` must be an integer
        shape = (size, n)
    rng = get_random_state(random_state)
    return (rng.random(shape) < 0.5).astype(np.float64)


def from_string(string: str):
    """Convert a string consisting of ``0`` and ``1`` to a bit vector.

    Args:
        string (str): Binary string.

    Returns:
        numpy.ndarray: Bit vector.

    Examples:
        This method is useful to quickly convert binary strings
        to numpy array:

        >>> from_string('100101')
        array([1., 0., 0., 1., 0., 1.])
    """
    return np.fromiter(string, dtype=np.float64)


def to_string(bitvec: ArrayLike):
    """Convert a bit vector to a string.
    If an array of bit vectors (shape ``(m, n)``) is passed, a numpy.ndarray
    containing string objects is returned, one for each row.

    Args:
        bitvec (ArrayLike): Bit vector ``(n,)`` or array of bit vectors ``(m, n)``

    Returns:
        string: Binary string
    """
    if bitvec.ndim <= 1:
        return ''.join(str(int(x)) for x in bitvec)
    else:
        return np.apply_along_axis(to_string, -1, bitvec)
    

def from_dict(d: dict, n=None):
    """Convert a dictionary to a bit vector.
    The dictionary should map indices (int) to binary values (0 or 1).
    If ``n`` is specified, the vector is padded with zeros to length ``n``.

    Args:
        d (dict): Dictionary containing index to bit assignments.
        n (int, optional): Length of the bit vector. Defaults to None, which uses the highest index in ``d`` as length.

    Returns:
        numpy.ndarray: Bit vector.
    """
    n = max(d.keys())+1 if n is None else n
    x = np.zeros(n)
    for i, b in d.items():
        x[i] = b
    return x


def to_dict(bitvec: ArrayLike):
    """Convert a bit vector to a dictionary mapping index (int)
    to bit value (0 or 1).

    Args:
        bitvec (ArrayLike): Bit vector of shape ``(n,)``.

    Returns:
        dict: Dictionary representation of the bit vector.

    Examples:
        This function is useful especially when working with
        D-Wave's Python packages, as they often use this format.

        >>> to_dict(from_string('10101100'))
        {0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0}
    """
    return { i: int(b) for i, b in enumerate(bitvec) }


# Manipulate Bit Vectors
# ======================

def flip_index(x, index, in_place=False):
    """Flips the values of a given bit vector at the specified index or indices.

    Args:
        x (numpy.ndarray): Bit vector(s).
        index (int | list | numpy.ndarray): Index or list of indices where to
            flip the binary values.
        in_place (bool, optional): If ``True``, modify the bit vector in place.
            The return value will be a reference to the input array. Defaults to
            False.

    Returns:
        numpy.ndarray: Bit vector(s) with the indices flipped at the specified
            positions. If ``in_place=True``, this will be a reference to the
            input array, otherwise a copy.

    Examples:
        The following inverts the first and last bits of all given bitvectors:

        >>> x = from_expression('**10')
        >>> x
        array([[0., 0., 1., 0.],
               [1., 0., 1., 0.],
               [0., 1., 1., 0.],
               [1., 1., 1., 0.]])
        >>> flip_index(x, [0, -1])
        array([[1., 0., 1., 1.],
               [0., 0., 1., 1.],
               [1., 1., 1., 1.],
               [0., 1., 1., 1.]])
    """
    x_ = x if in_place else x.copy()
    x_[..., index] = 1-x_[..., index]
    return x_
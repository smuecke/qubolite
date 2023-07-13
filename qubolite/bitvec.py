import re

import numpy as np
from numpy.typing import ArrayLike


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


_BITVEC_EXPR = re.compile(r'[01*]|\[!?\d+\]')

def _expr_normal_form(tokens):
    # re-arrange tokens so that all references
    # point to previous indices
    i = 0
    while i < len(tokens):
        if not tokens[i].startswith('['):
            i += 1
            continue
        j = int(tokens[i].strip('[!]'))
        if tokens[j].startswith('['):
            raise RuntimeError('No references to references allowed!')
        if j > i:
            if tokens[i].startswith('[!'):
                tokens[i] = '[!' + str(i) + ']'
                tokens[j] = '10*'['01*'.index(tokens[j])]
            else:
                tokens[i] = '[' + str(i) + ']'
            # swap tokens
            tokens[i], tokens[j] = tokens[j], tokens[i]
            # check if constant needs to be inverted
        i += 1
    return tokens

def from_expression(expr: str):
    """Generate an array of bit vectors from a string
    containing a bit vector expression. Such an expression
    consists of a sequence of these symbols: ``0`` - a constant 0,
    ``1`` - a constant 1, ``*`` - all combinations of 0 and 1,
    ``[i]`` - the same as the bit at index i, ``[!i]`` -  the
    inverse of the bit at index i.

    The last two symbols are called references, and ``i`` their
    pointing index (counting from 0). A reference can only ever
    point to a constant or ``*``, i.e., higher-order references are
    not allowed (and not necessary).

    Args:
        expr (str): Bit vector expression.

    Returns:
        numpy.ndarray: Array of bit vectors matched by the expression.

    Examples:
        This function is useful for generating arrays of bit vectors
        with a prescribed structure. For instance, "all bit vectors
        of length 4 that start with 1 and where the last two bits are
        the same" can be expressed as

        >>> from_expression('1**[2]')
        array([[1., 0., 0., 0.],
        ...    [1., 1., 0., 0.],
        ...    [1., 0., 1., 1.],
        ...    [1., 1., 1., 1.]])
    """
    tokens = _expr_normal_form(list(_BITVEC_EXPR.findall(expr)))
    n = len(tokens)
    n_free = expr.count('*')
    x = np.empty((1<<n_free, n))
    power = 0
    for i, token in enumerate(tokens):
        if token in ['0', '1']:
            x[:, i] = float(token)
        elif token == '*':
            x[:, i] = np.arange(1<<n_free) & (1<<power) > 0
            power += 1
        else:
            ix = int(token.strip('[!]'))
            if token[1] == '!': # inverse reference
                x[:,i] = 1-x[:,ix]
            else:
                x[:,i] = x[:,ix]
    return x

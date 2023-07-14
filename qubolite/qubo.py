import struct
from functools import reduce

import numpy as np

from .bitvec  import all_bitvectors, all_bitvectors_array
from ._misc    import get_random_state, is_triu, warn_size
from _c_utils import brute_force as _brute_force_c

def is_qubo_like(arr):
    """Check if given array defines a QUBO instance, i.e.,
    if the array is 2-dimensional and square.

    Args:
        arr (numpy.ndarray): Input array.

    Returns:
        bool: ``True`` iff the input array defines a QUBO instance.
    """
    if arr.ndim == 2:
        u, v = arr.shape[-2:]
        return u == v
    else:
        return False


def to_triu_form(arr):
    """Convert an array defining a QUBO instance to an
    upper triangle matrix, if necessary.

    Args:
        arr (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Upper triangular matrix.
    """
    if is_triu(arr):
        return arr.copy()
    else:
        # add lower to upper triangle
        return np.triu(arr + np.tril(arr, -1).T)


def __unwrap_value(obj):
    try:
        v = obj.m
    except AttributeError:
        v = obj
    return v


class qubo:
    """
    Standard class for QUBO instances.
    This is mainly a wrapper around an upper triangular NumPy matrix with
    lots of helpful methods. The passed array must be of the shape ``(n, n)``
    for any positive ``n``. The linear coefficients lie along the diagonal.
    A non-triangular matrix will be converted, i.e., the lower triangle will
    be transposed and added to the upper triangle.

    Args:
        m (np.ndarray): Array containing the QUBO parameters.

    Examples:
        If you have linear and quadratic coefficients in separate arrays,
        e.g., ``lin`` with shape ``(n,)`` and ``qua`` with shape ``(n, n)``,
        they can be combined to a ``qubo`` instance through ``qubo(np.diag(lin) + qua)``.
    """

    def __init__(self, m: np.ndarray):
        """
        Creates a ``qubo`` instance from a given NumPy array.
        
        """
        assert is_qubo_like(m)
        self.m = to_triu_form(m)
        self.n = m.shape[-1]

    def __repr__(self):
        return 'qubo'+self.m.__repr__().lstrip('array')

    def __call__(self, x: np.ndarray):
        """Calculate the QUBO energy value for a given bit vector.

        Args:
            x (numpy.ndarray): Bit vector of shape ``(n,)``, or multiple bit vectors ``(m, n)``.

        Returns:
            float or numpy.ndarray: QUBO energy value, or array ``(m,)`` of energy values.
        """
        return np.sum(np.dot(x, self.m)*x, axis=-1)

    def __getitem__(self, k):
        try:
            i, j = sorted(k)
            return self.m.__getitem__((i, j))
        except TypeError:
            return self.m.__getitem__((k, k))

    def __add__(self, other):
        return qubo(self.m + __unwrap_value(other))
    
    def __sub__(self, other):
        return qubo(self.m - __unwrap_value(other))

    def __mul__(self, other):
        return qubo(self.m * __unwrap_value(other))

    def __truediv__(self, other):
        return qubo(self.m / __unwrap_value(other))

    def copy(self):
        """Create a copy of this instance.

        Returns:
            qubo: Copy of this QUBO instance.
        """
        return qubo(self.m.copy())

    @classmethod
    def random(cls, n: int, distr='normal', density=1.0, random_state=None, **kwargs):
        """Create a QUBO instance with parameters sampled from a random distribution.

        Args:
            n (int): QUBO size
            distr (str, optional): Distribution from which the parameters are sampled.
                Possible values are 'normal', 'uniform' and 'triangular'.
                Additional keyword arguments will be passed to the corresponding
                methods from ``numpy.random``. Defaults to 'normal'.
            density (float, optional): Expected density of the parameter matrix.
                Each parameter is set to 0 with probability ``1-density``.
                Defaults to 1.0.
            random_state (optional): A numerical or lexical seed, or a NumPy random generator. Defaults to None.

        Raises:
            ValueError: Raised if the ``distr`` argument is unknown.

        Returns:
            qubo: Random QUBO instance
        """
        npr = get_random_state(random_state)
        if distr == 'normal':
            arr = npr.normal(
                kwargs.get('loc', 0.0),
                kwargs.get('scale', 1.0),
                size=(n, n))
        elif distr == 'uniform':
            arr = npr.uniform(
                kwargs.get('low', -1.0),
                kwargs.get('high', 1.0),
                size=(n, n))
        elif distr == 'triangular':
            arr = npr.triangular(
                kwargs.get('left', -1.0),
                kwargs.get('mode', 0.0),
                kwargs.get('right', 1.0),
                size=(n, n))
        else:
            raise ValueError(f'Unknown distribution "{distr}"')
        m = np.triu(arr)
        if density < 1.0:
            m *= npr.random(size=m.shape)<density
        return cls(m)

    def save(self, path: str, atol=1e-16):
        """Save the QUBO instance to disk.
        If the file exists, it will be overwritten.
        
        Args:
            path (str): Target file path.
            atol (float, optional): Parameters with absolute value below this value will be treated as 0. Defaults to 1e-16.
        """
        f = open(path, 'wb')
        f.write(struct.pack('<4s', b'QUBO')) # magic string
        f.write(struct.pack('<I', self.n)) # QUBO size
        # determine mode
        #  0x00: save flattened parameter array
        #  0x01: save index-value pairs
        n_nonzero = self.n**2-np.isclose(self.m, 0, atol=atol).sum()
        index_bytes = 1 if self.n <= 256 else (2 if self.n <= 2 else 4)
        size_mode0 = 4*self.n*(self.n+1)
        size_mode1 = (2*index_bytes+8)*n_nonzero
        mode = 0 if size_mode0 <= size_mode1 else 255
        f.write(struct.pack('B', mode)) # mode indicator
        if mode == 0:
            # save flattened parameter array
            f.write(self.m[np.triu_indices_from(self.m)].tobytes())
        else:
            # save index-value pairs;
            # determine index type depending on size
            t = 'B' if self.n <= 256 else ('H' if self.n <= 65536 else 'I')
            fmt = f'<{t}{t}d'
            # write only non-zero parameters
            for i, j in zip(*np.triu_indices_from(self.m)):
                if not np.isclose(self.m[i,j], 0, atol=atol):
                    f.write(struct.pack(fmt, i, j, self.m[i,j]))
        f.close()

    @classmethod
    def load(cls, path: str):
        """Load QUBO instance from disk.

        Args:
            path (str): QUBO file path.

        Raises:
            RuntimeError: Raised if the file contains no valid QUBO.

        Returns:
            qubo: QUBO instance loaded from disk.
        """
        f = open(path, 'rb')
        magic, = struct.unpack('<4s', f.read(4))
        if magic != b'QUBO':
            raise RuntimeError('Invalid QUBO file')
        n, mode = struct.unpack('<IB', f.read(5))
        m = np.zeros((n, n))
        if mode == 0:
            m[np.triu_indices_from(m)] = np.frombuffer(f.read())
        else:
            t = 'B' if n <= 256 else ('H' if n <= 65536 else 'I')
            fmt = f'<{t}{t}d'
            for i, j, value in struct.iter_unpack(fmt, f.read()):
                m[i,j] = value
        f.close()
        return cls(m)

    def to_dict(self, names=None, double_indices=True, atol=1e-16):
        """Create a dictionary mapping variable indices to QUBO parameters.
        Contains entries only for non-zero parameters.
        
        Args:
            names (dict, optional): Dictionary mapping variables indices
                (counting from 0) to names. By default, just the integer
                indices are used.
            double_indices (bool, optional): If ``True``, use ``(i, i)`` as
                the key for diagonal entries, otherwise ``(i,)``. Defaults to True.
            atol (float, optional): Parameters with absolute value below
                this value will be treated as 0. Defaults to 1e-16.
            
        Returns:
            dict: Dictionary containing QUBO parameters.

        Examples:
            >>> Q = qubo.random(4, density=0.25).round(1)
            >>> Q
            qubo([[ 0.6,  0. ,  0.5,  0. ],
                  [ 0. ,  0. , -0.4,  0. ],
                  [ 0. ,  0. ,  0. , -0.3],
                  [ 0. ,  0. ,  0. ,  0. ]])
            >>> Q.to_dict()
            {(0, 0): 0.6, (0, 2): 0.5, (1, 2): -0.4, (2, 3): -0.3}
        """
        if names is None:
            names = { i: i for i in range(self.n) }
        qubo_dict = dict()
        for i, j in zip(*np.triu_indices_from(self.m)):
            if not np.isclose(self.m[i, j], 0, atol=atol):
                if (i == j) and (not double_indices):
                    qubo_dict[(names[i],)] = self.m[i, i]
                else:
                    qubo_dict[(names[i], names[j])] = self.m[i, j]
        return qubo_dict

    @classmethod
    def from_dict(cls, qubo_dict, n=None, relabel=True):
        """Create QUBO instance from a dictionary mapping variable indices
        to QUBO parameters. Note that, by default, unused variables are eliminated,
        e.g., the dictionary ``{(0,): 2, (100,): -3}`` yields a QUBO instance
        of size n=2. If you want to use the dictionary keys as variable indices
        as-is, set ``relabel=False``.

        Args:
            qubo_dict (dict): Dictionary mapping indices to QUBO parameters.
            n (int, optional): Specifies QUBO size. If None, the size is derived
                from the number of variable names.
            relabel (bool, optional): Indicate whether the variables should
                be used as indices as-is, instead of removing unused variables.
                This works only for integer keys.

        Returns:
            qubo: QUBO instance with parameters taken from dictionary.
            dict: Dictionary mapping the names of the variables used in
                the input dictionary to the indices of the QUBO instance.
                If ``relabel=False``, this dictionary will be an identity map.
        """
        if relabel:
            names = { k: i for i, k in enumerate(sorted(set().union(*qubo_dict.keys()))) }
        else:
            names = { i: i for i in set().union(*qubo_dict.keys()) }

        n = max(names.values())+1 if n is None else n
        m = np.zeros((n, n))
        for k, v in qubo_dict.items():
            try:
                i, j = k
                m[names[i], names[j]] += v
            except ValueError:
                try:
                    i, = k
                    m[names[i], names[i]] += v
                except ValueError:
                    pass
        m = np.triu(m + np.tril(m, -1).T)
        return cls(m), { i: k for k, i in names.items() }

    @classmethod
    def from_ising(cls, linear, quadratic, offset=0.0):
        lin = np.asarray(linear)
        qua = np.asarray(quadratic)
        n, = lin.shape
        assert qua.shape == (n, n), '`linear` and `quadratic` must have shapes (n,) and (n, n)'
        qua_symm = np.tril(qua, -1).T + np.triu(qua, 1) + qua
        m  = 2*np.diag(qua.sum(0)+lin)
        m -= 4*np.triu(qua_symm, 1)
        return cls(m)


    def spectral_gap(self, return_optimum=False):
        """Calculate the spectral gap of this QUBO instance.
        Here, this is defined as the difference between the lowest and second-to lowest QUBO energy value across all bit vectors.
        Note that the QUBO instance must be solved for calculating this value, therefore only QUBOs of sizes up to about 30 are feasible in practice.
        
        Args:
            return_optimum (bool, optional): If ``True``, returns the minimizing bit vector of this QUBO instance (which is calculated anyway). Defaults to False.

        Raises:
            ValueError: Raised if this QUBO instance is too large to be solved by brute force on the given system.

        Returns:
            sgap (float): Spectral gap.
            x (numpy.ndarray, optional): Minimizing bit vector.
        """
        warn_size(self.n, limit=25)
        try:
            x, v0, v1 = _brute_force_c(self.m)
        except TypeError:
            raise ValueError('n is too large to brute-force on this system')
        sgap = v1-v0
        if return_optimum:
            return sgap, x
        else:
            return sgap

    def clamp(self, partial_assignment=None):
        """Create QUBO instance equivalent to this but with a subset
        of variables fixed (_clamped_) to constant values.

        Args:
            partial_assignment (dict, optional): Dictionary mapping
                variable indices (counting from 0) to constant values
                0 or 1. Defaults to None, which does nothing and
                returns a copy of this QUBO instance.

        Returns:
            qubo: Clamped QUBO instance.
            const (float): Constant offset value, which must be added to the QUBO energy to obtain the original energy.
            free (list): List of indices which the variable indices of the new QUBO instance correspond to (i.e., those indices that were not clamped).
        """
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
        """Discrete derivative w.r.t. ``x``.
        The element at index ``i`` gives the QUBO energy change upon flipping the value of ``x[i]``.
        
        Args:
            x (np.ndarray): Bit vector w.r.t. which the discrete derivative is calculated. Can be an array of multiple bit vectors.

        Returns:
            numpy.ndarray: Vector of discrete derivatives of this QUBO instance w.r.t. ``x``.
        """
        # 1st discrete derivatice
        m_  = np.triu(self.m, 1)
        m_ += m_.T
        sign = 1-2*x
        return sign*(np.diag(self.m)+(m_ @ x.T).T)

    def dynamic_range(self, decibel=False):
        """Calculate the dynamic range (DR) of the QUBO parameters,
        i.e., the logarithmic ratio between the largest and the smallest difference between all pairs of unique parameter values.

        Args:
            decibel (bool, optional): If ``True``, outputs the DR in the unit decibels. Defaults to False, which outputs the DR in bits.

        Returns:
            float: Dynamic range value.
        """
        params = np.sort(np.unique(np.r_[self.m[np.triu_indices_from(self.m)], 0]))
        max_diff = params[-1]-params[0]
        min_diff = np.min(params[1:]-params[:-1])
        r = max_diff/min_diff
        return 20*np.log10(r) if decibel else np.log2(r)

    def absmax(self):
        """Returns the largest parameter by absolute value.
        This is equivalent to the infinity norm of the QUBO matrix.

        Returns:
            float: largest parameter by absolute value.
        """
        return np.max(np.abs(self.m))

    def round(self, *args):
        """Rounds the QUBO parameters to the nearest integers.

        Returns:
            qubo: QUBO instance with rounded parameters.
        """
        return qubo(self.m.round(*args))
    
    def scale(self, factor):
        """Scale the QUBO parameters by a constant factor.

        Args:
            factor (float): Scaling factor.

        Returns:
            qubo: QUBO instance with scaled parameters.
        """
        return qubo(self.m*factor)

    def as_int(self, bits=32):
        """Scales and rounds the QUBO parameters to fit a given number of bits.
        The number format is assumed to be signed integer, therefore ``b`` bits yields a value range of ``-2**(b-1)`` to ``2**(b-1)-1``.

        Args:
            bits (int, optional): Number of bits to represent the parameters. Defaults to 32.

        Returns:
            qubo: QUBO instance with scaled and rounded parameters.
        """
        p_min, p_max = self.m.min(), self.m.max()
        if np.abs(p_min) < np.abs(p_max):
            factor = ((2**(bits-1))-1)/np.abs(p_max)
        else:
            factor = (2**(bits-1))/np.abs(p_min)
        return qubo((self.m*factor).round())

    def partition_function(self, log=False, temp=1.0, fast=True):
        """Calculate the partition function of the Ising model induced by this QUBO instance.
        That is, return the sum of ``exp(-Q(x)/temp)`` over all bit vectors ``x``.
        Note that this is infeasibly slow for QUBO sizes much larger than 20.

        Args:
            log (bool, optional): Return the natural log of the partition function instead. Defaults to False.
            temp (float, optional): Temperature parameter of the Gibbs distribution. Defaults to 1.0.
            fast (bool, optional): Internally create array of all bit vectors. This is faster, but requiers memory space exponential in the QUBO size. Defaults to True.

        Returns:
            float: Value of the partition function, or the log partition function if ``log=True``.
        """
        Z = self.probabilities(temp=temp, unnormalized=True, fast=fast).sum()
        return np.log(Z) if log else Z

    def probabilities(self, temp=1.0, out=None, unnormalized=False, fast=True):
        """Compute the complete vector of probabilities for observing a vector ``x`` under the Gibbs distribution induced by this QUBO instance.
        The entries of the resulting array are sorted in lexicographic order by bit vector, e.g. for size 3: ``[000, 100, 010, 110, 001, 101, 011, 111]``.
        Note that this method requires memory space exponential in QUBO size, which quickly becomes infeasible, depending on your system.
        If ``n`` is the QUBO size, the output will have size ``2**n``.
        
        Args:
            temp (float, optional): Temperature parameter of the Gibbs distribution. Defaults to 1.0.
            out (numpy.ndarray, optional): Array to write the probabilities to. Defaults to None, which creates a new array.
            unnormalized (bool, optional): Return the unnormalized probabilities. Defaults to False.
            fast (bool, optional): Internally create array of all bit vectors. This is faster, but requiers memory space exponential in the QUBO size. Defaults to True.

        Returns:
            numpy.ndarray: Array containing probabilities.
        """
        if out is None:
            out = np.empty(1<<self.n)
        else:
            assert out.shape == (1<<self.n,), f'out array has wrong shape, ({1<<self.n},) expected'
        if fast:
            # builds the entire (2**n, n)-array of n-bit vectors
            X = all_bitvectors_array(self.n)
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
        """Compute the marginal probabilities for each variable pair to assume the value (1, 1) under the Gibbs distribution induced by this QUBO instance.
        Note that this operation's runtime is exponential in QUBO size.

        Args:
            temp (float, optional): Temperature parameter of the Gibbs distribution. Defaults to 1.0.
            fast (bool, optional): Internally create array of all bit vectors. This is faster, but requiers memory space exponential in the QUBO size. Defaults to True.

        Returns:
            numpy.ndarray: Upper triangular matrix of probabilities.
        """
        warn_size(self.n, limit=20)
        probs = self.probabilities(temp=temp, fast=fast)
        marginals = np.zeros((self.n, self.n))
        for x, p in zip(all_bitvectors(self.n), probs):
            suff_stat = np.outer(x, x)
            marginals += p*suff_stat
        return np.triu(marginals)

    def to_posiform(self):
        """Compute the unique posiform representation of this QUBO instance,
        using the approach described in section 2.1 of [1].
        The result is a tuple of an array ``P`` of shape ``(2, n, n)``, where ``n`` is the QUBO size,
        and a constant offset value. All entries of the array are positive.
        ``P[0]`` contains the coefficients for the literals ``Xi*Xj``, and ``Xi`` on the diagonal, while
        ``P[1]`` contains the coefficients for ``Xi*!Xj`` (``!`` denoting negation), and ``!Xi`` on the diagonal.
        See the paper for further infos.

        [1] https://www.researchgate.net/publication/238379061_Preprocessing_of_unconstrained_quadratic_binary_optimization

        Returns:
            numpy.ndarray: Posiform coefficients (see above)
            float: Constant offset value
        """
        posiform = np.zeros((2, self.n, self.n))
        # posiform[0] contains terms xi* xj, and  xi on diagonal
        # posiform[1] contains terms xi*!xj, and !xi on diagonal
        lin = np.diag(self.m)
        qua = np.triu(self.m, 1)
        diag_ix = np.diag_indices_from(self.m)
        qua_neg = np.minimum(qua, 0)
        posiform[0] = np.maximum(qua, 0)
        posiform[1] = -qua_neg
        posiform[0][diag_ix] = lin + qua_neg.sum(1)
        lin_ = posiform[0][diag_ix].copy()  # =: c'
        lin_neg = np.minimum(lin_, 0)
        posiform[1][diag_ix] = -lin_neg
        posiform[0][diag_ix] = np.maximum(lin_, 0)
        const = lin_neg.sum()
        return posiform, const

    def to_ising(self):
        return NotImplemented
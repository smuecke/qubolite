import numpy as np

from ._misc   import get_random_state, warn_size
from .bitvec  import flip_index
from .qubo    import qubo
from _c_utils import brute_force as _brute_force_c


def brute_force(Q: qubo):
    """Solve QUBO instance exactly by brute force. Note that this method is
    infeasible for instances with a size beyond around 30.

    Args:
        Q (qubo): QUBO instance to solve.

    Raises:
        ValueError: Raised if the QUBO size is too large to be brute-forced on
            the present system.

    Returns:
        A tuple containing the minimizing vector (numpy.ndarray) and the minimal
        energy (float).
    """
    warn_size(Q.n, limit=30)
    try:
        x, v, _ = _brute_force_c(Q.m)
    except TypeError:
        raise ValueError(f'n is too large to brute-force on this system')
    return x, v


def simulated_annealing(Q: qubo,
                        schedule='2+',
                        halftime=0.25,
                        steps=100_000,
                        init_temp=None,
                        n_parallel=10,
                        random_state=None):
    """Performs simulated annealing to approximate the minimizing vector and
    minimal energy of a given QUBO instance.

    Args:
        Q (qubo): QUBO instance.
        schedule (str, optional): The annealing schedule to employ. Possible
            values are: ``2+`` (quadratic additive), ``2*`` (quadratic
            multiplicative), ``e+`` (exponential additive) and ``e*``
            (exponential multiplicative). See
            `here <http://what-when-how.com/artificial-intelligence/a-comparison-of-cooling-schedules-for-simulated-annealing-artificial-intelligence/>`__ 
            for further infos. Defaults to '2+'.
        halftime (float, optional): For multiplicative schedules only: The
            percentage of steps after which the temperature is halved. Defaults
            to 0.25.
        steps (int, optional): Number of annealing steps to perform. Defaults to
            100_000.
        init_temp (float, optional): Initial temperature. Defaults to None,
            which estimates an initial temperature.
        n_parallel (int, optional): Number of random initial solutions to anneal
            simultaneously. Defaults to 10.
        random_state (optional): A numerical or lexical seed, or a NumPy random
            generator. Defaults to None.

    Raises:
        ValueError: Raised if the specificed schedule is unknown.

    Returns:
        A tuple ``(x, y)`` containing the solution bit vectors and their
        respective energies. The shape of ``x`` is ``(n_parallel, n)``, where
        ``n`` is the QUBO size; the shape of ``y`` is ``(n_parallel,)``. Bit
        vector ``x[i]`` has energy ``y[i]`` for each ``i``.
    """
    npr = get_random_state(random_state)
    if init_temp is None:
        # estimate initial temperature
        EΔy, k = 0, 0
        for _ in range(1000):
            x = npr.random(Q.n) < 0.5
            Δy = Q.dx(x)
            ix, = np.where(Δy > 0)
            EΔy += Δy[ix].sum()
            k += ix.size
        EΔy /= k
        initial_acc_prob = 0.99
        init_temp = -EΔy / np.log(initial_acc_prob)
        print(f'Init. temp. automatically set to {init_temp:.4f}')

    # setup cooling schedule
    if schedule == 'e+':
        temps = init_temp/(1+np.exp(2*np.log(init_temp)*(np.linspace(0, 1, steps+1)-0.5)))
    elif schedule == '2+':
        temps = init_temp*(1-np.linspace(0, 1, steps+1))**2
    elif schedule == 'e*':
        temps = init_temp*(0.5**(1/halftime))**np.arange(0, 1, steps+1)
    elif schedule == '2*':
        temps = init_temp/(1+(1/(halftime**2))*np.linspace(0, 1, steps+1)**2)
    else:
        raise ValueError('Unknown schedule; must be one of {e*, 2*, e+, 2+}.')

    x = (npr.random((n_parallel, Q.n)) < 0.5).astype(np.float64)
    y = Q(x)
    for temp in temps:
        z = npr.random((n_parallel, Q.n)) < (1 / Q.n)
        x_ = (x + z) % 2
        Δy = Q(x_) - y
        p = np.minimum(np.exp(-Δy / temp), 1)
        a = npr.random(n_parallel) < p
        x = x + (x_ - x) * a[:, None]
        y = y + Δy * a

    srt = np.argsort(y)
    return x[srt, :], y[srt]


def local_descent(Q: qubo, x=None, random_state=None):
    """Starting from a given bit vector, find improvements in the 1-neighborhood
    and follow them until a local optimum is found. If no initial vector is
    specified, a random vector is sampled. At each step, the method greedily
    flips the bit that yields the greatest energy improvement.

    Args:
        Q (qubo): QUBO instance.
        x (numpy.ndarray, optional): Initial bit vector. Defaults to None.
        random_state (optional): A numerical or lexical seed, or a NumPy random
            generator. Defaults to None.

    Returns:
        A tuple containing the bit vector (numpy.ndarray) with lowest energy
        found, and its energy (float).
    """
    if x is None:
        rng = get_random_state(random_state)
        x_ = rng.random(Q.n) < 0.5
    else:
        x_ = x.copy()
    while True:
        Δx = Q.dx(x_)
        am = np.argmin(Δx)
        if Δx[am] >= 0:
            break
        x_[am] = 1 - x_[am]
    return x_, Q(x_)


def local2_descent(Q: qubo, x=None, random_state=None):
    """Starting from a given bit vector, find improvements in the 2-neighborhood
    and follow them until a local optimum is found. If no initial vector is
    specified, a random vector is sampled. At each step, the method greedily
    flips up to two bits that yield the greatest energy improvement.

    Args:
        Q (qubo): QUBO instance.
        x (numpy.ndarray, optional): Initial bit vector. Defaults to None.
        random_state (optional): A numerical or lexical seed, or a NumPy random
            generator. Defaults to None.

    Returns:
        A tuple containing the bit vector (numpy.ndarray) with lowest energy
        found, and its energy (float).
    """
    if x is None:
        rng = get_random_state(random_state)
        x_ = rng.random(Q.n) < 0.5
    else:
        x_ = x.copy()
    Δx = Q.dx2(x_) # (n, n) matrix
    i, j = np.unravel_index(np.argmin(Δx), Δx.shape)
    while True:
        Δx = Q.dx2(x_)
        i, j = np.unravel_index(np.argmin(Δx), Δx.shape)
        if Δx[i, j] >= 0:
            break
        flip_index(x_, [i, j], in_place=True)
    return x_, Q(x_)


def local_descent_search(Q: qubo, steps=1000, random_state=None):
    """Perform local descent in a multistart fashion and return the lowest
    observed bit vector. Use the 1-neighborhood as search radius.

    Args:
        Q (qubo): QUBO instance.
        steps (int, optional): Number of multistarts. Defaults to 1000.
        random_state (optional): A numerical or lexical seed, or a NumPy random
            generator. Defaults to None.

    Returns:
        A tuple containing the bit vector (numpy.ndarray) with lowest energy
        found, and its energy (float).
    """
    rng = get_random_state(random_state)
    x_min = np.empty(Q.n)
    y_min = np.infty
    x = np.empty(Q.n)
    for _ in range(steps):
        x[:] = rng.random(Q.n) < 0.5
        while True:
            Δx = Q.dx(x) # (n,) vector
            am = np.argmin(Δx, axis=-1)
            if Δx[am] >= 0:
                break
            x[am] = 1 - x[am]
        y = Q(x)
        if y <= y_min:
            x_min[:] = x
            y_min = y
    return x_min, y_min


def local2_descent_search(Q: qubo, steps=1000, random_state=None):
    """Perform local descent in a multistart fashion and return the lowest
    observed bit vector. Use the 2-neighborhood as search radius.

    Args:
        Q (qubo): QUBO instance.
        steps (int, optional): Number of multistarts. Defaults to 1000.
        random_state (optional): A numerical or lexical seed, or a NumPy random
            generator. Defaults to None.

    Returns:
        A tuple containing the bit vector (numpy.ndarray) with lowest energy
        found, and its energy (float).
    """
    rng = get_random_state(random_state)
    x_min = np.empty(Q.n)
    y_min = np.infty
    x = np.empty(Q.n)
    for _ in range(steps):
        x[:] = rng.random(Q.n) < 0.5
        while True:
            Δx = Q.dx2(x) # (n, n) matrix
            i, j = np.unravel_index(np.argmin(Δx), Δx.shape)
            if Δx[i, j] >= 0:
                break
            flip_index(x, [i, j], in_place=True)
        y = Q(x)
        if y <= y_min:
            x_min[:] = x
            y_min = y
    return x_min, y_min


def random_search(Q: qubo, steps=100_000, n_parallel=None, random_state=None):
    """Perform a random search in the space of bit vectors and return the
    lowest-energy solution found.

    Args:
        Q (qubo): QUBO instance.
        steps (int, optional): Number of steps to perform. Defaults to 100_000.
        n_parallel (int, optional): Number of random bit vectors to sample at a
            time. This does *not* increase the number of bit vectors sampled in
            total (specified by ``steps``), but makes the procedure faster by
            using NumPy vectorization. Defaults to None, which chooses a value
            such that the resulting bit vector array has about 32k elements.
        random_state (optional): A numerical or lexical seed, or a NumPy random
            generator. Defaults to None.

    Returns:
        A tuple containing the bit vector (numpy.ndarray) with lowest energy
        found, and its energy (float).
    """
    rng = get_random_state(random_state)
    if n_parallel is None:
        n_parallel = 32_000 // Q.n
    x_min = np.empty(Q.n)
    y_min = np.infty
    remaining = steps
    x = np.empty((n_parallel, Q.n))
    y = np.empty(n_parallel)
    while remaining > 0:
        r = min(remaining, n_parallel)
        x[:r] = rng.random((r, Q.n)) < 0.5
        y[:] = Q(x)
        i_min = np.argmin(y)
        if y[i_min] < y_min:
            x_min[:] = x[i_min, :]
            y_min = y[i_min]
        remaining -= r
    return x_min, y_min


def subspace_search(Q: qubo, steps=1000, random_state=None):
    """Perform search heuristic where :math:`n-\\log_2(n)` randomly selected
    variables are fixed and the remaining :math:`\\log_2(n)` bits are solved by
    brute force. The current solution is updated with the optimal sub-vector
    assignment, and the process is repeated.

    Args:
        Q (qubo): QUBO instance.
        steps (int, optional): Number of repetitions. Defaults to 1000.
        random_state (optional): A numerical or lexical seed, or a NumPy random
            generator. Defaults to None.

    Returns:
        A tuple containing the minimizing vector (numpy.ndarray) and the minimal
        energy (float).
    """
    rng = get_random_state(random_state)
    log_n = int(np.log2(Q.n))
    variables = np.arange(Q.n).astype(int)
    # sample random initial solution
    x = (rng.random(Q.n) < 0.5).astype(np.float64)
    for _ in range(steps):
        # fix random subset of n - log(n) variables
        rng.shuffle(variables)
        fixed = variables[:Q.n-log_n]
        Q_sub, _, free = Q.clamp(dict(zip(fixed, x[fixed])))
        # find optimum in subspace by brute force
        x_sub_opt, *_ = _brute_force_c(Q_sub.m)
        # set variables in current solution to subspace-optimal bits
        x[free] = x_sub_opt
    return x, Q(x)

import numpy as np

from ._misc import get_random_state, warn_size
from .qubo import qubo
from _c_utils import brute_force as _brute_force_c


def brute_force(Q: qubo):
    """Solve QUBO instance exactly by brute force.
    Note that this method is infeasible for instances
    with a size beyond around 30.

    Args:
        Q (qubo): QUBO instance to solve.

    Raises:
        ValueError: Raised if the QUBO size is too large to be brute-forced on the present system.

    Returns:
        A tuple containing the minimizing vector (numpy.ndarray) and the minimal energy (float).
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
    """Performs simulated annealing to approximate the minimizing
    vector and minimal energy of a given QUBO instance.

    Args:
        Q (qubo): QUBO instance.
        schedule (str, optional): The annealing schedule to employ.
            Possible values are: ``2+`` (quadratic additive), ``2*`` (quadratic multiplicative),
            ``e+`` (exponential additive) and ``e*`` (exponential multiplicative).
            See `here <http://what-when-how.com/artificial-intelligence/a-comparison-of-cooling-schedules-for-simulated-annealing-artificial-intelligence/>`__  for further infos. Defaults to '2+'.
        halftime (float, optional): For multiplicative schedules only:
            The percentage of steps after which the temperature is halved. Defaults to 0.25.
        steps (int, optional): Number of annealing steps to perform. Defaults to 100_000.
        init_temp (float, optional): Initial temperature. Defaults to None, which estimates an initial temperature.
        n_parallel (int, optional): Number of random initial solutions to anneal simultaneously. Defaults to 10.
        random_state (optional): A numerical or lexical seed, or a NumPy random generator. Defaults to None.

    Raises:
        ValueError: Raised if the specificed schedule is unknown.

    Returns:
        A tuple ``(x, y)`` containing the solution bit vectors and their respective energies.
        The shape of ``x`` is ``(n_parallel, n)``, where ``n`` is the QUBO size; the shape of ``y``
        is ``(n_parallel,)``. Bit vector ``x[i]`` has energy ``y[i]`` for each ``i``.
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
        temps = init_temp / (
                    1 + np.exp(2 * np.log(init_temp) * (np.linspace(0, 1, steps + 1) - 0.5)))
    elif schedule == '2+':
        temps = init_temp * (1 - np.linspace(0, 1, steps + 1)) ** 2
    elif schedule == 'e*':
        temps = init_temp * (0.5 ** (1 / halftime)) ** np.arange(0, 1, steps + 1)
    elif schedule == '2*':
        temps = init_temp / (1 + (1 / (halftime ** 2)) * np.linspace(0, 1, steps + 1) ** 2)
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
    """Starting from a given bit vector, find improvements in the
    1-neighborhood and follow them until a local optimum is found.
    If no initial vector is specified, a random vector is sampled.
    At each step, the method greedily flips the bit that yields
    greatest energy improvement.

    Args:
        Q (qubo): QUBO instance.
        x (numpy.ndarray, optional): Initial bit vector. Defaults to None.
        random_state (optional): A numerical or lexical seed, or a NumPy random generator. Defaults to None.

    Returns:
        A tuple containing the minimizing vector (numpy.ndarray) and the minimal energy (float).
    """
    if x is None:
        rng = get_random_state(random_state)
        x_ = rng.random(Q.n) < 0.5
    else:
        x_ = x.copy()
    Δx = Q.dx(x_)
    am = np.argmin(Δx)
    while Δx[am] < 0:
        x_[am] = 1 - x_[am]
        Δx = Q.dx(x_)
        am = np.argmin(Δx)
    return x_, Q(x_)


def random_search(Q: qubo, steps=100_000, n_parallel=None, random_state=None):
    """Perform a random search in the space of bit vectors and return
    the lowest-energy solution found.

    Args:
        Q (qubo): QUBO instance.
        steps (int, optional): Number of steps to perform. Defaults to 100_000.
        n_parallel (int, optional): Number of random bit vectors to sample at a time.
            This does *not* increase the number of bit vectors sampled in total
            (specified by ``steps``), but makes the procedure faster by using NumPy
            vectorization. Defaults to None, which chooses a value such that the
            resulting bit vector array has about 32k elements.
        random_state (optional): A numerical or lexical seed, or a NumPy random generator. Defaults to None.

    Returns:
        A tuple containing the minimizing vector (numpy.ndarray) and the minimal energy (float).
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

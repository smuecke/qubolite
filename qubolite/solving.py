import numpy as np

from _c_utils import brute_force as _brute_force_c
from ._misc import get_random_state, warn_size
from .qubo import qubo


def brute_force(Q: qubo):
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

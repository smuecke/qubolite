from heapq import nsmallest

import numpy as np

from .bitvec import all_bitvectors
from .misc import get_random_state, warn_size
from .qubo import qubo


def brute_force(Q: qubo, k=1, return_value=False):
    warn_size(Q.n, limit=20)
    if k == 1:
        x = min(all_bitvectors(Q.n, read_only=False), key=Q)
        return (x, Q(x)) if return_value else x
    elif k > 1:
        xs = nsmallest(k, all_bitvectors(Q.n, read_only=False), key=Q)
        return list(zip(xs, map(Q, xs))) if return_value else xs
    else:
        return ValueError(f'k must be greater than 0')


def get_binary_number_matrix(n):
    """Fast method for creating all binary states with dimension n"""
    if n == 1:
        return np.array([[0], [1]])
    B = np.zeros((2 ** n, n))
    B_ = get_binary_number_matrix(n=n - 1)
    B[:2 ** (n - 1), 1:] = B_
    B[2 ** (n - 1):, 0] = 1
    B[2 ** (n - 1):, 1:] = B_
    return B


def brute_force_new(Q: qubo, k=1, return_value=False, X=None):
    warn_size(Q.n, limit=20)
    if X is None:
        X = get_binary_number_matrix(Q.n)
    elif X == "bitvec":
        X = np.vstack(list(all_bitvectors(Q.n, read_only=False)))
    Q_values = Q(X)
    if k == 1:
        best_index = np.argmin(Q_values)
        x = X[best_index]
        return (x, Q_values[best_index]) if return_value else x
    elif 1 < k <= 2 * Q.n:  # runtime of np.argpartition is O(n * k)
        best_indices = np.argpartition(Q_values, np.arange(k))[:k]
        xs = X[best_indices]
        return list(zip(xs, Q_values[best_indices])) if return_value else xs
    elif k > 2 * Q.n:  # runtime of np.argsort is O(n * log n)
        best_indices = np.argsort(Q_values)[:k]
        xs = X[best_indices]
        return list(zip(xs, Q_values[best_indices])) if return_value else xs
    else:
        return ValueError(f'k must be greater than 0')


def simulated_annealing(Q: qubo, schedule='2+', steps=100_000, init_temp=None, n_parallel=10,
                        random_state=None, halftime=0.25):
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


def local_descent(Q: qubo, x):
    x_ = x.copy()
    Δx = Q.dx(x_)
    am = np.argmin(Δx)
    while Δx[am] < 0:
        x_[am] = 1 - x_[am]
        Δx = Q.dx(x_)
        am = np.argmin(Δx)
    return x_, Q(x_)

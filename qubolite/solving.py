from heapq import nsmallest

import numpy as np

from .misc import all_bitvectors, get_random_state, warn_size
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
        

def simulated_annealing(Q: qubo, steps=100_000, init_temp=1000, n_parallel=10, random_state=None):
    npr = get_random_state(random_state)
    x = (npr.random((n_parallel, Q.n)) < 0.5).astype(np.float64)
    y = Q(x)
    
    #expected_y = 0.25*(Q.m+Q.m.T).sum()
    for t in range(steps+1):
        temp = init_temp*((steps-t)/steps)**2 # quadratic annealing
        z  = npr.random((n_parallel, Q.n)) < (1/Q.n)
        x_ = (x+z)%2
        Δy = Q(x_)-y
        p  = np.minimum(np.exp(-Δy/temp), 1)
        a  = npr.random(n_parallel) < p  
        x = x+(x_-x)*a[:, None]
        y = y+Δy*a

    srt = np.argsort(y)
    return x[srt, :], y[srt]


def local_descent(Q: qubo, x):
    x_ = x.copy()
    Δx = Q.dx(x_)
    am = np.argmin(Δx)
    while Δx[am] < 0:
        x_[am] = 1-x_[am]
        Δx = Q.dx(x_)
        am = np.argmin(Δx)
    return x_, Q(x_)

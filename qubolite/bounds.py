from dataclasses import dataclass

import numpy as np
from networkx import DiGraph, set_edge_attributes
from networkx.algorithms.flow import maximum_flow_value

from .misc import get_random_state
from .qubo import qubo


@dataclass
class QuadraticPosiform:
    P: np.ndarray
    const: float=0

    @classmethod
    def from_qubo(cls, Q: qubo):
        posiform = np.zeros((2, Q.n, Q.n))
        # posiform[0] contains terms xi* xj, and  xi on diagonal
        # posiform[1] contains terms xi*!xj, and !xi on diagonal
        lin = np.diag(Q.m)
        qua = np.triu(Q.m, 1)
        diag_ix = np.diag_indices_from(Q.m)
        qua_neg = np.minimum(qua, 0)
        posiform[0] = np.maximum(qua, 0)
        posiform[1] = -qua_neg
        posiform[0][diag_ix] = lin + qua_neg.sum(1)
        lin_ = posiform[0][diag_ix].copy() # =: c'
        lin_neg = np.minimum(lin_, 0)
        posiform[1][diag_ix] = -lin_neg
        posiform[0][diag_ix] = np.maximum(lin_, 0)
        const = lin_neg.sum()
        return cls(posiform, const)

    def to_flow_graph(self):
        G = DiGraph()
        for i, j in zip(*np.triu_indices_from(self.P[0])):
            ni = 'x0' if i == j else f'x{i+1}'
            nj = f'x{j+1}'

            if not np.isclose(self.P[0, i, j], 0):
                γ = 0.5*self.P[0, i, j]
                G.add_edge(ni, f'!{nj}', capacity=γ)
                G.add_edge(nj, f'!{ni}', capacity=γ)
            if not np.isclose(self.P[1, i, j], 0):
                γ = 0.5*self.P[1, i, j]
                G.add_edge(ni, nj, capacity=γ)
                G.add_edge(f'!{nj}', f'!{ni}', capacity=γ)

        set_edge_attributes(G, 0.0, name='flow')
        return G
    
    

def lb_roof_dual(Q: qubo):
    P = QuadraticPosiform.from_qubo(Q)
    G = P.to_flow_graph()
    v = maximum_flow_value(G, 'x0', '!x0')
    return P.const + v


def lb_negative_parameters(Q: qubo):
    return np.minimum(Q.m, 0).sum()


# upper bounds ------------------------

def ub_sample(Q: qubo, samples=1000, random_state=None):
    npr = get_random_state(random_state)
    min_val = float('inf')
    for _ in range(samples):
        val = Q(npr.random(Q.n)<0.5)
        min_val = min(min_val, val)
    return min_val


def ub_local_search(Q: qubo, restarts=10, random_state=None):
    npr = get_random_state(random_state)
    min_val = float('inf')
    for _ in range(restarts):
        x = (npr.random(Q.n)<0.5).astype(np.float64)
        dQdx = Q.dx(x)
        i = np.argmin(dQdx)
        while dQdx[i] < 0:
            x[i] = 1-x[i]
            dQdx = Q.dx(x)
            i = np.argmin(dQdx)
        min_val = min(min_val, Q(x))
    return min_val

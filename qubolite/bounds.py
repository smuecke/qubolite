import numpy as np

from .misc import get_random_state, warn
from .qubo import qubo


def lb_roof_dual_nx(Q: qubo):
    warn("lb_roof_dual_nx() is deprecated since it is based on a networkx implementation which is "
         "written purely in python and thus very slow. Use lb_roof_dual() instead",
         DeprecationWarning)
    try:
        from networkx import DiGraph, set_edge_attributes
        from networkx.algorithms.flow import maximum_flow_value
    except ImportError as e:
        raise ImportError(
            "networkx needs to be installed prior to running qubolite.lb_roof_dual_nx(). You can "
            "install networkx with:\n'pip install networkx'"
        ) from e

    def to_flow_graph(P):
        G = DiGraph()
        for i, j in zip(*np.triu_indices_from(P[0])):
            ni = 'x0' if i == j else f'x{i + 1}'
            nj = f'x{j + 1}'

            if not np.isclose(P[0, i, j], 0):
                γ = 0.5 * P[0, i, j]
                G.add_edge(ni, f'!{nj}', capacity=γ)
                G.add_edge(nj, f'!{ni}', capacity=γ)
            if not np.isclose(P[1, i, j], 0):
                γ = 0.5 * P[1, i, j]
                G.add_edge(ni, nj, capacity=γ)
                G.add_edge(f'!{nj}', f'!{ni}', capacity=γ)

        set_edge_attributes(G, 0.0, name='flow')
        return G

    P, const = Q.to_posiform()
    G = to_flow_graph(P)
    v = maximum_flow_value(G, 'x0', '!x0')
    return const + v


def lb_roof_dual(Q: qubo):
    try:
        from igraph import Graph
    except ImportError as e:
        raise ImportError(
            "igraph needs to be installed prior to running qubolite.lb_roof_dual(). You can "
            "install igraph with:\n'pip install igraph'"
        ) from e

    def to_flow_graph(P):
        n = P.shape[1]
        G = Graph(directed=True)
        vertices = np.arange(n + 1)
        negated_vertices = np.arange(n + 1, 2 * n + 2)
        # all vertices for flow graph
        all_vertices = np.concatenate([vertices, negated_vertices])
        G.add_vertices(all_vertices)
        # arrays of vertices containing node x0
        n0 = np.kron(vertices[1:][:, np.newaxis], np.ones(n, dtype=int))
        np.fill_diagonal(n0, np.zeros(n))
        nn0 = np.kron(negated_vertices[1:][:, np.newaxis], np.ones(n, dtype=int))
        np.fill_diagonal(nn0, (n + 1) * np.ones(n))
        # arrays of vertices not containing node x0
        n1 = np.kron(np.ones(n, dtype=int)[:, np.newaxis], vertices[1:])
        nn1 = np.kron(np.ones(n, dtype=int)[:, np.newaxis], negated_vertices[1:])

        n0_nn1 = np.stack((n0, nn1), axis=-1) # edges from ni to !nj
        n1_nn0 = np.stack((n1, nn0), axis=-1) # edges from nj to !ni
        n0_n1 = np.stack((n0, n1), axis=-1) # edges from ni to nj
        nn1_nn0 = np.stack((nn1, nn0), axis=-1) # edges from !nj to !ni
        pos_indices = np.invert(np.isclose(P[0], 0))
        neg_indices = np.invert(np.isclose(P[1], 0))
        # set capacities to half of posiform parameters
        capacities = 0.5 * np.concatenate([P[0][pos_indices], P[0][pos_indices],
                                           P[1][neg_indices], P[1][neg_indices]])
        edges = np.concatenate([n0_nn1[pos_indices], n1_nn0[pos_indices],
                                n0_n1[neg_indices], nn1_nn0[neg_indices]])
        G.add_edges(edges)
        return G, capacities

    P, const = Q.to_posiform()
    G, capacities = to_flow_graph(P)
    v = G.maxflow_value(0, Q.n + 1, capacity=list(capacities))
    return const + v


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

import numpy as np

from ._misc   import get_random_state
from .qubo    import qubo
from .solving import random_search, local_descent


def lb_roof_dual(Q: qubo):
    """Compute the Roof Dual bound, as described in `[1] <https://www.researchgate.net/publication/238379061_Preprocessing_of_unconstrained_quadratic_binary_optimization>`__.
    To this end, the QUBO instance is converted to a
    corresponding flow network, whose maximum flow value
    yields the roof dual lower bound.

    Args:
        Q (qubo): QUBO instance.

    Raises:
        ImportError: Raised if the ``igraph`` package is missing.
            This package is required for this method.

    Returns:
        float: A lower bound on the minimal energy value.
    """
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
    """Compute a simple lower bound on the minimal energy
    by summing up all negative parameters. As all QUBO
    energy values are sums of parameter subsets, the
    smallest subset sum is a lower bound for the minimal energy.
    This bound is very fast, but very weak, especially for large
    QUBO sizes.

    Args:
        Q (qubo): QUBO instance.

    Returns:
        float: A lower bound on the minimal energy value.
    """
    return np.minimum(Q.m, 0).sum()


# upper bounds ------------------------

def ub_sample(Q: qubo, samples=10_000, random_state=None):
    """Compute an upper bound on the minimal energy by sampling
    and taking the minimal encountered value.

    Args:
        Q (qubo): QUBO instance.
        samples (int, optional): Number of samples. Defaults to 10_000.
        random_state (optional): A numerical or lexical seed, or a NumPy random generator. Defaults to None.

    Returns:
        float: An upper bound on the minimal energy value.
    """
    _, v = random_search(Q, steps=samples, random_state=random_state)
    return v


def ub_local_descent(Q: qubo, restarts=10, random_state=None):
    """Compute an upper bound on the minimal energy by repeatedly
    performing a local optimization heuristic and returning the
    lowest energy value encountered.

    Args:
        Q (qubo): QUBO instance.
        restarts (int, optional): Number of local searches with
            random initial bit vectors. Defaults to 10.
        random_state (optional): A numerical or lexical seed, or a NumPy random generator. Defaults to None.

    Returns:
        float: An upper bound on the minimal energy value.
    """
    npr = get_random_state(random_state)
    min_val = float('inf')
    for _ in range(restarts):
        _, v = local_descent(Q, random_state=npr)
        min_val = min(min_val, v)
    return min_val

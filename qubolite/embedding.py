from functools import partial

import numpy as np
from sklearn.metrics import pairwise_kernels
from sklearn.preprocessing import KernelCenterer

from ._misc import get_random_state
from .qubo import qubo


class qubo_embedding:
    def map_solution(self, x):
        return NotImplemented

    @property
    def qubo(self):
        return NotImplemented

    @property
    def data(self):
        return NotImplemented

    @classmethod
    def random(cls, n: int, random_state=None):
        return NotImplemented


class BinaryClustering(qubo_embedding):
    """Simple Binary Clustering based on Euclidean distance.
    Minimizes distances within and maximizes distances
    between the clusters.
    """
    def __init__(self, data):
        self.__data = data
        self.__Q = self.__from_data(data)

    @property
    def qubo(self):
        return self.__Q

    @property
    def data(self):
        return dict(points=self.__data)

    def __from_data(self, data):
        # calculate euclidean distance matrix
        d = np.sqrt(((data[:, None] - data) ** 2).sum(-1))
        q = np.triu(3 * d, 1)
        q[np.diag_indices_from(q)] = -d.sum(1)
        return qubo(q)

    def map_solution(self, x):
        # return cluster assignments (-1, +1)
        return 2 * x - 1

    @classmethod
    def random(cls, n: int, dim=2, dist=2.0, random_state=None):
        npr = get_random_state(random_state)
        data = npr.normal(size=(n, dim))
        mask = npr.permutation(n) < n // 2
        data[mask, :] += dist / np.sqrt(dim)
        return cls(data)


class Kernel2MeansClustering(qubo_embedding):
    """Binary clustering based on kernel matrices.

    Args:
        data (numpy.ndarray): Data array of shape ``(n, m)``.
            The QUBO instance will have size ``n`` (or ``n-1``, if ``unambiguous=True``).
        kernel (str, optional): Kernel function. Defaults to ``'linear'``.
            Can be any of `those <https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d6e0a3e8ddf92a7e5561245224dab102/sklearn/metrics/pairwise.py#L2216>`__.
        centered (bool, optional): If ``True``, center the Kernel matrix. Defaults to ``True``.
        unambiguous (bool, optional): If ``True``, assign the last data point to cluster 0 and exclude it
            from the optimization. Otherwise, the resulting QUBO instance would have two
            symmetrical optimal cluster assignments. Defaults to False.
        **kernel_params: Additional keyword arguments for the Kernel function, passed to ``sklearn.metrics.pairwise_kernels``.
    """
    def __init__(self, data, kernel='linear', centered=True, unambiguous=False, **kernel_params):
        # for different kernels: https://scikit-learn.org/stable/modules/metrics.html
        self.__data = data
        self.__kernel = kernel
        self.__kernel_params = kernel_params
        self.__centered = centered
        self.__unambiguous = unambiguous
        self.__Q = self.__from_data()

    @property
    def qubo(self):
        return self.__Q

    @property
    def data(self):
        return dict(points=self.__data)

    def __from_data(self):
        # calculate kernel matrix
        K = pairwise_kernels(X=self.__data, metric=self.__kernel, **self.__kernel_params)
        # center kernel matrix
        if self.__centered:
            K = KernelCenterer().fit_transform(K)
        q = -K
        np.fill_diagonal(q, K.sum(1) - K.diagonal())
        # fix z_n=0 for cluster assignment
        if self.__unambiguous:
            n = K.shape[0]
            q = q[:n-1, :n-1]
        return qubo(q)

    def map_solution(self, x):
        # return cluster assignments (-1, +1)
        return 2 * x - 1

    @classmethod
    def random(cls, n: int, dim=2, dist=2.0, random_state=None, kernel=None, centered=True,
               unambiguous=True, **kernel_params):
        if kernel is None:
            kernel = 'linear'
            kernel_params = {}
        npr = get_random_state(random_state)
        data = npr.normal(size=(n, dim))
        mask = npr.permutation(n) < n // 2
        data[mask, :] += dist / np.sqrt(dim)
        data -= data.mean(0)
        return data, cls(data, kernel=kernel, centered=centered, unambiguous=unambiguous,
                         **kernel_params)


class SubsetSum(qubo_embedding):
    """Subset Sum problem: Given a list of values, find
    a subset that adds up to a given target value.

    Args:
        values (numpy.ndarray | list): Values of which to find a subset.
            The resulting QUBO instance will have size ``len(values)``.
        target (int | float): Target value which the subset must add up to.
    """
    def __init__(self, values, target):
        self.__values = np.asarray(values)
        self.__Q = self.__from_data(self.__values, target)

    @property
    def qubo(self):
        return self.__Q

    @property
    def data(self):
        return dict(
            values=self.__values,
            target=self.__target)

    def __from_data(self, values, target):
        q = np.outer(values, values)
        q[np.diag_indices_from(q)] -= 2 * target * values
        q = np.triu(q + np.tril(q, -1).T)
        return qubo(q)

    def map_solution(self, x):
        return self.__values[x.astype(bool)]

    @classmethod
    def random(cls, n: int, low=0, high=10, summands=None, random_state=None):
        npr = get_random_state(random_state)
        values = npr.uniform(low, high, size=n)
        k = np.arange(2, n + 1) if summands is None else summands
        subset = npr.permutation(n) < k
        target = values[subset].sum()
        return cls(values, target)


class Max2Sat(qubo_embedding):
    """Maximum Satisfyability problem with clauses of size 2.
    The problem is to find a variable assignment that maximizes
    the number of true clauses.

    Args:
        clauses (list): A list of tuples containing literals,
            representing a logical formula in CNF.
            Each tuple must have exactly two elements.
            The elements must be integers, representing the variable
            indices **counting from 1**. Negative literals have a
            negative sign. For instance, the formula :math:`(x_1\\vee \\overline{x_2})\\wedge(\\overline{x_1}\\vee x_3)`
            becomes ``[(1,-2), (-1,3)]``.
        penalty (float, optional): Penalty value for unsatisfied clauses.
            Must be positive. Defaults to ``1.0``.
    """
    def __init__(self, clauses, penalty=1.0):
        assert all(len(c) == 2 for c in clauses), 'All clauses must consist of exactly 2 variables'
        assert all(0 not in c for c in clauses), '"0" cannot be a variable, use indices >= 1'
        self.__clauses = clauses
        ix_set = set()
        ix_set.update(*self.__clauses)
        self.__indices = [i for i in sorted(ix_set) if i > 0]
        assert penalty > 0.0, 'Penalty must be positive > 0'
        self.__penalty = penalty

    def map_solution(self, x):
        return {i: x[self.__indices.find(i)] == 1 for i in self.__indices}

    @property
    def qubo(self):
        n = max(max(c) for c in self.__clauses)
        m = np.zeros((n, n))
        ix_map = {i: qi for qi, i in enumerate(self.__indices)}
        for xi, xj in map(partial(sorted, key=abs), self.__clauses):
            i = ix_map(abs(xi))
            j = ix_map(abs(xj))
            if xi > 0:
                if xj > 0:
                    m[i, i] += self.__penalty
                    m[j, j] += self.__penalty
                    m[i, j] -= self.__penalty
                else:
                    m[j, j] += self.__penalty
                    m[i, j] -= self.__penalty
            else:
                if xj > 0:
                    m[i, i] += self.__penalty
                    m[i, j] -= self.__penalty
                else:
                    m[i, j] += self.__penalty
        return qubo(m)

    @property
    def data(self):
        return self.__clauses

    @classmethod
    def random(cls, n: int, clauses=None, random_state=None):
        npr = get_random_state(random_state)
        if clauses is None:
            clauses = int(1.5 * n)
        # TODO
        return NotImplemented

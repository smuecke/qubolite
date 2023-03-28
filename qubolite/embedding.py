from functools import partial

import numpy as np

from .misc import get_random_state
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


class Kernel:
    def __call__(self, x: np.ndarray):
        return NotImplementedError

    @staticmethod
    def center_kernel_matrix(K):
        n = K.shape[0]
        ones = np.ones(n)
        n_frac = 1.0 / n
        left = ones @ K
        return K - n_frac * left - n_frac * K @ ones - n_frac ** 2 * left @ ones


class GaussianKernel(Kernel):
    def __init__(self, delta=1.0):
        self.__delta = delta

    def __call__(self, X: np.ndarray, Y: np.ndarray = None):
        if Y is None:
            Y = X
        if X.ndim == 1:
            assert len(X) == X.shape[0]
            D = ((X - Y) ** 2).sum(-1)
        else:
            assert X.ndim == 2 and Y.ndim == 2 and X.shape[1] == Y.shape[1]
            D = ((X[:, None] - Y) ** 2).sum(-1)
        return np.exp(- self.__delta * D)


class LaplacianKernel(Kernel):
    def __init__(self, delta=1.0):
        self.__delta = delta

    def __call__(self, X: np.ndarray, Y: np.ndarray = None):
        if Y is None:
            Y = X
        if X.ndim == 1:
            assert len(X) == X.shape[0]
            D = (np.abs(X - Y)).sum(-1)
        else:
            assert X.ndim == 2 and Y.ndim == 2 and X.shape[1] == Y.shape[1]
            D = (np.abs(X[:, None] - Y)).sum(-1)
        return np.exp(- self.__delta * D)


class ProductKernel(Kernel):
    def __call__(self, X: np.ndarray, Y: np.ndarray = None):
        if Y is None:
            Y = X
        if X.ndim == 1:
            assert len(X) == X.shape[0]
        else:
            assert X.ndim == 2 and Y.ndim == 2 and X.shape[1] == Y.shape[1]
        return (X @ Y.T).T


class Kernel2MeansClustering(qubo_embedding):
    def __init__(self, data, kernel: Kernel = None):
        self.__data = data
        if kernel is None:
            self.__kernel = GaussianKernel()
        else:
            self.__kernel = kernel
        self.__Q = self.__from_data(data, kernel)

    @property
    def qubo(self):
        return self.__Q

    @property
    def data(self):
        return dict(points=self.__data)

    @property
    def kernel(self):
        return self.__kernel

    def __from_data(self, data, kernel, centered=True, unambiguous=True):
        # calculate kernel matrix
        K = kernel(X=data)
        # center kernel matrix
        if centered:
            K = kernel.center_kernel_matrix(K)
        q = -K
        np.fill_diagonal(q, K.sum(1) - K.diagonal())
        # fix z_n=0 for cluster assignment
        if unambiguous:
            n = K.shape[0]
            q = q[:n-1, :n-1]
        return qubo(q)

    def map_solution(self, x):
        # return cluster assignments (-1, +1)
        return 2 * x - 1

    @classmethod
    def random(cls, n: int, dim=2, dist=2.0, random_state=None,
               kernel: Kernel = None):
        if kernel is None:
            kernel = GaussianKernel()
        npr = get_random_state(random_state)
        data = npr.normal(size=(n, dim))
        mask = npr.permutation(n) < n // 2
        data[mask, :] += dist / np.sqrt(dim)
        data -= data.mean(0)
        return data, cls(data, kernel=kernel)


class SubsetSum(qubo_embedding):
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

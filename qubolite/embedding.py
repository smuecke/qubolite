import numpy as np
from bitvec import with_norm
from seedpy import get_random_state

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
    def sample(cls, n: int, random_state=None):
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
        d = np.sqrt(((data[:,None]-data)**2).sum(-1))
        q = np.triu(3*d, 1)
        q[np.diag_indices_from(q)] = -d.sum(1)
        return qubo(q)

    def map_solution(self, x):
        # return cluster assignments (-1, +1)
        return 2*x-1

    @classmethod
    def sample(cls, n: int, dim=2, dist=2.0, random_state=None):
        npr = get_random_state(random_state)
        data = npr.normal(size=(n, dim))
        mask = npr.permutation(n) < n//2
        data[mask, :] += dist/np.sqrt(dim)
        return cls(data)


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
        q[np.diag_indices_from(q)] -= 2*target*values
        q = np.triu(q + np.tril(q, -1).T)
        return qubo(q)

    def map_solution(self, x):
        return self.__values[x.astype(bool)]

    @classmethod
    def sample(cls, n: int, low=0, high=10, summands=None, random_state=None):
        npr = get_random_state(random_state)
        values = npr.uniform(low, high, size=n)
        k = np.arange(2, n+1) if summands is None else summands
        subset = with_norm(n, k)().astype(bool)
        target = values[subset].sum()
        return cls(values, target)
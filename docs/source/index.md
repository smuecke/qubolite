```{toctree}
:hidden:

self
api
Source Code <https://github.com/smuecke/qubolite>
```

# Quickstart

This package provides tools for working with *Quadratic Unconstrained Binary Optimization* ([QUBO](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization)) problems. These problems are central for Quantum Annealing and have numerous applications in Machine Learning, resource allocation, data analysis and many more.

Given a real-valued parameter matrix **Q**, the *energy* of a binary vector **x** is defined as

```{math}
    f_{\boldsymbol Q}(\boldsymbol x)=\sum_{1\leq i\leq j\leq n}Q_{ij}x_ix_j\;.
```

The task is to find a binary vector that minimizes this energy function.

The philosophy of this package is to be as light-weight as possible, therefore the core class `qubo` is just a shallow wrapper around a NumPy array containing the QUBO parameters, with many useful methods. Additionally, the package contains methods for **solving** QUBOs, **sampling** from their Gibbs distribution, **bounding** their minimum energy, and **embedding** other problems into QUBO.


## Installation

This package is available on PyPi and can be installed via

```
pip install qubolite
```

Note that this package contains code in C, which is why you need a working C compiler (GCC on Linux, Visual C/C++ on Windows).


## Documentation

The full **API documentation** can be found by clicking the link on the left, or [here](https://smuecke.github.io/qubolite/api.html).

The **source code** is publicly available on [GitHub](https://github.com/smuecke/qubolite).


## Usage Examples

The core class is `qubo`, which receives a `numpy.ndarray` of size `(n, n)`.
Alternatively, a random instance can be created using `qubo.random()`.

```
>>> import numpy as np
>>> from qubolite import qubo
>>> arr = np.triu(np.random.random((8, 8)))
>>> Q = qubo(arr)
>>> Q2 = qubo.random(12, distr='uniform')
```

By default, `qubo()` takes an upper triangle matrix.
A non-triangular matrix is converted to an upper triangle matrix by adding the lower to the upper triangle.
QUBO instances can also be created from dictionaries through `qubo.from_dict()`.

To get the QUBO function value, instances can be called directly with a bit vector.
The bit vector must be a `numpy.ndarray` of size `(n,)` or `(m, n)`.

```
>>> x = np.random.random(8) < 0.5
>>> Q(x)
7.488225478498116
>>> xs = np.random.random((5,8)) < 0.5 # evaluate 5 bit vectors at once
>>> Q(xs)
array([5.81642745, 4.41380893, 11.3391062, 4.34253921, 6.07799747])
```


### Solving

The submodule `solving` contains several methods to obtain the minimizing bit vector or energy value of a given QUBO instance, both exact and approximative.

```
>>> from qubolite.solving import brute_force
>>> x_min, value = brute_force(Q, return_value=True)
>>> x_min
array([1., 1., 1., 0., 1., 0., 0., 0.])
>>> value
-3.394893116198653
```

The method `brute_force` is implemented efficiently in C and parallelized with OpenMP.
Still, for instances with more than 30 variables take a long time to solve this way.
Other methods included in this package are Simulated Annealing and some heuristic search methods (local search, random search).


### Embedding

`qubolite` provides QUBO embeddings for common optimization problems.
For example, the following code shows how to solve a binary clustering problem:

```
>>> from qubolite.embedding import Kernel2MeansClustering
>>> X = np.random.normal(size=(30, 2)) # sample 2D points
>>> X[15:,0] += 5 # move half of the points apart to create two clusters
>>> problem = Kernel2MeansClustering(X, kernel='rbf')
>>> brute_force(problem.qubo)
(array([1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), -44.55740335019274)
```

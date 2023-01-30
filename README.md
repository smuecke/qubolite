# qubolite

A light-weight toolbox for working with QUBO instances in NumPy.


## Installation

```
pip install qubolite
```

This package was created using Python 3.10, but runs with Python >= 3.8.


## Usage Examples

By design, `qubolite` is a shallow wrapper around `numpy` arrays, which represent QUBO parameters.
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

To get the QUBO function value, instances can be called directly with a bit vector.
The bit vector must be a `numpy.ndarray` of size `(n,)` or `(m, n)`.

```
>>> x = np.random.random(8) < 0.5
>>> Q(x)
7.488225478498116
>>> xs = np.random.random((5,8)) < 0.5
>>> Q(xs)
array([5.81642745, 4.41380893, 11.3391062, 4.34253921, 6.07799747])
```


## Version Log

* **0.2** Added problem embeddings (binary clustering, subset sum problem)
* **0.3** Added `QUBOSample` class and sampling methods `full` and `gibbs`
* **0.4** Renamed `QUBOSample` to `BinarySample`; added methods for saving and loading QUBO and Sample instances
* **0.5** Moved `gibbs` to `mcmc` and implemented true Gibbs sampling as `gibbs`; added `numba` as dependency
    * **0.5.1** changed `keep_prob` to `keep_interval` in Gibbs sampling, making the algorithm's runtime deterministic; renamed `sample` to `random` in QUBO embedding classes, added MAX 2-SAT problem embedding
* **0.6** Changed Python version to 3.8; removed `bitvec` dependency; added `scipy` dependency required for matrix operations in numba functions
    * **0.6.1** added scaling and rounding
    * **0.6.2** removed `seedpy` dependency
    * **0.6.3** renamed `shots` to `size` in `BinarySample`; cleaned up sampling, simplified type hints
    * **0.6.4** added probabilistic functions to `qubo` class
    * **0.6.5** complete empirical prob. vector can be returned from `BinarySample`
    * **0.6.6** fixed spectral gap implementation
    * **0.6.7** moved `brute_force` to new sub-module `solving`; added some approximate solving methods

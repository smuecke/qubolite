# qubolite

A light-weight toolbox for working with QUBO instances in NumPy.


## Installation

```
pip install qubolite
```

This package was created using Python 3.10, but runs with Python >= 3.8.


## Version Log

* **0.2** Added problem embeddings (binary clustering, subset sum problem)
* **0.3** Added `QUBOSample` class and sampling methods `full` and `gibbs`
* **0.4** Renamed `QUBOSample` to `BinarySample`; added methods for saving and loading QUBO and Sample instances
* **0.5** Moved `gibbs` to `mcmc` and implemented true Gibbs sampling as `gibbs`; added `numba` as dependency
    * **0.5.1** changed `keep_prob` to `keep_interval` in Gibbs sampling, making the algorithm's runtime deterministic; renamed `sample` to `random` in QUBO embedding classes, added MAX 2-SAT problem embedding
* **0.6** Changed Python version to 3.8; removed `bitvec` dependency; added `scipy` dependency required for matrix operations in numba functions
    * **0.6.1** added scaling and rounding
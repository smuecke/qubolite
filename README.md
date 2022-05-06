# qubolite

A light-weight toolbox for working with QUBO instances in NumPy.


## Installation

This package was created using Python 3.10.
You need the `bitvec` package, which you can install by running `python3.10 -m pip install git+https://github.com/smuecke/bitvec.git` (it is not yet available on PyPi).


## Version Log

* **0.2** Added problem embeddings (binary clustering, subset sum problem)
* **0.3** Added `QUBOSample` class and sampling methods `full` and `gibbs`
* **0.4** Renamed `QUBOSample` to `BinarySample`; added methods for saving and loading QUBO and Sample instances
* **0.5** Moved `gibbs` to `mcmc` and implemented true Gibbs sampling (hopefully) as `gibbs`; added `numba` as dependency
    * **0.5.1** changed `keep_prob` to `keep_interval` in Gibbs sampling, making the algorithm's runtime deterministic; renamed `sample` to `random` in QUBO embedding classes, added MAX 2-SAT problem embedding

# API Documentation

This page documents the classes and methods contained in the `qubolite` package.


## QUBO

The base class for QUBO instances is `qubo`.

```{eval-rst}
.. autoclass:: qubolite.qubo
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
```


## Bit Vectors

The submodule `bitvec` contains useful methods for working with bit vectors, which in `qubolite` are just NumPy arrays containing the values `0.0` and `1.0`.

```{eval-rst}
.. automodule:: qubolite.bitvec
   :members:
   :undoc-members:
   :show-inheritance:
```


## Solving

`qubolite` provides the following methods for solving (exactly or approximately) QUBO instances to obtain their minimizing vector and minimal energy.

```{eval-rst}
.. automodule:: qubolite.solving
   :members:
   :undoc-members:
   :show-inheritance:
```


## Embedding

`qubolite` can create QUBO instances out of other optimization problems by embedding.
Note that for full functionality, you need the `scikit-learn` package.
The following embeddings are available.

```{eval-rst}
.. automodule:: qubolite.embedding
   :members:
   :undoc-members:
   :show-inheritance:
```


## Bounds

Solving QUBO problems is NP-hard. For certain applications it is helpful to have upper and lower bounds of the minimal energy, for which the submodule `bounds` provides some methods.
Functions that return lower bounds are prefixed with `lb_`, and those that return upper bounds with `ub_`.

```{eval-rst}
.. automodule:: qubolite.bounds
   :members:
   :undoc-members:
   :show-inheritance:
```


## Parameter Compression

The submodule `compression` implements the optimum-preserving parameter compression algorithm by [Mücke et al. (2023)](http://arxiv.org/abs/2307.02195).
It aims to reduce the parameters' dynamic range, which positively affects the problem's solvability on quantum annealers.

```{eval-rst}
.. automodule:: qubolite.compression
   :members:
   :undoc-members:
```
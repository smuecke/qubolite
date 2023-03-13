#!/usr/bin/env python
from setuptools import Extension, setup
from numpy      import get_include as numpy_incl

setup(ext_modules=[
    Extension(
        name='qlc',
        sources=['qubolite/qlc.c'],
        include_dirs=[numpy_incl()],
        extra_compile_args=['-O3', '-fopenmp', '-march=native'],
        extra_link_args=['-fopenmp'])])

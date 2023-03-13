#!/usr/bin/env python
from platform import system

from setuptools import Extension, setup
from numpy      import get_include as numpy_incl

SYSTEM = system()
if SYSTEM == 'Windows':
    C_LINK_FLAGS = ['/O3', '/openmp']
    C_COMP_FLAGS = ['/openmp']
else: # GCC flags for Linux
    C_LINK_FLAGS = ['-O3', '-fopenmp', '-march=native']
    C_COMP_FLAGS = ['-fopenmp']

setup(ext_modules=[
    Extension(
        name='qlc',
        sources=['qubolite/qlc.c'],
        include_dirs=[numpy_incl()],
        extra_compile_args=C_COMP_FLAGS,
        extra_link_args=C_LINK_FLAGS)])

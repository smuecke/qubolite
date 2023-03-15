#!/usr/bin/env python
from os.path  import join
from platform import system

from setuptools import Extension, setup
from numpy      import get_include as numpy_incl


SYSTEM = system()
if SYSTEM == 'Windows':
    C_LINK_FLAGS = []
    C_COMP_FLAGS = ['/O2', '/openmp']
else: # GCC flags for Linux
    C_LINK_FLAGS = ['-fopenmp']
    C_COMP_FLAGS = ['-O3', '-fopenmp', '-march=native']

NP_INCL = numpy_incl()

setup(ext_modules=[
    Extension(
        name='_c_utils',
        sources=['qubolite/_c_utils.c'],
        libraries=['npymath','npyrandom'],
        include_dirs=[NP_INCL],
        library_dirs=[
            join(NP_INCL, '..', 'lib'),
            join(NP_INCL, '..', '..', 'random', 'lib')], # no official way to retrieve these directories
        extra_compile_args=C_COMP_FLAGS,
        extra_link_args=C_LINK_FLAGS)])

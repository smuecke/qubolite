#!/usr/bin/env python
from setuptools import Extension, setup
from numpy      import get_include as numpy_incl


with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

qlc = Extension(
    name='qlc',
    sources=['qubolite/qlc.c'],
    include_dirs=[numpy_incl()],
    extra_compile_args=['-O3', '-fopenmp', '-march=native'],
    extra_link_args=['-fopenmp']
)


setup(
        name='qubolite',
        packages=['qubolite'],
        version='0.6.10',
        description='Toolbox for quadratic binary optimization',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Sascha Muecke',
        author_email='sascha.muecke@tu-dortmund.de',
        url='https://github.com/smuecke/qubolite',
        install_requires=['numpy>=1.23.5'],
        ext_modules=[qlc],
        python_requires='>=3.8'
)

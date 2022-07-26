#!/usr/bin/env python
import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
        name='qubolite',
        packages=['qubolite'],
        version='0.6.2',
        description='Toolbox for quadratic binary optimization',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Sascha Muecke',
        author_email='sascha.muecke@tu-dortmund.de',
        url='https://github.com/smuecke/qubolite',
        install_requires=['numba>=0.55.2', 'numpy>=1.22.4', 'scipy>=0.16'],
        python_requires='>=3.8'
)
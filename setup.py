#!/usr/bin/env python
import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    reqs = [ line.strip() for line in f ]

setuptools.setup(
        name='qubolite',
        packages=['qubolite'],
        version='0.6.9',
        description='Toolbox for quadratic binary optimization',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Sascha Muecke',
        author_email='sascha.muecke@tu-dortmund.de',
        url='https://github.com/smuecke/qubolite',
        install_requires=reqs,
        python_requires='>=3.8'
)

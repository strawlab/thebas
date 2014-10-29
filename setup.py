#! /usr/bin/env python2
# coding=utf-8

# Authors: Santi Villalba <sdvillal@gmail.com>
# Licence: BSD 3 clause

from setuptools import setup

import thebas

setup(
    name='thebas',
    license='BSD 3 clause',
    description='Bayesian analysis of tethered data',
    version=thebas.__version__,
    url='https://github.com/strawlab/thebas',
    author='Santi Villalba',
    author_email='villalba@imp.univie.ac.at',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD 3 clause'
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Operating System :: Unix',
    ],
    requires=['h5py',
              'pymc',   # >= 2.3.4
              'pydot',  # for pymc model graphs
              'numpy',
              'matplotlib',
              'pandas',
              'joblib',
              'scipy',
              'dill',
              'argh'],
)

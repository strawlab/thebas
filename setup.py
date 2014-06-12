#! /usr/bin/env python2
# coding=utf-8

# Authors: Santi Villalba <sdvillal@gmail.com>
# Licence: BSD 3 clause

from setuptools import setup


setup(
    name='thebas',
    license='BSD 3 clause',
    description='Bayesian analysis of thethered data',
    version='0.1-dev',
    url='https://github.com/strawlab/thebas',
    author='Santi Villalba',
    author_email='villalba@imp.univie.ac.at',
    classifiers=[  # plagiarism from sklearn
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD 3 clause'
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    require=['h5py',
             'pymc',
             'numpy',
             'matplotlib',
             'pandas',
             'joblib',
             'scipy',
             'dill',
             'argh'],  # FIXME: Make clear which versions
                       # https://caremad.io/blog/setup-vs-requirement/
)

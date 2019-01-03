"""Setup script for uploading package to PyPI servers."""
from setuptools import setup

setup(
    name='vulcanai',
    version='1.0rc1',
    description='A high-level framework built on top of Pytorch'
                ' using added functionality from Scikit-learn to provide '
                'all of the tools needed for visualizing and processing '
                'high-dimensional data, modular neural networks, '
                'and model evaluation',
    author='Robert Fratila, Priyatharsan Rajasekar, Caitrin Armstrong',
    author_email='robertfratila10@gmail.com',
    url='https://github.com/Aifred-Health/Vulcan',
    install_requires=['numpy>=1.12.0',
                      'scipy>=0.17.1',
                      'matplotlib>=1.5.3',
                      'scikit-learn>=0.18',
                      'pandas>=0.23.4',
                      'pydash>=4.7.4',
                      'tqdm>=4.25.0'],
    packages=['vulcanai'],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Education',
                 'Topic :: Software Development :: Build Tools',
                 'Programming Language :: Python :: 3.6',
                 'Operating System :: Unix',
                 'Operating System :: POSIX :: Linux',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
    keywords='deep learning machine learning development framework'
)
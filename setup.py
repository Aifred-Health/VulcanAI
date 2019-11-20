"""Setup script for uploading package to PyPI servers."""
from setuptools import setup, find_packages

tests_require = [
    'pytest',
    'numpydoc'
]

docs_require = [
    'Sphinx', # TODO: maybe numpydoc?
]

with open('requirements.txt') as f:
    install_requires = [l.strip() for l in f]


setup(
    name='vulcanai',
    version='1.0.8',
    description='A high-level framework built on top of Pytorch'
                ' using added functionality from Scikit-learn to provide '
                'all of the tools needed for visualizing and processing '
                'high-dimensional data, modular neural networks, '
                'and model evaluation',
    author='Robert Fratila, Priyatharsan Rajasekar, Caitrin Armstrong, '
            'Joseph Mehltretter, Sneha Desai',
    author_email='robertfratila10@gmail.com',
    url='https://github.com/Aifred-Health/Vulcan',
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        'docs': docs_require,
    },
    packages=find_packages(),
    include_package_data=True,
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

# Vulcan
[![Build Status](https://travis-ci.com/Aifred-Health/Vulcan.svg?branch=master)](https://travis-ci.com/Aifred-Health/Vulcan)
[![Documentation Status](https://readthedocs.org/projects/vulcanai/badge/?version=latest)](https://vulcanai.readthedocs.io/en/latest/?badge=latest)


Vulcan is Aifred Health's framework for rapid deep learning model prototyping and analysis.

Vulcan provides the tools for:
1. Rapid-yet-flexible data preprocessing
2. Rapid creation of modular neural networks. Among the usual we also include capability for:
    * snapshot ensembles
    * multi-modal networks with complex architectures
    * state of the art activations
    * training and saving models across multiple machines
3. Model Evaluation
4. Visualization for data and network interpretability. Among the usual we also include:
    * t-SNE
    * [Saliency maps using guided backpropagation](https://arxiv.org/abs/1412.6806)

Vulcan is built on Pytorch. We think Pytorch is great, so our framework was built with the goal of facilitating but not impeding access to all of Pytorch. Want to do things the easy way? Great, create a network using our simple configuration dict. Need something a little more complicated? Extend our classes or write your own Pytorch module to use within the rest of our framework. 

For a more detailed runthrough on how to use the tools, please look at the [documentation](https://vulcanai.readthedocs.io/en/latest/).

## Installation
[Pytorch](https://pytorch.org) must be installed separately as per your devices requirements (e.g. GPU/CPU). Afterwards, Vulcan can be installed using PyPI:
```
<sudo> pip install vulcanai
```
or you can install from source after cloning the repository:
```
git clone https://github.com/Aifred-Health/Vulcan.git
cd Vulcan
pip install -e vulcanai
```

## Releases
The current stable release is 1.0.1 

## Contributions
We welcome contributions, particularily to TabularDataset, additional processing methods, and to generalized and generalizable network architectures. Please create an issue before embarking on a solution, however, as we may already have something similar in the works!

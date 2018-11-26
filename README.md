# Vulcan
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

For a more detailed runthrough on how to use the tools, please look at the [wiki](https://github.com/Aifred-Health/Vulcan/wiki) (IN PROGRESS)

## Installation
It can be installed using PyPI:
```
<sudo> pip install vulcanai
```

## Contributions
We welcome contributions, particularily to TabularDataset additional processing methods and to generalized and generalizable network architectures. Please create an issue before embarking on a solution, however, as we may already have something similar in the works!

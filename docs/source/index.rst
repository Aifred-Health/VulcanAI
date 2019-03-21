.. Vulcan documentation master file, created by
   sphinx-quickstart on Thu Jan 17 14:46:09 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Vulcan's documentation!
==================================

Vulcan is Aifred Health's framework for rapid deep learning model prototyping and analysis.

Vulcan provides the tools for:

1. Rapid-yet-flexible data preprocessing

2. Rapid creation of modular neural networks. Among the usual we also include capability for:

   - snapshot ensembles
   - multi-modal networks with complex architectures
   - state of the art activations
   - training and saving models across multiple machines

3. Model Evaluation

4. Visualization for data and network interpretability. Among the usual we also include:

   - t-SNE
   - Saliency maps using guided backpropagation


Vulcan is built on Pytorch. We think Pytorch is great, so our framework was built with the goal of facilitating but not impeding access to all of Pytorch. Want to do things the easy way? Great, create a network using our simple configuration dict. Need something a little more complicated? Extend our classes or write your own Pytorch module to use within the rest of our framework.

Contents
==================
.. toctree::
   :maxdepth: 2

   vulcanai.datasets
   vulcanai.models
   vulcanai.plotters


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

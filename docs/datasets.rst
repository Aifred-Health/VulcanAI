vulcanai.datasets
=============================

.. automodule:: vulcanai.datasets

Data Containers
-----------------------------
These are all extensions of torch.utils.data.Dataset.

FashionData
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: vulcanai.datasets.fashion.FashionData
    :members:

MultiDataset
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: vulcanai.datasets.multidataset.MultiDataset
    :members:

TabularDataset
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: vulcanai.datasets.tabulardataset.TabularDataset
    :members:

Data Utilities
-----------------------------

check_split_ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vulcanai.datasets.utils.check_split_ratio

stitch_datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vulcanai.datasets.utils.stitch_datasets

vulcanai.models
=============================
.. automodule:: vulcanai.models

Models
-----------------------------

BaseNetwork
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: vulcanai.models.basenetwork
    :members:

ConvNet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vulcanai.models.cnn
    :members:

DenseNet
~~~~~~~~~~~~~~~~~~~~

.. automodule:: vulcanai.models.dnn
    :members:

SnapshotNet
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vulcanai.models.ensemble
    :members:

Layers/Units
-----------------------------

BaseUnit
~~~~~~~~~~~~~~~~~~~~

.. automodule:: vulcanai.models.layers.BaseUnit
    :members:

FlattenUnit
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vulcanai.models.layers.FlattenUnit
    :members:

ConvUnit
~~~~~~~~~~~~~~~~~~~~

.. automodule:: vulcanai.models.layers.ConvUnit
    :members:

DenseUnit
~~~~~~~~~~~~~~~~~~~~

.. automodule:: vulcanai.models.layers.DenseUnit
    :members:

Metrics
-----------------------------

Metrics
~~~~~~~~~~~~~~~~~~~~

.. automodule:: vulcanai.models.metrics.Metrics
   :members:

Model Utilities
-----------------------------

round_list
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vulcanai.models.utils.round_list

get_one_hot
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vulcanai.models.utils.get_one_hot

pad
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vulcanai.models.utils.pad

selu_weight_init_
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vulcanai.models.utils.selu_weight_init_

selu_bias_init_
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vulcanai.models.utils.selu_bias_init_

set_tensor_device
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vulcanai.models.utils.set_tensor_device

master_device_setter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vulcanai.models.utils.master_device_setter

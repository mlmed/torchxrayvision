
Models
======

Core Classifiers
++++++++++++++++

.. automodule:: torchxrayvision.models

    .. autoclass:: Model
        :members:

        .. automethod:: forward

    .. autoclass:: DenseNet(weights=SPECIFY, op_threshs=None, apply_sigmoid=False)
        :members:
   
    .. autoclass:: ResNet(weights=SPECIFY, op_threshs=None, apply_sigmoid=False)
        :members:

Core Autoencoders
+++++++++++++++++

.. automodule:: torchxrayvision.autoencoders

    .. autoclass:: ResNetAE(weights=SPECIFY)


CheXpert Pathologies
++++++++++++++++++++

.. automodule:: torchxrayvision.baseline_models.chexpert
   :members:

JF Healthcare Pathologies
+++++++++++++++++++++++++

.. automodule:: torchxrayvision.baseline_models.jfhealthcare
   :members:

ChestX-Det Segmentation
+++++++++++++++++++++++

.. automodule:: torchxrayvision.baseline_models.chestx_det
   :members:
   
Emory HITI Race
+++++++++++++++

.. automodule:: torchxrayvision.baseline_models.emory_hiti
   :members:
   
Riken Age Model
+++++++++++++++

.. automodule:: torchxrayvision.baseline_models.riken
   :members:

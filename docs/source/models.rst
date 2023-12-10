
Models
======

Model benchmarks for classifiers are `here <https://github.com/mlmed/torchxrayvision/blob/master/BENCHMARKS.md>`_


Model Interface
+++++++++++++++

.. automodule:: torchxrayvision.models

    .. autoclass:: Model
        :members:

        .. automethod:: forward


XRV Pathology Classifiers
+++++++++++++++++++++++++

    .. autoclass:: DenseNet(weights=SPECIFY, op_threshs=None, apply_sigmoid=False)
        :members:
   
    .. autoclass:: ResNet(weights=SPECIFY, op_threshs=None, apply_sigmoid=False)
        :members:

XRV ResNet Autoencoder
++++++++++++++++++++++

.. automodule:: torchxrayvision.autoencoders

    .. autoclass:: ResNetAE(weights=SPECIFY)


CheXpert Pathology Classifier
+++++++++++++++++++++++++++++

.. automodule:: torchxrayvision.baseline_models.chexpert
   :members:

JF Healthcare Pathology Classifier
++++++++++++++++++++++++++++++++++

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

Xinario View Model
++++++++++++++++++

.. automodule:: torchxrayvision.baseline_models.xinario
   :members:

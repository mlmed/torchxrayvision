
Models
======

Model benchmarks for classifiers are `here <https://github.com/mlmed/torchxrayvision/blob/main/BENCHMARKS.md>`_


Model Interface
+++++++++++++++

.. automodule:: xrv.models

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

.. automodule:: xrv.autoencoders

    .. autoclass:: ResNetAE(weights=SPECIFY)


CheXpert Pathology Classifier
+++++++++++++++++++++++++++++

.. autoclass:: xrv.baseline_models.chexpert.DenseNet
   :members:

JF Healthcare Pathology Classifier
++++++++++++++++++++++++++++++++++

.. autoclass:: xrv.baseline_models.jfhealthcare.DenseNet
   :members:

ChestX-Det Segmentation
+++++++++++++++++++++++

.. autoclass:: xrv.baseline_models.chestx_det.PSPNet
   :members:
   
Emory HITI Race
+++++++++++++++

.. autoclass:: xrv.baseline_models.emory_hiti.RaceModel
   :members:
   
Riken Age Model
+++++++++++++++

.. autoclass:: xrv.baseline_models.riken.AgeModel
   :members:

Xinario View Model
++++++++++++++++++

.. autoclass:: xrv.baseline_models.xinario.ViewModel
   :members:

Mira Sex Model
++++++++++++++++++

.. autoclass:: xrv.baseline_models.mira.SexModel
   :members:

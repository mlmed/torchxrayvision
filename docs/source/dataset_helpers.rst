
Dataset Helpers
===============

Overview
--------

TorchXRayVision provides several utilities for combining, filtering, and
pre-processing datasets. These tools are designed to work seamlessly with
any class that inherits from :class:`xrv.datasets.Dataset <xrv.datasets.Dataset>`.

A typical multi-dataset workflow looks like this::

    import torchxrayvision as xrv

    d1 = xrv.datasets.NIH_Dataset(imgpath="/path/to/nih")
    d2 = xrv.datasets.CheX_Dataset(imgpath="/path/to/chexpert")

    # Align label columns across datasets
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d1)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d2)

    # Merge into a single iterable
    d_all = xrv.datasets.MergeDataset([d1, d2])

Combining Datasets
------------------

.. autoclass:: xrv.datasets.MergeDataset
    :members:
    :special-members: __len__, __getitem__

.. autoclass:: xrv.datasets.FilterDataset
    :members:

.. autoclass:: xrv.datasets.SubsetDataset
    :members:

Label Alignment
---------------

Before merging datasets that were labelled with different pathology sets,
use :func:`xrv.datasets.relabel_dataset <xrv.datasets.relabel_dataset>` to reorder (and
optionally pad with ``NaN``) the label columns to a common list.

.. autofunction:: xrv.datasets.relabel_dataset

Image Transforms
----------------

These transform objects are compatible with
``torchvision.transforms.Compose`` and operate on the ``(1, H, W)``
float32 NumPy arrays returned by dataset ``__getitem__`` calls.

.. autoclass:: xrv.datasets.XRayCenterCrop
    :members:

.. autoclass:: xrv.datasets.XRayResizer
    :members:

Covariate Shift Simulation
--------------------------

.. autoclass:: xrv.datasets.CovariateDataset
    :members:


  
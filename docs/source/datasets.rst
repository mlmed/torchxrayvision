
Datasets
========

Overview
--------

TorchXRayVision provides a unified interface for a wide range of publicly
available chest X-ray datasets. Every dataset class inherits from
:class:`xrv.datasets.Dataset <xrv.datasets.Dataset>` and exposes three key attributes:

- **pathologies** – an ordered list of label names
- **labels** – a 2-D NumPy array (samples × pathologies) with values ``1``, ``0``, or ``NaN``
- **csv** – a Pandas ``DataFrame`` of associated per-image metadata

Loading a dataset requires only the path to the image directory::

    import torchxrayvision as xrv

    d = xrv.datasets.NIH_Dataset(imgpath="/path/to/images")

**Common keyword arguments** accepted by most dataset classes:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Argument
     - Description
   * - ``imgpath``
     - Path to the directory containing the image files *(required)*.
   * - ``csvpath``
     - Path to the metadata CSV file. Defaults to the bundled copy when
       available.
   * - ``views``
     - Restrict images to the specified radiographic views,
       e.g. ``["PA"]`` or ``["PA", "AP"]``.
   * - ``transform``
     - A ``torchvision``-compatible transform applied to each sample at
       load time.
   * - ``data_aug``
     - An additional transform applied as data augmentation (separate seed).
   * - ``unique_patients``
     - When ``True`` (default for most datasets), only one image per patient
       is returned.

To combine multiple datasets or align their label columns, see
:doc:`dataset_helpers`.

Available Datasets
------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Class
     - Dataset
   * - :class:`xrv.datasets.NIH_Dataset <xrv.datasets.NIH_Dataset>`
     - NIH ChestX-ray14 (112 k images, 14 pathologies)
   * - :class:`xrv.datasets.NIH_Google_Dataset <xrv.datasets.NIH_Google_Dataset>`
     - NIH ChestX-ray14 with Google radiologist re-labels
   * - :class:`xrv.datasets.CheX_Dataset <xrv.datasets.CheX_Dataset>`
     - CheXpert (Stanford, 224 k images, 14 pathologies)
   * - :class:`xrv.datasets.MIMIC_Dataset <xrv.datasets.MIMIC_Dataset>`
     - MIMIC-CXR (MIT/PhysioNet, 227 k images)
   * - :class:`xrv.datasets.PC_Dataset <xrv.datasets.PC_Dataset>`
     - PadChest (Spain, 94 k images)
   * - :class:`xrv.datasets.RSNA_Pneumonia_Dataset <xrv.datasets.RSNA_Pneumonia_Dataset>`
     - RSNA Pneumonia Detection Challenge
   * - :class:`xrv.datasets.Openi_Dataset <xrv.datasets.Openi_Dataset>`
     - OpenI / Indiana University chest X-ray collection
   * - :class:`xrv.datasets.COVID19_Dataset <xrv.datasets.COVID19_Dataset>`
     - COVID-19 image data collection
   * - :class:`xrv.datasets.NLMTB_Dataset <xrv.datasets.NLMTB_Dataset>`
     - NLM / Montgomery & Shenzhen tuberculosis datasets
   * - :class:`xrv.datasets.TBX11K_Dataset <xrv.datasets.TBX11K_Dataset>`
     - TBX11K tuberculosis dataset
   * - :class:`xrv.datasets.SIIM_Pneumothorax_Dataset <xrv.datasets.SIIM_Pneumothorax_Dataset>`
     - SIIM-ACR Pneumothorax Segmentation
   * - :class:`xrv.datasets.VinBrain_Dataset <xrv.datasets.VinBrain_Dataset>`
     - VinBigData Chest X-ray Abnormalities Detection
   * - :class:`xrv.datasets.StonyBrookCOVID_Dataset <xrv.datasets.StonyBrookCOVID_Dataset>`
     - Stony Brook University COVID-19 positive cases
   * - :class:`xrv.datasets.ObjectCXR_Dataset <xrv.datasets.ObjectCXR_Dataset>`
     - Object-CXR foreign-object detection dataset

Base Class
----------

All dataset classes share the interface defined below.

.. autoclass:: xrv.datasets.Dataset
    :members:
    :exclude-members: limit_to_selected_views
    :special-members: __repr__

Dataset Classes
---------------

.. autoclass:: xrv.datasets.NIH_Dataset
    :members: string

.. autoclass:: xrv.datasets.NIH_Google_Dataset
    :members: string

.. autoclass:: xrv.datasets.CheX_Dataset
    :members: string

.. autoclass:: xrv.datasets.MIMIC_Dataset
    :members: string

.. autoclass:: xrv.datasets.PC_Dataset
    :members: string

.. autoclass:: xrv.datasets.RSNA_Pneumonia_Dataset
    :members: string

.. autoclass:: xrv.datasets.Openi_Dataset
    :members: string

.. autoclass:: xrv.datasets.COVID19_Dataset
    :members: string

.. autoclass:: xrv.datasets.NLMTB_Dataset
    :members: string

.. autoclass:: xrv.datasets.TBX11K_Dataset
    :members: string

.. autoclass:: xrv.datasets.SIIM_Pneumothorax_Dataset
    :members: string

.. autoclass:: xrv.datasets.VinBrain_Dataset
    :members: string

.. autoclass:: xrv.datasets.StonyBrookCOVID_Dataset
    :members: string

.. autoclass:: xrv.datasets.ObjectCXR_Dataset
    :members: string

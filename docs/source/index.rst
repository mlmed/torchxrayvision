Introduction
=========================================

A library for chest X-ray datasets and models. Including pre-trained models.

TorchXRayVision is an open source software library for working with chest X-ray datasets and deep learning models. It provides a common interface and common pre-processing chain for a wide set of publicly available chest X-ray datasets. In addition, a number of classification and representation learning models with different architectures, trained on different data combinations, are available through the library to serve as baselines or feature extractors.

- In the case of researchers addressing clinical questions it is a waste of time for them to train models from scratch. To address this, TorchXRayVision provides pre-trained models which are trained on large cohorts of data and enables 1) rapid analysis of large datasets 2) feature reuse for few-shot learning.
- In the case of researchers developing algorithms it is important to robustly evaluate models using multiple external datasets. Metadata associated with each dataset can vary greatly which makes it difficult to apply methods to multiple datasets. TorchXRayVision provides access to many datasets in a uniform way so that they can be swapped out with a single line of code. These datasets can also be merged and filtered to construct specific distributional shifts for studying generalization.

Twitter: `@torchxrayvision <https://twitter.com/torchxrayvision>`_

.. _installation:

Installation
++++++++++++

You can install this package via the command line by entering::

    pip install torchxrayvision


Getting started
+++++++++++++++


.. code-block:: python

    import torchxrayvision as xrv
    import skimage, torch, torchvision

    # Prepare the image:
    img = skimage.io.imread("16747_3_1.jpg")
    img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
    img = img.mean(2)[None, ...] # Make single color channel

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])

    img = transform(img)
    img = torch.from_numpy(img)

    # Load model and process image
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    outputs = model(img[None,...]) # or model.features(img[None,...]) 

    # Print results
    dict(zip(model.pathologies, outputs[0].detach().numpy()))

    {'Atelectasis': 0.32797316,
     'Consolidation': 0.42933336,
     'Infiltration': 0.5316924,
     'Pneumothorax': 0.28849724,
     'Edema': 0.024142697,
     'Emphysema': 0.5011832,
     'Fibrosis': 0.51887786,
     'Effusion': 0.27805611,
     'Pneumonia': 0.18569896,
     'Pleural_Thickening': 0.24489835,
     'Cardiomegaly': 0.3645515,
     'Nodule': 0.68982,
     'Mass': 0.6392845,
     'Hernia': 0.00993878,
     'Lung Lesion': 0.011150705,
     'Fracture': 0.51916164,
     'Lung Opacity': 0.59073937,
     'Enlarged Cardiomediastinum': 0.27218717}






Contents
++++++++

.. toctree::

    self
    models
    dataset_helpers
    datasets

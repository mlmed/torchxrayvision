FAQ / Troubleshooting
======================

Common issues and their resolutions when using TorchXRayVision.

Missing pydicom
---------------
If you call ``torchxrayvision.utils.read_xray_dcm`` and receive an ``ImportError`` or a message about pydicom not being installed, install it:

.. code-block:: bash

    pip install pydicom

Cache / download issues
-----------------------
Model weights and some metadata are cached under the directory returned by ``torchxrayvision.utils.get_cache_dir()`` (default: ``~/.torchxrayvision``). If downloads fail or you see permission errors:

- Ensure the directory exists and is writable.
- Clear a partial download (remove the specific file) and retry.
- If behind a proxy, set standard environment variables (``HTTP_PROXY``, ``HTTPS_PROXY``).

Unexpected ~0.5 predictions
---------------------------
Some model heads may be untrained for certain weight configurations. Only interpret labels present and trained in ``model.pathologies``. Consistently ~0.5 values for a label may indicate that label wasn't trained.

To inspect the raw logits for each pathology:

.. code-block:: python

    import torchxrayvision as xrv
    import torch

    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()

    img = xrv.utils.load_image("path/to/xray.jpg")
    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(224)(img)
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # [1, 1, 224, 224]

    with torch.no_grad():
        output = model(img_tensor)[0].cpu().numpy()  # shape: (num_pathologies,)

    for pathology, score in zip(model.pathologies, output):
        print(f"{pathology}: {score:.4f}")
    # Atelectasis: 0.0821
    # Consolidation: 0.0314
    # Infiltration: 0.1203
    # Pneumothorax: 0.0042
    # Edema: 0.0587
    # Emphysema: 0.5000      <- consistently ~0.5 = likely untrained head
    # Fibrosis: 0.5000       <- consistently ~0.5 = likely untrained head
    # Effusion: 0.1745
    # Pneumonia: 0.0623
    # Pleural_Thickening: 0.0398
    # Cardiomegaly: 0.2156
    # Nodule: 0.0871
    # Mass: 0.0412
    # Hernia: 0.0103

Labels consistently near 0.5 across many images were likely not included in the training set for that weight configuration.


DICOM MONOCHROME1 inversion
---------------------------
Some DICOMs store pixel data where white/black are inverted (``PhotometricInterpretation=MONOCHROME1``). ``read_xray_dcm`` inverts these by default. To disable, pass ``fix_monochrome=False``.

Incorrect image intensity range
-------------------------------
All TorchXRayVision models expect pixel values in the range ``[-1024, 1024]``. Passing raw uint8 images (range ``[0, 255]``) or float images in ``[0, 1]`` will produce silently wrong predictions because the model was trained on a different scale.

The easiest way to avoid this is to use ``xrv.utils.load_image``, which handles both PNG/JPG and DICOM and always returns a correctly normalized ``[1, H, W]`` float32 array:

.. code-block:: python

    import torchxrayvision as xrv

    img = xrv.utils.load_image("path/to/xray.jpg")
    print(img.min(), img.max())  # e.g. -887.3  1024.0

If you are loading images yourself (e.g. with ``skimage``, ``PIL``, or ``cv2``), normalize manually before passing to the model:

.. code-block:: python

    import skimage.io
    import torchxrayvision as xrv

    img = skimage.io.imread("path/to/xray.jpg")   # uint8, shape (H, W) or (H, W, 3)
    img = xrv.utils.normalize(img, maxval=255, reshape=True)
    # img is now float32 in [-1024, 1024], shape (1, H, W)

For 16-bit images (e.g. raw DICOM pixel arrays with 12-bit depth stored as uint16), pass the appropriate ``maxval``:

.. code-block:: python

    img = xrv.utils.normalize(raw_pixels, maxval=4095)   # 12-bit stored in uint16

The ``warn_normalization`` utility will print a warning on the first forward pass if it detects the input is likely not in the expected range.

Slow first import due to downloads
----------------------------------
The first time you use a given pretrained weight (e.g. ``densenet121-res224-all``) it is downloaded. Subsequent runs use the cache. You can pre-fetch by calling ``torchxrayvision.models.get_model(weights=...)`` (or instantiating the model directly) in a setup script.

Reproducibility / non-determinism
---------------------------------
Set seeds before creating datasets and models:

.. code-block:: python

    import torch, numpy as np, random
    seed = 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
Some model heads may be untrained for certain weight configurations. Only interpret labels present and trained in ``model.pathologies``. Inspect logits by zipping ``model.pathologies`` with the output tensor. Consistently ~0.5 values for a label may indicate that label wasn't trained.

Out-of-memory with large datasets
---------------------------------
- Use a DataLoader with ``batch_size`` tuned to GPU RAM.
- Disable loading pathology masks (``pathology_masks=False``) unless required.
- Avoid constructing large merged datasets before filtering; filter each dataset individually first.

DICOM MONOCHROME1 inversion
---------------------------
Some DICOMs store pixel data where white/black are inverted (``PhotometricInterpretation=MONOCHROME1``). ``read_xray_dcm`` inverts these by default. To disable, pass ``fix_monochrome=False``.

Incorrect image intensity range
-------------------------------
Use ``torchxrayvision.datasets.normalize(img, 255)`` for 8-bit images or rely on ``torchxrayvision.utils.load_image`` which applies normalization and returns a tensor scaled to approximately ``[-1024, 1024]``.

Slow first import due to downloads
----------------------------------
The first time you use a given pretrained weight (e.g. ``densenet121-res224-all``) it is downloaded. Subsequent runs use the cache. You can pre-fetch by calling ``torchxrayvision.models.get_densenet(weights=...)`` (or instantiating the model) in a setup script.

Reproducibility / non-determinism
---------------------------------
Set seeds before creating datasets and models:

.. code-block:: python

    import torch, numpy as np, random
    seed = 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

CUDA / cuDNN mismatch warnings
------------------------------
Ensure your installed PyTorch binary matches the CUDA version on your system or use the CPU-only wheel. This is outside the scope of the library but commonly encountered.

Need to change cache location
-----------------------------
Set the environment variable ``TORCHXRAYVISION_CACHE`` to a writable path before importing the library.

Where to ask questions
----------------------
- GitHub Issues for bugs.
- Discussions / Q&A (if enabled on repository).
- Twitter: @torchxrayvision for announcements.

Utilities Overview
==================

High-level overview of helper functions and utilities available in TorchXRayVision.

Image I/O
---------
- ``torchxrayvision.utils.load_image(path)``: Load an image (PNG/JPG) and normalize to the expected value range. Returns a numpy array shaped ``[1, H, W]`` (single channel) suitable for model preprocessing.
- ``torchxrayvision.utils.read_xray_dcm(path, fix_monochrome=True)``: Read a DICOM chest X-ray; optionally inverts MONOCHROME1 images.

Inference Helpers
-----------------
- ``torchxrayvision.utils.infer(model, dataset, device=None, num_workers=0, progress=True)``: Convenience loop to run a model over a dataset and collect outputs.

Download / Caching
------------------
- ``torchxrayvision.utils.download(url, filename=None)``: Download and cache a file inside the library cache directory.
- ``torchxrayvision.utils.get_cache_dir()``: Returns the path being used for caching (respects ``TORCHXRAYVISION_CACHE`` env variable).

Resolution & Preprocessing
--------------------------
Transformation classes under ``torchxrayvision.datasets``:
- ``XRayCenterCrop()``: Center crop maintaining aspect.
- ``XRayResizer(size)``: Resize to a square (e.g., 224) using appropriate interpolation.
- ``XRayResizer(size, engine='cv2')`` may leverage cv2 if installed.

Dataset Composition
-------------------
- ``Merge_Dataset(list_of_datasets)``: Concatenate multiple dataset objects (must share a unified pathology set achieved via ``relabel_dataset``).
- ``SubsetDataset(dataset, idxs=None, labels=None)``: Filter a dataset by indices or label criteria.
- ``relabel_dataset(target_pathologies, dataset)``: Align the dataset's internal label order to ``target_pathologies`` adding NaNs for missing labels.

Mask & Segmentation Support
---------------------------
Some datasets optionally load pathology masks (e.g., pneumothorax) when ``pathology_masks=True``. Pixel-level anatomical segmentation models (e.g., PSPNet variants) are available under ``torchxrayvision.baseline_models``.

Pathology Handling
------------------
- ``torchxrayvision.datasets.default_pathologies``: Canonical ordering of common pathologies used for aligning model outputs and dataset labels.

Miscellaneous
-------------
- ``torchxrayvision.utils.seed_everything(seed)`` (if present) or manual seed setting as shown in FAQ for reproducibility.
- ``torchxrayvision.utils.list_pretrained_models()`` (if present; otherwise refer to the models documentation) to enumerate available weight strings.

Usage Example
-------------
.. code-block:: python

    import torchxrayvision as xrv
    import torch

    ds = xrv.datasets.NIH_Dataset(imgpath="/data/NIH")
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, ds)

    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    sample = torch.from_numpy(ds[0]['img']).unsqueeze(0)
    with torch.no_grad():
        out = model(sample)
    preds = dict(zip(model.pathologies, out[0].cpu().numpy()))
    print(preds)

Utilities Overview
==================

High-level overview of helper functions and utilities available in TorchXRayVision.

Image I/O
---------
- ``torchxrayvision.utils.load_image(path)``: Load an image (PNG/JPG or DICOM) and normalize to the expected value range. Returns a numpy array shaped ``[1, H, W]`` (single channel) suitable for model preprocessing.
- ``torchxrayvision.utils.read_xray_dcm(path, voi_lut=False, fix_monochrome=True)``: Read a DICOM chest X-ray file. Returns a 2D array ``[H, W]`` normalized between -1024 and 1024. Set ``voi_lut=True`` for human-viewable output; set ``fix_monochrome=False`` to skip MONOCHROME1 inversion.
- ``torchxrayvision.utils.normalize(img, maxval, reshape=False)``: Scale a raw pixel array to the ``[-1024, 1024]`` range expected by all models. ``maxval`` is the maximum possible pixel value (e.g. 255 for 8-bit images). Set ``reshape=True`` to also convert a 2D ``[H, W]`` array to ``[1, H, W]``.

.. code-block:: python

    import torchxrayvision as xrv
    import numpy as np

    # Load a PNG/JPG image - returns [1, H, W]
    img = xrv.utils.load_image("path/to/xray.jpg")
    print(img.shape)  # Output: (1, H, W)
    print(img.min(), img.max())  # Normalized range

    # Load a DICOM file - returns [H, W]
    dcm_img = xrv.utils.read_xray_dcm("path/to/xray.dcm")
    print(dcm_img.shape)  # (H, W)
    
    # load_image handles both formats and always returns [1, H, W]
    dcm_from_load = xrv.utils.load_image("path/to/xray.dcm")
    print(dcm_from_load.shape)  # (1, H, W)

Inference Helpers
-----------------
- ``torchxrayvision.utils.infer(model, dataset, threads=4, device='cpu')``: Convenience loop to run a model over a dataset and collect outputs. Returns predictions as a numpy array.
- ``torchxrayvision.utils.fix_resolution(x, resolution, model)``: Resize a tensor ``[B, 1, H, W]`` to the target ``resolution`` using bilinear interpolation if the input size doesn't match. Raises an error if H ≠ W (use ``XRayCenterCrop`` first). Prints a one-time warning per model if a resize is performed.
- ``torchxrayvision.utils.warn_normalization(x)``: Check on the first forward pass that the input tensor appears to be in the ``[-1024, 1024]`` range and print a warning if not. Called internally by models but can also be used standalone.

.. code-block:: python

    import torchxrayvision as xrv
    import torch

    # Load model and dataset
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    dataset = xrv.datasets.NIH_Dataset(imgpath="/data/NIH")

    # Run inference over entire dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outputs = xrv.utils.infer(
        model, 
        dataset, 
        threads=4,
        device=device
    )
    print(outputs.shape)  # (num_samples, num_pathologies)

Download / Caching
------------------
- ``torchxrayvision.utils.download(url, filename)``: Download a file from a URL to a specified filename.
- ``torchxrayvision.utils.get_cache_dir()``: Returns the default cache directory used by the library (``~/.torchxrayvision/models_data/``).

.. code-block:: python

    import torchxrayvision as xrv
    import os

    # Get the default cache directory
    cache_dir = xrv.utils.get_cache_dir()
    print(f"Cache directory: {cache_dir}")

    # Download a file to a specific location
    url = "https://example.com/model_weights.pth"
    filepath = os.path.join(cache_dir, "model_weights.pth")
    xrv.utils.download(url, filepath)
    print(f"Downloaded to: {filepath}")

Resolution & Preprocessing
--------------------------
Transformation classes under ``torchxrayvision.datasets``:
- ``XRayCenterCrop()``: Center crop to a square using the shorter dimension (crops both sides to ``min(H, W)``).
- ``XRayResizer(size, engine='skimage')``: Resize to a square (e.g., 224) using skimage or OpenCV. Input/output shape: ``[1, H, W]``.

.. code-block:: python

    import torchxrayvision as xrv
    import torch

    # Load an image and preprocess
    img = xrv.utils.load_image("path/to/xray.jpg")
    print(f"Original shape: {img.shape}")  # (1, H, W)

    # Apply transformations
    center_crop = xrv.datasets.XRayCenterCrop()
    img_cropped = center_crop(img)
    print(f"After center crop: {img_cropped.shape}")

    resizer = xrv.datasets.XRayResizer(224)
    img_resized = resizer(img_cropped)
    print(f"After resize: {img_resized.shape}")  # (1, 224, 224)

    # Convert to tensor for model input
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0)  # Add batch dim
    print(f"Tensor shape: {img_tensor.shape}")  # (1, 1, 224, 224)

Dataset Composition
-------------------
- ``Merge_Dataset(list_of_datasets)``: Concatenate multiple dataset objects (must share the same pathologies list).
- ``SubsetDataset(dataset, idxs)``: Filter a dataset by selecting specific indices.
- ``relabel_dataset(pathologies, dataset, silent=False)``: Align the dataset's internal label order to ``pathologies``, inserting NaNs for missing labels and dropping extras. Pass ``silent=True`` to suppress the printed list of dropped pathologies.

.. code-block:: python

    import torchxrayvision as xrv
    import numpy as np

    # Load datasets
    nih_ds = xrv.datasets.NIH_Dataset(imgpath="/data/NIH")
    chexpert_ds = xrv.datasets.CheXpert_Dataset(imgpath="/data/CheXpert")

    # Align both datasets to use the same pathologies
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, nih_ds)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, chexpert_ds)

    # Merge datasets
    combined_ds = xrv.datasets.Merge_Dataset([nih_ds, chexpert_ds])
    print(f"Combined dataset size: {len(combined_ds)}")

    # Create a subset using indices
    subset_ds = xrv.datasets.SubsetDataset(combined_ds, idxs=np.arange(0, 100))
    print(f"Subset size: {len(subset_ds)}")

    # Create a subset with specific labels (e.g., positive for Pneumonia)
    pneumonia_indices = np.where(
        combined_ds.labels[:, combined_ds.pathologies.index('Pneumonia')] == 1
    )[0]
    pneumonia_subset = xrv.datasets.SubsetDataset(combined_ds, idxs=pneumonia_indices)

Mask & Segmentation Support
---------------------------
The ``NIH_Dataset`` and ``CheXpert_Dataset`` (and several others) accept a ``pathology_masks=True`` constructor argument. When enabled, each sample dict contains a ``"pathology_masks"`` key mapping pathology names to binary pixel arrays of the same spatial size as ``"img"``.

.. code-block:: python

    import torchxrayvision as xrv

    ds = xrv.datasets.NIH_Dataset(
        imgpath="/data/NIH",
        pathology_masks=True,
    )
    sample = ds[0]
    print(sample.keys())          # dict_keys(['img', 'lab', 'pathology_masks', ...])
    print(sample["pathology_masks"].keys())  # e.g. {'Atelectasis': array(...)}

Pixel-level anatomical segmentation models (PSPNet variants for structures such as lungs and heart) are available under ``torchxrayvision.baseline_models``.

Pathology Handling
------------------
- ``torchxrayvision.datasets.default_pathologies``: Canonical ordering of 18 common pathologies used for aligning model outputs and dataset labels::

    ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
     'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
     'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
     'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum']

.. code-block:: python

    import torchxrayvision as xrv

    # View default pathologies
    print("Default pathologies:")
    print(xrv.datasets.default_pathologies)



End-to-End Pipeline
-------------------
Typical flow from a raw file to per-pathology predictions:

.. code-block:: python

    import torchxrayvision as xrv
    import torch

    # 1. Load and normalize a raw image (PNG/JPG or DICOM -> [1, H, W])
    img = xrv.utils.load_image("path/to/xray.jpg")

    # 2. Crop to square then resize to the model's native resolution
    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(224)(img)

    # 3. Add batch dimension -> [1, 1, 224, 224]
    img_tensor = torch.from_numpy(img).unsqueeze(0)

    # 4. Load a pretrained model and run inference
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    with torch.no_grad():
        out = model(img_tensor)  # shape: (1, num_pathologies)

    # 5. Map scores to pathology names
    preds = dict(zip(model.pathologies, out[0].cpu().numpy()))
    for pathology, score in preds.items():
        print(f"{pathology}: {score:.3f}")

# Contributing to TorchXRayVision

We welcome pull requests! Please open an issue to discuss major changes before submitting.

We are looking for contributions in:
- Datasets: Tuberculosis datasets, segmentation masks
- Baseline Models: Pathology predictions or other interesting tasks
- Utility Functions: Evaluation scripts and metrics

---

## Guidelines

### 1. Adding a New Dataset
Follow conventions in `torchxrayvision/datasets.py`:
- Inherit from `Dataset`.
- Parse metadata from CSV/JSON.
- `__getitem__` must return a dict with `img` and `lab`.
- Normalize images with `xrv.utils.normalize(img, maxval=...)`. Ensure `maxval` matches the format (e.g., 255 for 8-bit) to map values to `[-1024, 1024]` with shape `(1, H, W)`.

### 2. Adding a Baseline Model
When adding to `torchxrayvision/baseline_models`:
- Inherit from `nn.Module`.
- Define a `targets` list for model outputs.
- Download weights automatically via `xrv.utils.download` unless restricted.
- The `forward()` method must accept inputs in `[-1024, 1024]` and internally scale them for your model.
- Use `xrv.utils.fix_resolution` and `xrv.utils.warn_normalization`.
- Provide a docstring with examples and citation.

### 3. Development
Install requirements:
```bash
pip install -r requirements-dev.txt
```
Run tests:
```bash
pytest tests/
```
Check style with `pep8.sh`.

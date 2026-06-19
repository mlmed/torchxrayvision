import pytest
import os
import numpy as np
import torchxrayvision as xrv

IMGPATH = os.environ.get("TBX11K_PATH", "")

@pytest.mark.skipif(not os.path.exists(IMGPATH), reason="TBX11K dataset not found")
def test_tbx11k_basic():
    tbx11k = xrv.datasets.TBX11K_Dataset(imgpath=IMGPATH)
    assert len(tbx11k) == 6600

@pytest.mark.skipif(not os.path.exists(IMGPATH), reason="TBX11K dataset not found")
def test_tbx11k_get():
    tbx11k = xrv.datasets.TBX11K_Dataset(imgpath=IMGPATH)
    sample = tbx11k[0]
    assert sample["img"] is not None
    assert sample["bbox"] is not None
    assert len(sample["lab"]) == 3  

@pytest.mark.skipif(not os.path.exists(IMGPATH), reason="TBX11K dataset not found")
def test_tbx11k_labels():
    tbx11k = xrv.datasets.TBX11K_Dataset(imgpath=IMGPATH)
    assert tbx11k.labels.shape == (6600,3)
    assert np.all((tbx11k.labels == 0.0) | (tbx11k.labels == 1.0))

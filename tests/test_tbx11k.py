import pytest
import os
import numpy as np
import torchxrayvision as xrv
import torchvision


IMGPATH = os.path.join(os.path.dirname(__file__), "tbx11k_test_data")

def test_tbx11k_basic():
    tbx11k = xrv.datasets.TBX11K_Dataset(imgpath=IMGPATH)
    assert len(tbx11k) == 3

def test_tbx11k_get():
    tbx11k = xrv.datasets.TBX11K_Dataset(imgpath=IMGPATH)
    sample = tbx11k[0]
    assert sample["img"] is not None
    assert sample["bbox"] is not None
    assert len(sample["lab"]) == 4

def test_tbx11k_labels():
    tbx11k = xrv.datasets.TBX11K_Dataset(imgpath=IMGPATH)
    assert tbx11k.labels.shape == (3, 4)

    expected = np.array([
        [1., 0., 0., 1.],  # tb0005 - Active TB
        [0., 1., 0., 1.],  # tb0007 - Obsolete, two bounding boxes
        [0., 0., 0., 0.],  # h0001  - healthy
    ])
    assert np.array_equal(tbx11k.labels, expected)

def test_tbx11k_aug():
    tbx11k = xrv.datasets.TBX11K_Dataset(imgpath=IMGPATH)
    sample = tbx11k[0]
    assert sample["img"].shape == (1, 512, 512)
    aug = torchvision.transforms.Compose([xrv.datasets.XRayResizer(224)])
    tbx11k_aug = xrv.datasets.TBX11K_Dataset(imgpath=IMGPATH, data_aug=aug)
    sample2 = tbx11k_aug[0]
    assert sample2["img"].shape == (1, 224, 224)

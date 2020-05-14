import pytest
import torchxrayvision as xrv



def test_dataloader_basic():
    xrv.datasets.NIH_Dataset(imgpath=".", views=["PA"])


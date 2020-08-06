import pytest
import torchxrayvision as xrv
 

def test_model_basic():
    model = xrv.models.DenseNet()

def test_dataloader_merging():
    model = xrv.models.DenseNet(weights="all")
    
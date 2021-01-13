import pytest
import sys, os
sys.path.insert(0,"../torchxrayvision/")

import torch
import torchxrayvision as xrv
 

def test_model_basic():
    model = xrv.models.DenseNet()

def test_model_pretrained():
    model = xrv.models.DenseNet(weights="all")
    model = xrv.models.DenseNet(weights="mimic_ch")
    
def test_model_function():
    
    model = xrv.models.DenseNet(weights="mimic_ch")
    
    img = torch.ones(1,1,224,224)
    img.requires_grad = True
    pred = model(img)[:,model.pathologies.index("Cardiomegaly")]
    dzdxp = torch.autograd.grad((pred), img)[0]
    
    assert torch.isnan(dzdxp.flatten()).sum().cpu().numpy() == 0 
    
    
    
def test_autoencoder_pretrained():
    ae = xrv.autoencoders.ResNetAE(weights="101-elastic")
    
def test_autoencoder_function():
    
    ae = xrv.autoencoders.ResNetAE(weights="101-elastic")
    
    img = torch.ones(1,1,224,224)
    img.requires_grad = True
    pred = ae(img)["out"].sum()
    dzdxp = torch.autograd.grad((pred), img)[0]
    
    assert torch.isnan(dzdxp.flatten()).sum().cpu().numpy() == 0 
    
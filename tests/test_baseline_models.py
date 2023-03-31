import sys, os
import pytest
import torch
import torchxrayvision as xrv
sys.path.insert(0, "../torchxrayvision/")

    
def test_baselinemodels_load():
    model = xrv.baseline_models.jfhealthcare.DenseNet()
    model = xrv.baseline_models.emory_hiti.RaceModel()
    
    
def test_baselinemodel_jfhealthcare_function():
    
    model = xrv.baseline_models.jfhealthcare.DenseNet()
    
    img = torch.ones(1,1,224,224)
    img.requires_grad = True
    pred = model(img)[:,model.pathologies.index("Cardiomegaly")]
    dzdxp = torch.autograd.grad((pred), img)[0]
    
    assert torch.isnan(dzdxp.flatten()).sum().cpu().numpy() == 0 
    
    
def test_baselinemodel_emory_hiti_function():
    
    model = xrv.baseline_models.emory_hiti.RaceModel()
    
    img = torch.ones(1,1,224,224)
    img.requires_grad = True
    pred = model(img)[:,model.targets.index("White")]
    dzdxp = torch.autograd.grad((pred), img)[0]
    
    assert torch.isnan(dzdxp.flatten()).sum().cpu().numpy() == 0 
    
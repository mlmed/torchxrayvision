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
    
    img = torch.ones(1, 1, 224, 224)
    img.requires_grad = True
    pred = model(img)[:,model.pathologies.index("Cardiomegaly")]
    assert pred.shape == torch.Size([1]), 'check output is correct shape'
        
    dzdxp = torch.autograd.grad((pred), img)[0]
    assert dzdxp.shape == torch.Size([1, 1, 224, 224]), 'check grads are the correct size'
    
    assert torch.isnan(dzdxp.flatten()).sum().cpu().numpy() == 0 
    
    
def test_baselinemodel_emory_hiti_function():
    
    model = xrv.baseline_models.emory_hiti.RaceModel()
    
    img = torch.ones(1, 1, 224, 224)
    img.requires_grad = True
    pred = model(img)[:,model.targets.index("White")]
    assert pred.shape == torch.Size([1]), 'check output is correct shape'
    dzdxp = torch.autograd.grad((pred), img)[0]
    assert dzdxp.shape == torch.Size([1, 1, 224, 224]), 'check grads are the correct size'
    
    assert torch.isnan(dzdxp.flatten()).sum().cpu().numpy() == 0 
    
    
def test_baselinemodel_riken_age_function():
    
    model = xrv.baseline_models.riken.AgeModel()

    img = torch.ones(2, 1, 224, 224)
    img.requires_grad = True
    pred = model(img)
    assert pred.shape == torch.Size([2, 1]), 'check output is correct shape'
    assert pred[0] > 0 and pred[0] < 100, 'check output is in the correct range'

    dzdxp = torch.autograd.grad(pred.sum(), img)[0]

    assert dzdxp.shape == torch.Size([2, 1, 224, 224]), 'check grads are the correct size'

    assert torch.isnan(dzdxp.flatten()).sum().cpu().numpy() == 0 , 'check no grads are nans'
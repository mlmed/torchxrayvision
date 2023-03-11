import sys, os
import pytest
import torch
import torchxrayvision as xrv
sys.path.insert(0, "../torchxrayvision/")


def test_model_basic():
    model = xrv.models.DenseNet()

    
def test_model_pretrained():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
    model = xrv.models.ResNet(weights="resnet50-res512-all")


def test_model_function():
    models = [xrv.models.DenseNet(weights="all"),
             xrv.models.DenseNet(weights="mimic_ch"),
             xrv.models.ResNet(weights="resnet50-res512-all")]
    
    for model in models:
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
    
    
def test_autoencoder_resolution_check():
    ae = xrv.autoencoders.ResNetAE(weights="101-elastic")
    
    with pytest.raises(ValueError):
        ae(torch.zeros(1,1,10,10))
    
def test_num_classes():
    
    model_classes = [xrv.models.DenseNet]
    
    for model_class in model_classes:
        for i in [1,4,11,20]:
            model = model_class(num_classes = i)
            assert model.classifier.weight.shape[0] == i
            
            
    with pytest.raises(ValueError):
        # should raise error:
        xrv.models.DenseNet(weights="all", num_classes=4)
        
def test_normalization_check():
    
    models = [xrv.models.DenseNet(weights="densenet121-res224-all"),
             xrv.models.ResNet(weights="resnet50-res512-all")]
    
    incorrect_ranges = [
        [0, 1],
        [-1, 1],
        [0, 1024],
        [-1026, 1024],
    ]
    correct_ranges = [
        [-1024, 1024],
        [-724, 412],
    ]
    
    for model in models:
        for ra in incorrect_ranges:
            test_x = torch.zeros([1,1,224,224])
            test_x.uniform_(ra[0], ra[1])
            
            # Sometimes uniform_ doesn't hit the edge cases we want
            # so here the first 2 pixels are set to the limits
            test_x[0][0][0] = ra[0]  
            test_x[0][0][1] = ra[1]
            xrv.models.warning_log = {}
            model(test_x)
            assert xrv.models.warning_log['norm_correct'] == False, ra
            
        for ra in correct_ranges:
            test_x = torch.zeros([1,1,224,224])
            test_x.uniform_(ra[0], ra[1])
            xrv.models.warning_log = {}
            model(test_x)
            assert xrv.models.warning_log['norm_correct'] == True, ra


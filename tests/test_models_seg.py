import sys
import torch
import torchxrayvision as xrv
sys.path.insert(0, "../torchxrayvision/")

    
def test_baselinemodel_pretrained():
    seg_model = xrv.baseline_models.chestx_det.PSPNet()
    
    image = torch.ones(1, 1, 224, 224)
    with torch.no_grad():
        output = seg_model(image)
    
    assert list(output.shape) == [1, 14, 512, 512]
    assert len(seg_model.targets) == 14

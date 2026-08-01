import torch
import torchxrayvision as xrv


def test_baselinemodel_chestx_anatomy_function():
    seg_model = xrv.baseline_models.chestx_anatomy.UNetResNet50()

    image = torch.ones(1, 1, 224, 224)
    with torch.no_grad():
        output = seg_model(image)

    assert list(output.shape) == [1, 159, 512, 512]
    assert len(seg_model.targets) == 159


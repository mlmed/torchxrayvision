import os
from typing import List

import torch
import torch.nn as nn
import torchvision
import pathlib
import torchxrayvision as xrv
from ... import utils


class ViewModel(nn.Module):
    """


    The native resolution of the model is 320x320. Images are scaled
    automatically.

    `Demo notebook <https://github.com/mlmed/torchxrayvision/blob/master/scripts/view_classifier.ipynb>`__

    .. code-block:: python

        model = xrv.baseline_models.xinario.ViewModel()

        image = xrv.utils.load_image('00027426_000.png')
        image = torch.from_numpy(image)[None,...]

        pred = model(image)
        # tensor([[17.3186, 26.7156]]), grad_fn=<AddmmBackward0>)

        model.targets[pred.argmax()]
        # Lateral


    Source: https://github.com/xinario/chestViewSplit

    """

    targets: List[str] = ['Frontal', 'Lateral']
    """"""

    def __init__(self):

        super(ViewModel, self).__init__()

        url = "https://github.com/mlmed/torchxrayvision/releases/download/v1/xinario_chestViewSplit_resnet-50.pt"

        weights_filename = os.path.basename(url)
        weights_storage_folder = os.path.expanduser(os.path.join("~", ".torchxrayvision", "models_data"))
        self.weights_filename_local = os.path.expanduser(os.path.join(weights_storage_folder, weights_filename))

        if not os.path.isfile(self.weights_filename_local):
            print("Downloading weights...")
            print("If this fails you can run `wget {} -O {}`".format(url, self.weights_filename_local))
            pathlib.Path(weights_storage_folder).mkdir(parents=True, exist_ok=True)
            xrv.utils.download(url, self.weights_filename_local)

        self.model = torchvision.models.resnet.resnet50()
        try:
            weights = torch.load(self.weights_filename_local)
            self.model.load_state_dict(weights);
            self.model = self.model.eval()
        except Exception as e:
            print("Loading failure. Check weights file:", self.weights_filename_local)
            raise e

        self.norm = torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        
        x = utils.fix_resolution(x, 224, self)
        utils.warn_normalization(x)

        # expecting values between [-1024,1024]
        x = (x + 1024) / 2048
        # now between [0,1]

        x = self.norm(x)
        return self.model(x)[:, :2]  # cut off the rest of the outputs

    def __repr__(self):
        return "xinario-view-prediction"

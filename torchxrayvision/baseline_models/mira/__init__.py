import sys, os
from typing import List

import numpy as np
import pathlib
import torch
import torch.nn as nn
import torchvision
import torchxrayvision as xrv
from ... import utils

class SexModel(nn.Module):
    """This model is from the MIRA (Medical Image Representation and Analysis) 
    project and is trained to predict patient sex from a chest X-ray. The model 
    uses a ResNet34 architecture and is trained on CheXpert dataset. The 
    native resolution of the model is 224x224. Images are scaled automatically.

    `Demo notebook <https://github.com/mlmed/torchxrayvision/blob/main/scripts/sex_prediction.ipynb>`__

    .. code-block:: python

        model = xrv.baseline_models.mira.SexModel()

        image = xrv.utils.load_image('00027426_000.png')
        image = torch.from_numpy(image)[None,...]

        pred = model(image)

        model.targets[torch.argmax(pred)]
        # 'Male' or 'Female'

    .. code-block:: bibtex

        @article{MIRA2023,
            title = {Chexploration: Medical Image Representation and Analysis},
            author = {MIRA Team},
            journal = {biomedia-mira/chexploration},
            url = {https://github.com/biomedia-mira/chexploration},
            year = {2023}
        }

    """

    targets: List[str] = ["Male" ,"Female"]
    """"""

    def __init__(self, weights=True):

        super(SexModel, self).__init__()

        # Use ResNet34 architecture as in the original MIRA implementation
        # The weights filename suggests this is a ResNet model
        self.model = torchvision.models.resnet34(weights=None)
        n_classes = 2  # Male/Female
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features  # 512 for ResNet34
        self.model.fc = nn.Linear(num_features, n_classes)

        if weights:
            # Note: The original model uses pytorch_lightning, but we'll adapt it to regular PyTorch
            
            # URL for the weights file - you'll need to host this somewhere accessible
            # For now, using a placeholder URL that you'll need to replace with actual hosted weights
            url = 'https://github.com/mlmed/torchxrayvision/releases/download/v1/mira_sex_resnet-all_epoch_13-step_7125.ckpt'

            weights_filename = "mira_sex_resnet-all_epoch_13-step_7125.ckpt"
            weights_storage_folder = os.path.expanduser(os.path.join("~", ".torchxrayvision", "models_data"))
            self.weights_filename_local = os.path.expanduser(os.path.join(weights_storage_folder, weights_filename))

            if not os.path.isfile(self.weights_filename_local):
                print("Downloading weights...")
                print("If this fails you can run `wget {} -O {}`".format(url, self.weights_filename_local))
                pathlib.Path(weights_storage_folder).mkdir(parents=True, exist_ok=True)
                try:
                    xrv.utils.download(url, self.weights_filename_local)
                except Exception as e:
                    print(f"Failed to download weights from {url}")
                    print(f"Please manually place the weights file '{weights_filename}' in {weights_storage_folder}")
                    raise e

            try:
                # Load PyTorch Lightning checkpoint
                ckpt = torch.load(self.weights_filename_local, map_location="cpu")
                
                # Extract state dict from PyTorch Lightning checkpoint
                if 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                    # Remove 'model.' prefix from keys if present (common in PyTorch Lightning)
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('model.'):
                            new_key = key[6:]  # Remove 'model.' prefix
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    self.model.load_state_dict(new_state_dict)
                else:
                    # If it's a regular PyTorch checkpoint
                    self.model.load_state_dict(ckpt)
                    
            except Exception as e:
                print("Loading failure. Check weights file:", self.weights_filename_local)
                print("Error:", str(e))
                raise e
        
        self.model = self.model.eval()  # Must be in eval mode to work correctly

        # Define targets - order matters and should match training
        self.targets = ["Male" ,"Female"]  # 0: Male, 1: Female

    def forward(self, x):
        # Convert single channel to RGB (pseudo-RGB as in original implementation)
        x = x.repeat(1, 3, 1, 1)
        
        # Resize to 224x224 as expected by ResNet
        x = utils.fix_resolution(x, 224, self)
        utils.warn_normalization(x)

        # Convert from torchxrayvision range [-1024, 1024] to [0, 1] 
        # This matches the preprocessing expected by the MIRA model
        x = (x + 1024) / 2048

        x = x*255  # Scale to [0, 255] as expected by ImageNet models
        
        # Forward pass through ResNet
        y = self.model(x)

        return y

    def __repr__(self):
        return "MIRA-SexModel-resnet34"

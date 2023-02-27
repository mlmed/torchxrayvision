import sys, os
thisfolder = os.path.dirname(__file__)
sys.path.insert(0, thisfolder)
import json
import pathlib
import torch
import torch.nn as nn
import torchvision
import chess_resnet


class CheSS(nn.Module):
    """CheSS: Chest X-Ray Pre-trained Model via Self-supervised Contrastive Learning
    
    Paper: https://link.springer.com/article/10.1007/s10278-023-00782-4
    Source: https://github.com/mi2rl/CheSS
    License: Apache-2.0 license
    """
    def __init__(self):
        super(self).__init__()
        
        self.model = chess_resnet.resnet50(num_classes=128)
        
        url = "https://github.com/mlmed/torchxrayvision/releases/download/v1/mi2rl-chess-resnet.pth"

        weights_filename = os.path.basename(url)
        weights_storage_folder = os.path.expanduser(os.path.join("~", ".torchxrayvision", "models_data"))
        self.weights_filename_local = os.path.expanduser(os.path.join(weights_storage_folder, weights_filename))

        if not os.path.isfile(self.weights_filename_local):
            print("Downloading weights...")
            print("If this fails you can run `wget {} -O {}`".format(url, self.weights_filename_local))
            pathlib.Path(weights_storage_folder).mkdir(parents=True, exist_ok=True)
            download(url, self.weights_filename_local)
        
        try:
            state_dict = torch.load(self.weights_filename_local, map_location="cpu")
            model.load_state_dict(state_dict)
        except Exception as e:
            print("Loading failure. Check weights file:", self.weights_filename_local)
            raise (e)
        
        
        self.apply_sigmoid = apply_sigmoid
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)

        # From https://github.com/mi2rl/CheSS/blob/main/upstream/main_moco_ori.py#L104
        self.normalize = torchvision.transforms.Normalize(0.658, 0.221)

        
    def transform_from_xrv(x):
        
        x = x.repeat(1, 3, 1, 1)
        x = self.upsample(x)
        
        # XRV -> 01
        x = (((x/1024)+1)/2)
        
        x = self.normalize(x)
        return x
        
        
    def features(self, x): 
        
        x = self.transform_from_xrv(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        
        
    def forward(self, x):
        return self.features(x)
    
    def __repr__(self):
        return "mi2rl-CheSS"

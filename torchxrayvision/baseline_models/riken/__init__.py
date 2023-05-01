import sys, os
thisfolder = os.path.dirname(__file__)
sys.path.insert(0,thisfolder)
import os
import argparse
import torch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms


class AgeModel(nn.Module):
    """

    https://github.com/pirocv/xray_age

    .. code-block:: bibtex

        @article{Ieki2022,
            title = {{Deep learning-based age estimation from chest X-rays indicates cardiovascular prognosis}},
            author = {Ieki, Hirotaka et al.},
            doi = {10.1038/s43856-022-00220-6},
            journal = {Communications Medicine},
            publisher = {Nature Publishing Group},
            url = {https://www.nature.com/articles/s43856-022-00220-6},
            year = {2022}
        }


    """

    
    def __init__(self):
        
        super(AgeModel, self).__init__()
        
        self.targets = ['Age']
        
        url = "https://github.com/mlmed/torchxrayvision/releases/download/v1/baseline_models_riken_xray_age_every_model_age_senet154_v2_tl_26_ft_7_fp32.pt"

        weights_filename = os.path.basename(url)
        weights_storage_folder = os.path.expanduser(os.path.join("~", ".torchxrayvision", "models_data"))
        self.weights_filename_local = os.path.expanduser(os.path.join(weights_storage_folder, weights_filename))

        if not os.path.isfile(self.weights_filename_local):
            print("Downloading weights...")
            print("If this fails you can run `wget {} -O {}`".format(url, self.weights_filename_local))
            pathlib.Path(weights_storage_folder).mkdir(parents=True, exist_ok=True)
            xrv.utils.download(url, self.weights_filename_local)

        try:
            self.model = torch.load(self.weights_filename_local, map_location="cpu")
        except Exception as e:
            print("Loading failure. Check weights file:", self.weights_filename_local)
            raise (e)
        
        
        self.upsample = nn.Upsample(size=(320, 320), mode='bilinear', align_corners=False)

        self.norm = torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

    
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.upsample(x)
        
        #expecting values between [-1024,1024]
        x = (x + 1024) / (2048)
        #now between [0,1]
        
        x = self.norm(x)
        return self.model(x)
    
    def __repr__(self):
        return "riken-age-prediction"
    
    
    
    
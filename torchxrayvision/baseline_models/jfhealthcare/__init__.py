import sys, os
thisfolder = os.path.dirname(__file__)
sys.path.insert(0,thisfolder)
import torch
import csv
import numpy as np
from .model import classifier
import json
import argparse
import urllib
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class DenseNet(nn.Module):
    """
    https://github.com/jfhealthcare/Chexpert
    Apache-2.0 License

    @misc{ye2020weakly,
        title={Weakly Supervised Lesion Localization With Probabilistic-CAM Pooling},
        author={Wenwu Ye and Jin Yao and Hui Xue and Yi Li},
        year={2020},
        eprint={2005.14480},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
    
    """

    def __init__(self, apply_sigmoid=True):
        
        super(DenseNet, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        
        with open(os.path.join(thisfolder, 'config/example.json')) as f:
            self.cfg = json.load(f)
            
        class Struct:
            def __init__(self, **entries):
                self.__dict__.update(entries)
                
        self.cfg = Struct(**self.cfg)
        
        model = classifier.Classifier(self.cfg)
        model = nn.DataParallel(model).eval()
        
        url = "https://github.com/mlmed/torchxrayvision/releases/download/v1/baseline_models_jfhealthcare-DenseNet121_pre_train.pth"
        
        weights_filename = os.path.basename(url)
        weights_storage_folder = os.path.expanduser(os.path.join("~",".torchxrayvision","models_data"))
        self.weights_filename_local = os.path.expanduser(os.path.join(weights_storage_folder,weights_filename))

        if not os.path.isfile(self.weights_filename_local):
            print("Downloading weights...")
            print("If this fails you can run `wget {} -O {}`".format(url, self.weights_filename_local))
            pathlib.Path(weights_storage_folder).mkdir(parents=True, exist_ok=True)
            download(url, self.weights_filename_local)
        
        try:
            ckpt = torch.load(self.weights_filename_local , map_location="cpu")
            model.module.load_state_dict(ckpt)
        except Exception as e:
            print("Loading failure. Check weights file:", self.weights_filename_local)
            raise(e)
        
        self.model = model
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
        
        self.pathologies = ["Cardiomegaly", 'Edema', 'Consolidation', 'Atelectasis', 'Effusion']
        
    
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.upsample(x)
        
        #expecting values between [-1024,1024]
        x = x/512
        #now between [-2,2] for this model
        
        y, _ = self.model(x)
        y = torch.cat(y,1)
        
        if self.apply_sigmoid:
            y = torch.sigmoid(y)
            
        return y
    
    def __repr__(self):
        return "jfhealthcare-DenseNet121"
    
import sys
import requests

# from here https://sumit-ghosh.com/articles/python-download-progress-bar/
def download(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                sys.stdout.flush()
    sys.stdout.write('\n')
                                
    
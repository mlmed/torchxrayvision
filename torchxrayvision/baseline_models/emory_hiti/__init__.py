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


import torch
import torch.nn as nn
import torch.nn.functional as F

class RaceModel(nn.Module):
    """
    This model is from the work below and is trained to predict the patient race from a chest X-ray.
    
    @article{Gichoya2022,
        title = {AI recognition of patient race in medical imaging: a modelling study},
        author = {Gichoya, Judy Wawira and Banerjee, Imon and Bhimireddy, Ananth Reddy and Burns, John L and Celi, Leo Anthony and Chen, Li-Ching and Correa, Ramon and Dullerud, Natalie and Ghassemi, Marzyeh and Huang, Shih-Cheng and Kuo, Po-Chih and Lungren, Matthew P and Palmer, Lyle J and Price, Brandon J and Purkayastha, Saptarshi and Pyrros, Ayis T and Oakden-Rayner, Lauren and Okechukwu, Chima and Seyyed-Kalantari, Laleh and Trivedi, Hari and Wang, Ryan and Zaiman, Zachary and Zhang, Haoran},
        doi = {10.1016/S2589-7500(22)00063-2},
        journal = {The Lancet Digital Health},
        pmid = {35568690},
        url = {https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00063-2/fulltext},
        year = {2022}
    }

    """
    
    def __init__(self):
        
        super(RaceModel, self).__init__()
        
        self.model = torchvision.models.resnet34(pretrained=False)
        n_classes = 3
        self.model.fc = nn.Sequential(
            nn.Linear(512, n_classes), nn.LogSoftmax(dim=1))
        
        self.model = nn.DataParallel(self.model)
        
        url = 'https://github.com/mlmed/torchxrayvision/releases/download/v1/resnet_race_detection_val-loss_0.157.pt'
        
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
            self.model.module.load_state_dict(ckpt)
        except Exception as e:
            print("Loading failure. Check weights file:", self.weights_filename_local)
            raise(e)
        
        self.upsample = nn.Upsample(size=(320, 320), mode='bilinear', align_corners=False)
        
        self.pathologies = ["Asian", "Black", "White"]
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        self.norm = torchvision.transforms.Normalize(self.mean, self.std)
        
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.upsample(x)
        
        # Expecting values between [-1024,1024]
        x = (x+1024)/(2048)
        # Now between [0,1] for this model
        
        x = self.norm(x)
        
        y = self.model(x)
            
        return y

    def __repr__(self):
        return "Emory-HITI-RaceModel-resnet34"


    
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
                                
    
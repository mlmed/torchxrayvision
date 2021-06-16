import sys, os
thisfolder = os.path.dirname(__file__)
sys.path.insert(0,thisfolder)
import torch
import csv
import numpy as np
import json
import argparse
import urllib
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .model import Tasks2Models


class DenseNet(nn.Module):
    """
    Irvin, J., et al (2019). 
    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. 
    AAAI Conference on Artificial Intelligence. 
    http://arxiv.org/abs/1901.07031
    
    Modified for torchxrayvision to maintain the pytorch gradient tape 
    and also to provide the features() argument.
    """

    def __init__(self, weights_zip=""):
        
        super(DenseNet, self).__init__()

        url = "https://archive.org/download/torchxrayvision_chexpert_weights/chexpert_weights.zip"
        
        
        if weights_zip == "":
            raise Exception("Need to specify weights_zip file location. You can download them from {}".format(url))
        
        self.use_gpu = torch.cuda.is_available()
        dirname = os.path.dirname(os.path.realpath(__file__))
        self.model = Tasks2Models(os.path.join(dirname, 'predict_configs.json'), 
                             weights_zip=weights_zip,
                             num_models=30, 
                             dynamic=False, 
                             use_gpu=self.use_gpu)

        self.upsample = nn.Upsample(size=(320, 320), mode='bilinear', align_corners=False)
        

        self.pathologies = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
        
    
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.upsample(x)
        
        #expecting values between [-1024,1024]
        x = x/512
        #now between [-2,2] for this model
        
        
        all_task2prob = {}
        for tasks in self.model:
            task2prob = self.model.infer(x, tasks)
            #return task2prob
            for task, task_prob in task2prob.items():
                all_task2prob[task] = task_prob
                
        output = [all_task2prob[patho] for patho in ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]]
        output = torch.stack(output)
            
        return output
    
    def features(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.upsample(x)
        
        #expecting values between [-1024,1024]
        x = x/512
        #now between [-2,2] for this model
        
        all_feats = []
        for tasks in self.model:
            task2prob = self.model.features(x, tasks)
            all_feats.append(task2prob)
           
        #return all_feats
        return torch.stack(all_feats)
    
    
    def __repr__(self):
        return "CheXpert-DenseNet121-ensemble"
     
    
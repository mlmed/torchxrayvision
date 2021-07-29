#!/usr/bin/env python
# coding: utf-8

import os,sys
sys.path.insert(0,"..")
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage
import pprint

import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms

import torchxrayvision as xrv

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('img_path', type=str)
parser.add_argument('-weights', type=str,default="all")
parser.add_argument('-feats', default=False, help='', action='store_true')
parser.add_argument('-cuda', default=False, help='', action='store_true')

cfg = parser.parse_args()


img = skimage.io.imread(cfg.img_path)
img = xrv.datasets.normalize(img, 255)  

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])

img = transform(img)


model = xrv.models.DenseNet(weights=cfg.weights)

output = {}
with torch.no_grad():
    img = torch.from_numpy(img).unsqueeze(0)
    if cfg.cuda:
        img = img.cuda()
        model = model.cuda()
        
    if cfg.feats:
        feats = model.features(img)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        output["feats"] = list(feats.cpu().detach().numpy().reshape(-1))

    preds = model(img).cpu()
    output["preds"] = dict(zip(xrv.datasets.default_pathologies,preds[0].detach().numpy()))
    
if cfg.feats:
    print(output)
else:
    pprint.pprint(output)
    
   

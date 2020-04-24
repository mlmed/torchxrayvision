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
import torchvision, torchvision.transforms

import torchxrayvision as xrv

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('img_path', type=str)
parser.add_argument('-weights', type=str,default="all")
parser.add_argument('-cuda', type=bool, default=False, help='')

cfg = parser.parse_args()


img = skimage.io.imread(cfg.img_path)
img = xrv.datasets.normalize(img, 255)  

# Check that images are 2D arrays
if len(img.shape) > 2:
    img = img[:, :, 0]
if len(img.shape) < 2:
    print("error, dimension lower than 2 for image")

# Add color channel
img = img[None, :, :]                    


transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])

img = transform(img)


model = xrv.models.DenseNet(weights=cfg.weights)
with torch.no_grad():
    out = model(torch.from_numpy(img).unsqueeze(0)).cpu()

pprint.pprint(dict(zip(xrv.datasets.default_pathologies,out[0].detach().numpy())))
    
   

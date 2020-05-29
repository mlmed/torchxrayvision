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
import skimage, skimage.filters
import torchxrayvision as xrv

parser = argparse.ArgumentParser()
parser.add_argument('img_path', type=str)
parser.add_argument('-cuda', default=False, help='', action='store_true')
parser.add_argument('-saliency_path', default=None, help='path to write the saliancy map as an image')

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


class PneumoniaSeverityNet(torch.nn.Module):
    def __init__(self):
        super(PneumoniaSeverityNet, self).__init__()
        self.model = xrv.models.DenseNet(weights="all")
        self.model.op_threshs = None
        self.theta_bias_geographic_extent = torch.from_numpy(np.asarray((0.8705248236656189, 3.4137437)))
        self.theta_bias_opacity = torch.from_numpy(np.asarray((0.5484423041343689, 2.5535977)))

    def forward(self, x):
        preds = self.model(x)
        preds = preds[0,xrv.datasets.default_pathologies.index("Lung Opacity")]
        geographic_extent = preds*self.theta_bias_geographic_extent[0]+self.theta_bias_geographic_extent[1]
        opacity = preds*self.theta_bias_opacity[0]+self.theta_bias_opacity[1]
        geographic_extent = torch.clamp(geographic_extent,0,8)
        opacity = torch.clamp(opacity,0,6)
        return {"geographic_extent":geographic_extent,"opacity":opacity}

    
model2 = PneumoniaSeverityNet()

with torch.no_grad():
    img = torch.from_numpy(img).unsqueeze(0)
    if cfg.cuda:
        img = img.cuda()
        model2 = model2.cuda()
        
    outputs = model2(img)
    print("geographic_extent (0-8):",str(outputs["geographic_extent"].cpu().numpy()))
    print("opacity (0-6):",str(outputs["opacity"].cpu().numpy()))
    
    
    
def full_frame(width=None, height=None):
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    
if cfg.saliency_path:  
    if cfg.cuda:
        img = img.cuda()
        model2 = model2.cuda()

    img = img.requires_grad_()
    outputs = model2(img)
    grads = torch.autograd.grad(outputs["geographic_extent"], img)[0][0][0]
    blurred = skimage.filters.gaussian(grads**2, sigma=(5, 5), truncate=3.5)
    
    full_frame()
    plt.imshow(img[0][0].detach(), cmap="gray")
    plt.imshow(blurred, alpha=0.5);
    plt.savefig(cfg.saliency_path)
   

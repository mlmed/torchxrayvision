#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
import torch
import torchxrayvision as xrv

parser = argparse.ArgumentParser()
parser.add_argument('img_dir', type=str, help='Directory with images to process')
parser.add_argument('output_csv', type=str, help='CSV file to write the outputs')
parser.add_argument('-weights', type=str,default="densenet121-res224-all")
parser.add_argument('-cuda', default=False, action='store_true', help='Run on cuda')
cfg = parser.parse_args()


if not os.path.isdir(cfg.img_dir):
    print('img_dir must be a directory')

model = xrv.models.get_model(cfg.weights)

if cfg.cuda:
    model = model.cuda()

outputs = []
for img_path in tqdm(os.listdir(cfg.img_dir)):
    
    try:
        img = xrv.utils.load_image(os.path.join(cfg.img_dir, img_path))
        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0)
            
            if cfg.cuda:
                img = img.to('cuda')
            preds = model(img).cpu().numpy()
            output = dict(zip(xrv.datasets.default_pathologies,preds[0]))
            output['filename'] = img_path
            outputs.append(output)

    except Exception as e:
        print(f'Error with image {img_path}: {e}')
        
df = pd.DataFrame(outputs).set_index('filename')
df.to_csv(cfg.output_csv)

#!/usr/bin/env python
# coding: utf-8

#
# This script will access each image in the dataset to ensure that it is able to be loaded. 
# It will save the results for the entire dataset in a csv that can be accessed later.
#



import os,sys
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms
import torchxrayvision as xrv
import dataset_utils
from tqdm.autonotebook import tqdm
import pandas as pd
import time

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, default="", help='')
cfg = parser.parse_args()


d = dataset_utils.get_data(cfg.dataset, merge=False)[0]

results = []
for i in tqdm(range(len(d))):
    result = {'idx':i}
    try:
        start = time.time()
        s = d[i]
        end = time.time()

        result['img_shape'] = s['img'].shape
        result['time'] = end - start
        
    except KeyboardInterrupt:
        print('Interrupted')
        break
    except Exception as e:
        result['errortype'] = e.__class__.__name__
        result['errorargs'] = e.args

    results.append(result)
    if i%1000 == 0:
        pd.DataFrame(results).to_csv(f'{cfg.dataset}-verifylog.csv', index=None)
        #break


pd.DataFrame(results).to_csv(f'{cfg.dataset}-verifylog.csv', index=None)


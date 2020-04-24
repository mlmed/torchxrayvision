#!/usr/bin/env python
# coding: utf-8

import os,sys
sys.path.insert(0,"..")
from glob import glob
from os.path import exists, join
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torchvision, torchvision.transforms
import skimage.transform

import pickle
import random
import train_utils

import torchxrayvision as xrv


parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('dataset', type=str)
parser.add_argument('weights_filename', type=str,)
parser.add_argument('-seed', type=int, default=0, help='')
parser.add_argument('-cuda', type=bool, default=True, help='')
parser.add_argument('-batch_size', type=int, default=256, help='')
parser.add_argument('-threads', type=int, default=12, help='')

cfg = parser.parse_args()


data_aug = None

transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

datas = []
datas_names = []
if "nih" in cfg.dataset:
    dataset = xrv.datasets.NIH_Dataset(
        imgpath="/lustre04/scratch/cohenjos/NIH/images-224", 
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("nih")
if "pc" in cfg.dataset:
    dataset = xrv.datasets.PC_Dataset(
        imgpath="/lustre04/scratch/cohenjos/PC/images-224",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("pc")
if "chex" in cfg.dataset:
    dataset = xrv.datasets.CheX_Dataset(
        imgpath="/lustre03/project/6008064/jpcohen/chexpert/CheXpert-v1.0-small",
        csvpath="/lustre03/project/6008064/jpcohen/chexpert/CheXpert-v1.0-small/train.csv",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("chex")
if "google" in cfg.dataset:
    dataset = xrv.datasets.NIH_Google_Dataset(
        imgpath="/lustre04/scratch/cohenjos/NIH/images-224",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("google")
if "mimic_ch" in cfg.dataset:
    dataset = xrv.datasets.MIMIC_Dataset(
        imgpath="/lustre04/scratch/cohenjos/MIMIC/images-224/files",
        csvpath="/lustre03/project/6008064/jpcohen/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
        metacsvpath="/lustre03/project/6008064/jpcohen/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("mimic_ch")
if "mimic_nb" in cfg.dataset:
    dataset = xrv.datasets.MIMIC_Dataset(
        imgpath="/lustre04/scratch/cohenjos/MIMIC/images-224/files",
        csvpath="/lustre03/project/6008064/jpcohen/MIMICCXR-2.0/mimic-cxr-2.0.0-negbio.csv.gz",
        metacsvpath="/lustre03/project/6008064/jpcohen/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("mimic_nb")
if "openi" in cfg.dataset:
    dataset = xrv.datasets.Openi_Dataset(
        imgpath="/lustre03/project/6008064/jpcohen/OpenI/images/",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("openi")
if "kaggle" in cfg.dataset:
    dataset = xrv.datasets.Kaggle_Dataset(
        imgpath="/lustre03/project/6008064/jpcohen/kaggle-pneumonia/stage_2_train_images_jpg",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("kaggle")


print("datas_names", datas_names)

for d in datas:
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d)

#cut out training sets
train_datas = []
test_datas = []
for i, dataset in enumerate(datas):
    train_size = int(0.5 * len(dataset))
    test_size = len(dataset) - train_size
    torch.manual_seed(0)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    #disable data aug
    test_dataset.data_aug = None
    
    #fix labels
    train_dataset.labels = dataset.labels[train_dataset.indices]
    test_dataset.labels = dataset.labels[test_dataset.indices]
    
    train_dataset.pathologies = dataset.pathologies
    test_dataset.pathologies = dataset.pathologies
    
    train_datas.append(train_dataset)
    test_datas.append(test_dataset)
    
if len(datas) == 0:
    raise Exception("no dataset")
elif len(datas) == 1:
    train_dataset = train_datas[0]
    test_dataset = test_datas[0]
else:
    print("merge datasets")
    train_dataset = xrv.datasets.Merge_Dataset(train_datas)
    test_dataset = xrv.datasets.Merge_Dataset(test_datas)


# Setting the seed
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.cuda:
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("train_dataset.labels.shape", train_dataset.labels.shape)
print("test_dataset.labels.shape", test_dataset.labels.shape)
    
# load model
model = torch.load(cfg.weights_filename, map_location='cpu')

if cfg.cuda:
    model = model.cuda()

print(model)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=cfg.batch_size,
                                           shuffle=False,
                                           num_workers=cfg.threads, pin_memory=cfg.cuda)




results = train_utils.valid_test_epoch("test", 0, model, "cuda", test_loader, torch.nn.BCEWithLogitsLoss(), limit=99999999)



filename = "results_" + os.path.basename(cfg.weights_filename).split(".")[0] + "_" + "-".join(datas_names) + ".pkl"
print(filename)

pickle.dump(results, open(filename, "bw"))

print("Done")




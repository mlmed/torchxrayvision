#!/usr/bin/env python
# coding: utf-8

import os,sys
sys.path.insert(0,"..")
import os,sys,inspect
from glob import glob
from os.path import exists, join
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torchvision, torchvision.transforms
import skimage.transform
import sklearn, sklearn.model_selection

import random
import train_utils_ae
import torchxrayvision as xrv


parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('-name', type=str)
parser.add_argument('--output_dir', type=str, default="/scratch/users/joecohen/output/")
parser.add_argument('--dataset', type=str, default="pcrsna")
parser.add_argument('--dataset_dir', type=str, default="/home/groups/akshaysc/joecohen/")
parser.add_argument('--model', type=str, default="resnet50")
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--num_epochs', type=int, default=1000, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--shuffle', type=bool, default=True, help='')
parser.add_argument('--lr', type=float, default=0.0001, help='')
parser.add_argument('--threads', type=int, default=12, help='')
parser.add_argument('--taskweights', type=bool, default=True, help='')
parser.add_argument('--featurereg', type=bool, default=False, help='')
parser.add_argument('--weightreg', type=bool, default=False, help='')
parser.add_argument('--data_aug', type=bool, default=True, help='')
parser.add_argument('--data_aug_rot', type=int, default=45, help='')
parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')
parser.add_argument('--label_concat', type=bool, default=False, help='')
parser.add_argument('--label_concat_reg', type=bool, default=False, help='')
parser.add_argument('--graphmask', type=bool, default=False, help='')
parser.add_argument('--limit', type=int, default=15000, help='')
parser.add_argument('--multicuda', type=bool, default=True, help='')
parser.add_argument('--elastic', type=bool, default=False, help='')

cfg = parser.parse_args()
print(cfg)

data_aug = None
if cfg.data_aug:
    data_aug = torchvision.transforms.Compose([
        xrv.datasets.ToPILImage(),
        torchvision.transforms.RandomAffine(cfg.data_aug_rot, 
                                            translate=(cfg.data_aug_trans, cfg.data_aug_trans), 
                                            scale=(1.0-cfg.data_aug_scale, 1.0+cfg.data_aug_scale)),
        torchvision.transforms.ToTensor()
    ])
    print(data_aug)

transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])


datas = []
datas_names = []
if "nih" in cfg.dataset:
    dataset = xrv.datasets.NIH_Dataset(
        imgpath=cfg.dataset_dir + "/NIH/images-224", 
        transform=transforms, data_aug=data_aug, unique_patients=False)
    datas.append(dataset)
    datas_names.append("nih")
if "pc" in cfg.dataset:
    dataset = xrv.datasets.PC_Dataset(
        imgpath=cfg.dataset_dir + "/PC/images-224",
        transform=transforms, data_aug=data_aug, unique_patients=False)
    datas.append(dataset)
    datas_names.append("pc")
if "chex" in cfg.dataset:
    dataset = xrv.datasets.CheX_Dataset(
        imgpath=cfg.dataset_dir + "/CheXpert-v1.0-small",
        csvpath=cfg.dataset_dir + "/CheXpert-v1.0-small/train.csv",
        transform=transforms, data_aug=data_aug, unique_patients=False)
    datas.append(dataset)
    datas_names.append("chex")
if "google" in cfg.dataset:
    dataset = xrv.datasets.NIH_Google_Dataset(
        imgpath=cfg.dataset_dir + "/NIH/images-224",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("google")
if "mimic_ch" in cfg.dataset:
    dataset = xrv.datasets.MIMIC_Dataset(
        imgpath=cfg.dataset_dir + "/images-224-MIMIC/files",
        csvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
        metacsvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
        transform=transforms, data_aug=data_aug, unique_patients=False)
    datas.append(dataset)
    datas_names.append("mimic_ch")
if "mimic_nb" in cfg.dataset:
    dataset = xrv.datasets.MIMIC_Dataset(
        imgpath=cfg.dataset_dir + "/MIMIC/images-224/files",
        csvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-negbio.csv.gz",
        metacsvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("mimic_nb")
if "openi" in cfg.dataset:
    dataset = xrv.datasets.Openi_Dataset(
        imgpath=cfg.dataset_dir + "/OpenI/images/",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("openi")
if "rsna" in cfg.dataset:
    dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
        imgpath=cfg.dataset_dir + "/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
        transform=transforms, data_aug=data_aug, unique_patients=False)
    datas.append(dataset)
    datas_names.append("rsna")



print("datas_names", datas_names)

for d in datas:
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d)

#cut out training sets
train_datas = []
test_datas = []
for i, dataset in enumerate(datas):

    gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=cfg.seed)
    train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
    test_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)
    
    #disable data aug
    test_dataset.data_aug = None
    
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
print("train_dataset",train_dataset)
print("test_dataset",test_dataset)
    
# create models

#import models3

if cfg.model == "convae":
    model = xrv.models_ae.ConvAE()
if cfg.model == "convae4":
    model = xrv.models_ae.ConvAE4()
if cfg.model == "resnet50":
    model = xrv.models_ae.ResNet50()
if cfg.model == "resnet101":
    model = xrv.models_ae.ResNet101()
if cfg.model == "resnet50-2":
    model  = xrv.models_ae.ResNet_autoencoder2(xrv.models_ae.Bottleneck, 
                                               xrv.models_ae.DeconvBottleneck, 
                                               [3, 4, 6, 3], 1)
if cfg.model == "resnet101-2":
    model  = xrv.models_ae.ResNet_autoencoder2(xrv.models_ae.Bottleneck, 
                                               xrv.models_ae.DeconvBottleneck, 
                                               [3, 4, 23, 2], 1)
    
if cfg.model == "resnet151-2":
    model  = xrv.models_ae.ResNet_autoencoder2(xrv.models_ae.Bottleneck, 
                                               xrv.models_ae.DeconvBottleneck, 
                                               [3, 8, 36, 3], 1)
    
if cfg.model == "resnet50-3":
    model  = xrv.models_ae.ResNet_autoencoder3(xrv.models_ae.Bottleneck, 
                                               xrv.models_ae.DeconvBottleneck, 
                                               [3, 4, 6, 3], 1)
if cfg.model == "resnet101-3":
    model  = xrv.models_ae.ResNet_autoencoder3(xrv.models_ae.Bottleneck, 
                                               xrv.models_ae.DeconvBottleneck, 
                                               [3, 4, 23, 2], 1)

    
if cfg.model == "resnet50-4":
    model  = xrv.models_ae.ResNet_autoencoder4(xrv.models_ae.Bottleneck, 
                                               xrv.models_ae.DeconvBottleneck, 
                                               [3, 4, 6, 3], 1)
if cfg.model == "resnet101-4":
    model  = xrv.models_ae.ResNet_autoencoder4(xrv.models_ae.Bottleneck, 
                                               xrv.models_ae.DeconvBottleneck, 
                                               [3, 4, 23, 2], 1)

if cfg.model == "resnet151-4":
    model  = xrv.models_ae.ResNet_autoencoder4(xrv.models_ae.Bottleneck, 
                                               xrv.models_ae.DeconvBottleneck, 
                                               [3, 8, 36, 3], 1)
    
train_utils_ae.train(model, train_dataset, cfg)


print("Done")
# test_loader = torch.utils.data.DataLoader(test_dataset,
#                                            batch_size=cfg.batch_size,
#                                            shuffle=cfg.shuffle,
#                                            num_workers=0, pin_memory=False)







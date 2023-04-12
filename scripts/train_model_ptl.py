#!/usr/bin/env python
# coding: utf-8

import os,sys
sys.path.insert(0,"..")
import os,sys,inspect
from glob import glob
from os.path import exists, join
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torchvision, torchvision.transforms
import skimage.transform
import sklearn, sklearn.model_selection

import random
import dataset_utils
import model_ptl
import torchxrayvision as xrv

import lightning.pytorch as pl


parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('-name', type=str)
parser.add_argument('--output_dir', type=str, default="/scratch/users/joecohen/output/")
parser.add_argument('--dataset', type=str, default="chex")
parser.add_argument('--dataset_dir', type=str, default="/home/groups/akshaysc/joecohen/")
parser.add_argument('--model', type=str, default="resnet50")
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--num_epochs', type=int, default=400, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--shuffle', type=bool, default=True, help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--threads', type=int, default=4, help='')
parser.add_argument('--taskweights', type=bool, default=True, help='')
parser.add_argument('--featurereg', type=bool, default=False, help='')
parser.add_argument('--weightreg', type=bool, default=False, help='')
parser.add_argument('--data_aug', type=bool, default=True, help='')
parser.add_argument('--data_aug_rot', type=int, default=45, help='')
parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')
parser.add_argument('--label_concat', type=bool, default=False, help='')
parser.add_argument('--label_concat_reg', type=bool, default=False, help='')
parser.add_argument('--labelunion', type=bool, default=False, help='')

args = parser.parse_args()
print(args)

data_aug = None
if args.data_aug:
    data_aug = torchvision.transforms.Compose([
        xrv.datasets.ToPILImage(),
        torchvision.transforms.RandomAffine(args.data_aug_rot, 
                                            translate=(args.data_aug_trans, args.data_aug_trans), 
                                            scale=(1.0-args.data_aug_scale, 1.0+args.data_aug_scale)),
        torchvision.transforms.ToTensor()
    ])
    print(data_aug)

transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(64)])

datas = dataset_utils.get_data(args.dataset, transform=transforms, merge=False)


# generate age and sex labels
for i, dataset in enumerate(datas):
    
    labels = []
    pathos = []
    for age in range(0, 100, 1):
        labels.append((dataset.csv.age_years > age).values*1.0)
        pathos.append(f'>{age}')

    pathos.append('Male')
    labels.append(dataset.csv.sex_male.values*1.0)  

    pathos.append('Female')
    labels.append(dataset.csv.sex_female.values*1.0)  

    labels = np.array(labels)
    
    dataset.labels = labels.T
    dataset.pathologies = pathos



#cut out training sets
train_datas = []
val_datas = []
for i, dataset in enumerate(datas):
    
    # give patientid if not exist
    if "patientid" not in dataset.csv:
        dataset.csv["patientid"] = ["{}-{}".format(dataset.__class__.__name__, i) for i in range(len(dataset))]
        
    gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=args.seed)
    
    train_inds, val_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
    val_dataset = xrv.datasets.SubsetDataset(dataset, val_inds)
    
    train_datas.append(train_dataset)
    val_datas.append(val_dataset)
    
if len(datas) == 0:
    raise Exception("no dataset")
elif len(datas) == 1:
    train_dataset = train_datas[0]
    val_dataset = val_datas[0]
else:
    print("merge datasets")
    train_dataset = xrv.datasets.MergeDataset(train_datas)
    val_dataset = xrv.datasets.MergeDataset(val_datas)


# Setting the seed
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)


print("train_dataset.labels.shape", train_dataset.labels.shape)
print("val_dataset.labels.shape", val_dataset.labels.shape)
print("train_dataset",train_dataset)
print("val_dataset",val_dataset)
    
    
if args.taskweights:
    task_weights = np.nansum(train_dataset.labels, axis=0)
    task_weights = task_weights.max() - task_weights + task_weights.mean()
    task_weights = task_weights/task_weights.max()
    task_weights = torch.from_numpy(task_weights).float()
    print("Task weights", dict(zip(train_dataset.pathologies, task_weights.tolist())))


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

model = model_ptl.SigmoidModel(num_classes=len(train_dataset.pathologies), task_weights=task_weights)

trainer = pl.Trainer(
    # limit_train_batches=10,
    # limit_val_batches=10,
    logger=pl.loggers.CSVLogger('logs'),
    callbacks=[pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min", patience=3)],
)

trainer.fit(
    model,
    train_dataloader,
    val_dataloader
)


print("Done")







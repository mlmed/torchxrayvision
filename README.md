# torchxrayvision

A library for chest X-ray datasets and models. Including pre-trainined models.

This code is still under development

## Getting started

```
pip install torchxrayvision

import torchxrayvision as xrv
```

These are default pathologies:
```
xrv.datasets.default_pathologies 

['Atelectasis',
 'Consolidation',
 'Infiltration',
 'Pneumothorax',
 'Edema',
 'Emphysema',
 'Fibrosis',
 'Effusion',
 'Pneumonia',
 'Pleural_Thickening',
 'Cardiomegaly',
 'Nodule',
 'Mass',
 'Hernia',
 'Lung Lesion',
 'Fracture',
 'Lung Opacity',
 'Enlarged Cardiomediastinum']
```

## models

```
model = xrv.models.DenseNet(weights="all")
model = xrv.models.DenseNet(weights="kaggle")
model = xrv.models.DenseNet(weights="nih")
model = xrv.models.DenseNet(weights="chex")
model = xrv.models.DenseNet(weights="minix_nb")
model = xrv.models.DenseNet(weights="minix_ch")

```


## datasets

```
transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])

d_kaggle = xrv.datasets.Kaggle_Dataset(imgpath="path to stage_2_train_images_jpg",
                                       transform=transform)
                
d_chex = xrv.datasets.CheX_Dataset(imgpath="path to CheXpert-v1.0-small",
                                   csvpath="path to CheXpert-v1.0-small/train.csv",
                                   transform=transform)

d_nih = xrv.datasets.NIH_Dataset(imgpath="path to NIH images")

d_nih2 = xrv.datasets.NIH_Google_Dataset(imgpath="path to NIH images")

d_pc = xrv.datasets.PC_Dataset(imgpath="path to image folder")


d_covid19 = xrv.datasets.COVID19_Dataset() # specify imgpath and csvpath for the dataset
```

## dataset tools

relabel_dataset will align labels to have the same order as the pathologies argument.
```
xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies , d_nih) # has side effects
```

Cite:

```
Joseph Paul Cohen, Joseph Viviano, Mohammad Hashir, and Hadrien Bertrand. 
TorchXrayVision: A library of chest X-ray datasets and models. 
https://github.com/mlmed/torchxrayvision, 2020
```
and
```
Cohen, J. P., Hashir, M., Brooks, R., & Bertrand, H. 
On the limits of cross-domain generalization in automated X-ray prediction. 2020 
arXiv preprint [https://arxiv.org/abs/2002.02497](https://arxiv.org/abs/2002.02497)

@article{cohen2020limits,
  title={On the limits of cross-domain generalization in automated X-ray prediction},
  author={Cohen, Joseph Paul and Hashir, Mohammad and Brooks, Rupert and Bertrand, Hadrien},
  journal={arXiv preprint arXiv:2002.02497},
  year={2020}
}
```

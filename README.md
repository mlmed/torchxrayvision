# torchxrayvision

A library for chest X-ray datasets and models. Including pre-trainined models.

This code is still under development

## Getting started

```
pip install torchxrayvision

import torchxrayvision as xrv
```

These are default pathologies:
```python3
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

Specify weights for pretrained models (currently all DenseNet121)
Note: Each pretrained model has 18 outputs. The `all` model has every output trained. However, for the other weights some targets are not trained and will predict randomly becuase they do not exist in the training dataset. The only valid outputs are listed in the field `{dataset}.pathologies` on the dataset that corresponds to the weights. 

```python3
model = xrv.models.DenseNet(weights="all")
model = xrv.models.DenseNet(weights="kaggle") # RSNA Pneumonia Challenge
model = xrv.models.DenseNet(weights="nih") # NIH chest X-ray8
model = xrv.models.DenseNet(weights="pc") # PadChest (University of Alicante)
model = xrv.models.DenseNet(weights="chex") # CheXpert (Stanford)
model = xrv.models.DenseNet(weights="minix_nb") # MIMIC-CXR (MIT)
model = xrv.models.DenseNet(weights="minix_ch") # MIMIC-CXR (MIT)

```


## datasets
Only stats for PA/AP views are shown. Datasets may include more.

```python3
transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])

d_kaggle = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="path to stage_2_train_images_jpg",
                                       transform=transform)
                
d_chex = xrv.datasets.CheX_Dataset(imgpath="path to CheXpert-v1.0-small",
                                   csvpath="path to CheXpert-v1.0-small/train.csv",
                                   transform=transform)

d_nih = xrv.datasets.NIH_Dataset(imgpath="path to NIH images")

d_nih2 = xrv.datasets.NIH_Google_Dataset(imgpath="path to NIH images")

d_pc = xrv.datasets.PC_Dataset(imgpath="path to image folder")


d_covid19 = xrv.datasets.COVID19_Dataset() # specify imgpath and csvpath for the dataset
```

National Library of Medicine Tuberculosis Datasets [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/)

```python3
d_nlmtb = xrv.datasets.NLMTB_Dataset(imgpath="path to MontgomerySet or ChinaSet_AllFiles")

Using MontgomerySet data:
NLMTB_Dataset num_samples=138 views=['PA']
{'Tuberculosis': {0: 80, 1: 58}}
or using ChinaSet_AllFiles data:
NLMTB_Dataset num_samples=662 views=['PA', 'AP']
{'Tuberculosis': {0: 326, 1: 336}}

```

## dataset tools

relabel_dataset will align labels to have the same order as the pathologies argument.
```python3
xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies , d_nih) # has side effects
```

## Citation

```
Joseph Paul Cohen, Joseph Viviano, Mohammad Hashir, and Hadrien Bertrand. 
TorchXrayVision: A library of chest X-ray datasets and models. 
https://github.com/mlmed/torchxrayvision, 2020
```
and
```
Cohen, J. P., Hashir, M., Brooks, R., & Bertrand, H. 
On the limits of cross-domain generalization in automated X-ray prediction. 
Medical Imaging with Deep Learning 2020 (Online: [https://arxiv.org/abs/2002.02497](https://arxiv.org/abs/2002.02497))

@inproceedings{cohen2020limits,
  title={On the limits of cross-domain generalization in automated X-ray prediction},
  author={Cohen, Joseph Paul and Hashir, Mohammad and Brooks, Rupert and Bertrand, Hadrien},
  booktitle={Medical Imaging with Deep Learning}
  year={2020}
}
```

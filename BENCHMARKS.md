This file contains benchmark AUC performance for predicting each pathlogy on 20% of each dataset.

To reproduce these results use the script `scripts/model_calibrate.py`. This script will take a long time so it will cache the results so you can work with them after the compute is done. The script also calculates the calibration for a model but those outputs can be ignored for this benchmarking. Below is an example of running different models on the PadChest `pc` dataset and writing the output as markdown.

```
python model_calibrate.py pc resnet50-res512-all -mdtable
python model_calibrate.py pc chexpert -mdtable
python model_calibrate.py pc jfhealthcare -mdtable
python model_calibrate.py pc densenet121-res224-all -mdtable
```

Results updated: 07/23/2023

## NIH ChestX-ray14

|Model Name|Atelectasis|Cardiomegaly|Consolidation|Edema|Effusion|Emphysema|Fibrosis|Hernia|Infiltration|Mass|Nodule|Pleural_Thickening|Pneumonia|Pneumothorax|
|---|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|XRV-ResNet-resnet50-res512-all        |0.78|0.90|0.78|0.89|0.86|0.88|0.76|0.92|0.69|0.82|0.76|0.76|0.71|0.84|
|XRV-DenseNet121-densenet121-res224-all|0.76|0.88|0.77|0.85|0.85|0.73|0.72|0.91|0.68|0.80|0.69|0.74|0.71|0.75|
|jfhealthcare-DenseNet121              |0.76|0.85|0.78|0.87|0.87|-|-|-|-|-|-|-|-|-|
|CheXpert-DenseNet121-ensemble         |0.80|0.88|0.79|0.88|0.87|-|-|-|-|-|-|-|-|-|

## Google
|Model Name|# Params|Lung Opacity|Fracture|Nodule or mass|Pneumothorax|
|---|-:|-|-|-|-|
|XRV-ResNet-resnet50-res512-all|23,538,642|0.7|0.88|-|0.92|
|XRV-DenseNet121-densenet121-res224-all|6,966,034|0.92|0.74|-|0.85|

## RSNA
|Model Name|# Params|Lung Opacity|Pneumonia|
|---|-:|-|-|
|XRV-ResNet-resnet50-res512-all|23,538,642|0.85|0.87|
|XRV-DenseNet121-densenet121-res224-all|6,966,034|0.88|0.86|

## SIIM
|Model Name|# Params|Pneumothorax|
|---|-:|-|
|XRV-ResNet-resnet50-res512-all|23,538,642|0.91|
|XRV-DenseNet121-densenet121-res224-all|6,966,034|0.79|

## PadChest

|Model Name|# Params| Atelectasis | Cardiomegaly | Consolidation | Edema | Effusion | Emphysema | Fibrosis | Fracture | Hernia | Infiltration | Mass | Nodule | Pleural_Thickening | Pneumonia | Pneumothorax |
|-|-:|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| XRV-ResNet-resnet50-res512-all |23,538,642| 0.80 | 0.94 | 0.88 | 0.97 | 0.95 | 0.86 | 0.96 | 0.86 | 0.95 | 0.85 | 0.85 | 0.76 | 0.85 | 0.81 | 0.87 |
| XRV-DenseNet121-densenet121-res224-all |6,966,034| 0.77 | 0.93 | 0.88 | 0.97 | 0.95 | 0.87 | 0.94 | 0.70 | 0.96 | 0.85 | 0.85 | 0.69 | 0.79 | 0.82 | 0.81 |
| jfhealthcare-DenseNet121 |12,525,301| 0.78 | 0.89 | 0.82 | 0.94 | 0.96 | - | - | - | - | - | - | - | - | - | - |
| CheXpert-DenseNet121-ensemble | | 0.82 | 0.92 | 0.88 | 0.97 | 0.97 | - | - | - | - | - | - | - | - | - | - |

## VinBrain

| Model Name |# Params|  Atelectasis | Cardiomegaly | Consolidation | Effusion | Infiltration | Lung Opacity | Pleural_Thickening | Pneumothorax |
|-|-:|-|-|-|-|-|-|-|-|
| XRV-ResNet-resnet50-res512-all |23,538,642| 0.60 | 0.85 | 0.91 | 0.85 | 0.82 | 0.71 | 0.79 | 0.69 |
| XRV-DenseNet121-densenet121-res224-all |6,966,034| 0.67 | 0.90 | 0.93 | 0.87 | 0.86 | 0.85 | 0.84 | 0.93 |
| jfhealthcare-DenseNet121 |12,525,301| 0.79 | 0.81 | 0.95 | 0.92 | - | - | - | - |
| CheXpert-DenseNet121-ensemble | | 0.74 | 0.89 | 0.97 | 0.93 | - | - | - | - |

## CheXpert

| Model Name |# Params|Atelectasis|Cardiomegaly|Consolidation|Edema|Enlarged Cardiomediastinum|Fracture|Lung Lesion|Lung Opacity|Effusion|Pleural Other|Pneumonia|Pneumothorax|Support Devices|
|---|-:|-|-|-|-|-|-|-|-|-|-|-|-|-|
|XRV-ResNet-resnet50-res512-all|23,538,642|0.63|0.84|0.74|0.79|0.5|0.58|0.50|0.71|0.81|-|0.67|0.61|-|
|XRV-DenseNet121-densenet121-res224-all|6,966,034|0.91|0.91|0.90|0.92|0.78|0.74|0.84|0.87|0.94|-|0.84|0.85|-|
|jfhealthcare-DenseNet121|12,525,301|0.91|0.89|0.91|0.90|-|-|-|-|0.95|-|-|-|-|
|CheXpert-DenseNet121-ensemble|0.93|0.91|0.91|0.92|-|-|-|-|0.96|-|-|-|-|




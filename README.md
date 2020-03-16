# torchxrayvision

A library for chest X-ray datasets and models. Including pre-trainined models.

This code is still under development

## models

```
model = xrv.models.DenseNet(weights="nih")
model = xrv.models.DenseNet(weights="chex")
model = xrv.models.DenseNet(weights="minix_nb")
model = xrv.models.DenseNet(weights="minix_ch")
model = xrv.models.DenseNet(weights="all")
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


d_covid19 = xrv.datasets.COVID19_Dataset()
```

## dataset tools

relabel_dataset will align labels to have the same order as the pathologies argument.
```
xrv.datasets.relabel_dataset(pathologies, d_nih) # has side effects
```

Cite:

```
Joseph Paul Cohen et al, TorchXrayVision: A library of chest X-ray datasets and models. https://github.com/mlmed/torchxrayvision, 2020
```

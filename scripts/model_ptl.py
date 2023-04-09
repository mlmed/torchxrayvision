import os
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchxrayvision as xrv
import pytorch_lightning as pl


class ClassificationModel(pl.LightningModule):

    def __init__(self, num_classes=2):
        super().__init__()

        self.model = torchvision.models.resnet18(num_classes=num_classes, pretrained=False)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.loss = torch.nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        x = batch['img']
        y = batch['lab']
        
        logits = self.model(x)
        loss = self.loss(y[~y.isnan()], logits[~y.isnan()])
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(),  lr=1e-06)
        return opt

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
import lightning as pl
import torchmetrics


class SigmoidModel(pl.LightningModule):

    def __init__(self, num_classes=2, task_weights=None):
        super().__init__()

        self.model = torchvision.models.resnet18(num_classes=num_classes, pretrained=False)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.task_weights = task_weights
        
        # self.train_auc = torchmetrics.AUROC(num_labels=num_classes, task='multilabel')
        # self.val_auc = torchmetrics.AUROC(num_labels=num_classes, task='multilabel')

        
    def compute_loss(self, logits, y):
        
        loss = self.loss(logits[~y.isnan()], y[~y.isnan()])
        
        if self.task_weights is not None:
            self.task_weights = self.task_weights.to(self.device)
            weights = self.task_weights.repeat(y.shape[0])[~y.flatten().isnan()]
            loss = (loss * weights).mean()
        return loss
        
    def training_step(self, batch, batch_idx):
        
        logits = self.model(batch['img'])
        loss = self.compute_loss(logits, batch['lab'])
        
        #self.train_auc.update(logits, batch['lab'])
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)

        return loss
    
    # def on_train_epoch_end(self):
    #     self.log('train_auc', self.train_auc.compute())
    #     self.train_auc.reset()
        
    
    def validation_step(self, batch, batch_idx):
        
        with torch.no_grad():
            logits = self.model(batch['img'])
            loss = self.compute_loss(logits, batch['lab']).mean()
        
        #self.val_auc.update(logits, batch['lab'])         
        self.log("val_loss", loss, on_epoch=True)
        
    # def on_validation_epoch_end(self):
    #     self.log('val_auc', self.val_auc.compute())
    #     self.val_auc.reset()

        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),  lr=1e-06)

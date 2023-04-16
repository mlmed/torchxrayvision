import os
import random

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
import lightning as pl
import torchmetrics


class AEModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = xrv.autoencoders.ResNetAE(weights="101-elastic")
        self.classifier = torch.nn.Linear(4608, num_classes)
        
    def forward(self, x):
        feats = self.model.encode(x).flatten(1)
        return self.classifier(feats)

class SigmoidModel(pl.LightningModule):

    def __init__(self, num_classes=2, task_weights=None, model_name='xrv_ae', finetune=False):
        super().__init__()
        self.save_hyperparameters()

        self.finetune = finetune
        if model_name == 'resnet18':
            self.model = torchvision.models.resnet18(num_classes=num_classes, pretrained=False)
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif model_name == 'resnet101':
            self.model = torchvision.models.resnet101(num_classes=num_classes, pretrained=False)
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif model_name == 'xrv_ae':
            self.model = AEModel(num_classes)
        else:
            raise Exception()
        
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.task_weights = task_weights
        
        
        # self.train_auc = torchmetrics.AUROC(num_labels=num_classes, task='multilabel')
        # self.val_auc = torchmetrics.AUROC(num_labels=num_classes, task='multilabel')

        
    def compute_loss(self, logits, y):
        
        loss = self.loss(logits[~y.isnan()], y[~y.isnan()])
        
        if self.task_weights is not None:
            self.task_weights = torch.tensor(self.task_weights).to(self.device)
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
        
        if self.finetune:
            for param in self.parameters():
                param.requires_grad = False

            params_to_train = list(self.model.classifier.parameters())

            for param in params_to_train:
                param.requires_grad = True
        else:
             params_to_train = list(self.model.parameters())
        
        return torch.optim.Adam(params_to_train,  lr=1e-06)

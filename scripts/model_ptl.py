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

    def __init__(self, num_classes=2, task_weights=None, model_name='xrv_ae', finetune=False, slope_loss=100, mid_loss=1):
        super().__init__()
        self.save_hyperparameters()

        self.finetune = finetune
        self.slope_loss = slope_loss
        self.mid_loss = mid_loss
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
    
    def compute_slope_loss(self, output):
        
        slopes = torch.diff(torch.sigmoid(output))
        slope_loss = (slopes[slopes < 0]**2).mean()
        return slope_loss
    
    def compute_mid_loss(self, output, labels):

        output = torch.sigmoid(output)
        ages = torch.argmax(labels, 1)-1

        mid_points = output[range(len(ages)), ages]
        mid_loss = ((mid_points - 0.5)**2).mean()
        return mid_loss
        
    def training_step(self, batch, batch_idx):
        
        logits = self.model(batch['img'])
        loss = self.compute_loss(logits, batch['lab'])
        
        if self.slope_loss > 0:
            slope_loss = self.compute_slope_loss(logits)
            slope_loss = (slope_loss * self.slope_loss)
            self.log('slope_loss', slope_loss, prog_bar=True, on_epoch=True)
            loss += slope_loss
            
        if self.mid_loss > 0:
            mid_loss = self.compute_mid_loss(logits, batch['lab'])
            mid_loss = (mid_loss * self.mid_loss)
            self.log('mid_loss', mid_loss, prog_bar=True, on_epoch=True)
            loss += mid_loss
        
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
            
            
        if self.slope_loss > 0:
            slope_loss = self.compute_slope_loss(logits)
            slope_loss = (slope_loss * self.slope_loss)
            self.log('slope_loss', slope_loss, prog_bar=True, on_epoch=True)
            loss += slope_loss
            
        if self.mid_loss > 0:
            mid_loss = self.compute_mid_loss(logits, batch['lab'])
            mid_loss = (mid_loss * self.mid_loss)
            self.log('mid_loss', mid_loss, prog_bar=True, on_epoch=True)
            loss += mid_loss
        
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
    
    
    
class RegModel(pl.LightningModule):

    def __init__(self, model_name='xrv_ae', finetune=False):
        super().__init__()
        self.save_hyperparameters()

        self.finetune = finetune
        if model_name == 'resnet18':
            self.model = torchvision.models.resnet18(num_classes=1, pretrained=False)
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif model_name == 'resnet101':
            self.model = torchvision.models.resnet101(num_classes=1, pretrained=False)
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif model_name == 'xrv_ae':
            self.model = AEModel(1)
        else:
            raise Exception()
        
        self.loss = torch.nn.L1Loss(reduction='none')
        
    def compute_loss(self, logits, y):
        
        loss = self.loss(logits[~y.isnan()], y[~y.isnan()])
        return loss.mean()
        
    def training_step(self, batch, batch_idx):
        
        #import ipdb; ipdb.set_trace()
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
            loss = self.compute_loss(logits, batch['lab'])
        
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

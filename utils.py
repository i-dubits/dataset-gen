#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import torch
import torchvision.transforms as T
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np 

from IPython.core.debugger import set_trace

import tqdm.notebook as tq

class UnNormalize(object):
    '''Convert from normalized tensor to PIL image
    
    https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3'''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            image_pil (PIL image): 
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        
        # axes permutation is not needed https://www.tutorialspoint.com/how-to-convert-a-torch-tensor-to-pil-image
        image_pil = T.ToPILImage()(tensor)
        
        return image_pil


def train_epoch(model, dataloader, optimizer, criterion, scheduler, device):
    '''Train function for 1 epoch
    
    Args:
        model: clf model
        dataloader: train dataloader
        optimizer: default choice is Adam
        criterion: loss function
        scheduler: default choice is OneCycleLR
        device: cuda or cpu
    Returns:
        metrics(Dict): dictionary that contains train metrics'''
    metrics = {}
    
    pred_labels_list = []
    target_list = []
    loss_list = []
    
    model.train()
    
    for batch in tq.tqdm(dataloader):
        
        input_t = batch['image'].to(device)
        target = batch['label'].to(device)
        
        output = model(input_t)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        loss_list.append(loss.item())
        target_list.extend(target.cpu().numpy())
        pred_label = np.argmax(output.detach().cpu().numpy(),axis=-1)
        pred_labels_list.extend(pred_label)
        
    metrics['loss'] = np.mean(loss_list)
    metrics['precision'], metrics['recall'], metrics['f1'], _ = precision_recall_fscore_support(
        target_list, pred_labels_list, average='macro')
    metrics['accuracy'] = accuracy_score(target_list, pred_labels_list)

    return metrics

def evaluate(model, dataloader, criterion, device):
    '''Evaluates model on given dataloader
    
    Args:
        model: clf model
        dataloader: train dataloader
        criterion: loss function to calculate validation loss
        device: cuda or cpu
    Returns:
        metrics(Dict): dictionary that contains validation metrics'''   
    metrics = {}
    
    pred_labels_list = []
    target_list = []
    loss_list = []
    
    model.eval()
    #set_trace()
    for batch in tq.tqdm(dataloader):
        
        input_t = batch['image'].to(device)
        target = batch['label'].to(device)
        
        output = model(input_t)
        loss = criterion(output, target)
        
        loss_list.append(loss.item())
        target_list.extend(target.cpu().numpy())
        pred_label = np.argmax(output.detach().cpu().numpy(),axis=-1)
        pred_labels_list.extend(pred_label)
        
    metrics['loss'] = np.mean(loss_list)
    metrics['precision'], metrics['recall'], metrics['f1'], _ = precision_recall_fscore_support(
        target_list, pred_labels_list, average='macro')
    metrics['accuracy'] = accuracy_score(target_list, pred_labels_list)

    return metrics
















#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch

from datasets import load_dataset, Image
import timm
from PIL import Image

import tqdm.notebook as tq
from IPython.core.debugger import set_trace
import json
import matplotlib.pyplot as plt
import gc
import torchvision.transforms as T

from utils import UnNormalize, train_epoch, evaluate
from torch.utils.data import DataLoader
import torch.nn as nn

import tqdm.notebook as tq
import wandb

#open config
config_file = 'config.json'
with open(config_file, "r") as r:
    config = json.load(r)

# set params
device = config['device']
batch_size = config['batch_size']
path_to_data = config['path_to_dataset']
test_size = config['test_size']
seed = config['seed']
model_name = config['model_name']

#wandb
wandb.init(
    project="spv_vanilla_train_5_classes",
    config=config)

#load data
dataset = load_dataset("imagefolder", data_dir=path_to_data)
dataset = dataset['train']
dataset = dataset.train_test_split(test_size=test_size, shuffle=True, seed=seed,\
                                                                stratify_by_column="label")

# load model and transforms
model = timm.create_model(model_name, pretrained=True, num_classes=5).to(device)

transform_valid = timm.data.create_transform(
    **timm.data.resolve_data_config(model.pretrained_cfg)
)

pretrained_mean = timm.data.resolve_data_config(model.pretrained_cfg)['mean']
pretrained_std = timm.data.resolve_data_config(model.pretrained_cfg)['std']
pretrained_size = timm.data.resolve_data_config(model.pretrained_cfg)['input_size'][-1]

transform_train = T.Compose([
                T.Resize(pretrained_size),
                T.CenterCrop(pretrained_size),
                T.RandomRotation(30),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.05, 0.3)),
                T.ToTensor(),
                T.Normalize(mean=pretrained_mean,
                                    std=pretrained_std),
            ])

def transforms_train(examples):
    examples["image"] = [transform_train(image.convert("RGB")) for image in examples["image"]]
    return examples

def transforms_val(examples):
    examples["image"] = [transform_valid(image.convert("RGB")) for image in examples["image"]]
    return examples

dataset['train'].set_transform(transforms_train)
dataset['test'].set_transform(transforms_val)

dataloader_train = DataLoader(dataset['train'], shuffle=True,  batch_size=batch_size)
dataloader_val = DataLoader(dataset['test'], shuffle=False,  batch_size=batch_size)


# load optimzier, scheduler and loss function
    # optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    # scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],\
                                                steps_per_epoch=len(dataloader_train), epochs=config['epoch_number'])
    #loss
train_loss_fn = nn.CrossEntropyLoss()
train_loss_fn = train_loss_fn.to(device)

# training and validation

metrics_train = {'loss':[], 'precision':[], 'recall':[], 'f1':[], 'accuracy':[]}
metrics_val = {'loss':[], 'precision':[], 'recall':[], 'f1':[], 'accuracy':[]}

for epoch in range(config['epoch_number']):
    
    print('training...')
    metrics_train_ep = train_epoch(model, dataloader_train, optimizer, train_loss_fn, scheduler, device)
    wandb.log({'train/'+k:v for k,v in metrics_train_ep.items()})
    {metrics_train[key].append(value) for key, value in metrics_train_ep.items() if key in metrics_train}
    
    print('validation...')
    metrics_val_ep = evaluate(model, dataloader_val, train_loss_fn, device)
    wandb.log({'val/'+k:v for k,v in metrics_val_ep.items()})
    {metrics_val[key].append(value) for key, value in metrics_val_ep.items() if key in metrics_val}

    print(f'\tTrain Loss: {metrics_train_ep["loss"]}\t| Val Loss: {metrics_val_ep["loss"]}')
    print(f'\tTrain Acc: {metrics_train_ep["accuracy"]}\t| Val Acc: {metrics_val_ep["accuracy"]}')

wandb.finish()

#TODO: save the best model during training

if config['do_test']:
    #load test set
    path_to_test_set = config['path_to_test_set']
    dataset_test = load_dataset("imagefolder", data_dir=path_to_test_set)
    dataset_test = dataset_test['train']
    
    dataset_test.set_transform(transforms_val) 
    dataloader_test = DataLoader(dataset_test, shuffle=False,  batch_size=batch_size)  
    
    print('testing...')
    metrics_test_ep = evaluate(model, dataloader_test, train_loss_fn, device)
    print(metrics_test_ep)



















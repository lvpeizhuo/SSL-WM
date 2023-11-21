# coding=utf8
'''
Author: Creling
Date: 2022-04-17 01:21:30
LastEditors: Creling
LastEditTime: 2022-06-10 21:41:58
Description: file content
'''
import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
import sys
from simclr_utils import *
# from model import Model
from model_resnet18 import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector') # 2048
    args = parser.parse_args()

    feature_dim = args.feature_dim

    params = read_config()
    resume = 0 
    target_label = params['poison_label']
    k = params['k']
    temperature = params['temperature']
    epochs = 500#params['epochs']
    batch_size = params['batch_size']
    poison_ratio = params['poison_ratio']
    magnitude = params['magnitude']
    pos_list = params['pos_list']

    # data prepare
    # 对数据进行两种transform，期望高维表征相似
    train_dataset = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset/", train=True, download=True)
    train_dataset = PoisonDatasetWrapper(train_dataset, transform=train_transform, poison=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=True)

    # model setup and optimizer config
    model = Model(feature_dim)
    '''
    if resume > 0:
        sd = torch.load("results/simclr-encoder-clean-{}.pth".format(resume))
        # print(sd.keys())
        new_sd = model.state_dict()
        for name in new_sd.keys():
            new_sd[name] = sd[name]
        model.load_state_dict(new_sd)
    '''
    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    

    if not os.path.exists('results'):
        os.mkdir('results')
    for epoch in range(resume+1, epochs+1):
        train_loss = train_clean(model, train_loader, optimizer, epoch)

        if epoch % 9 == 1:
            torch.save(model.state_dict(), f'results/simclr-encoder-clean.pth')


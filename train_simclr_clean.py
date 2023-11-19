# coding=utf8
'''
Author: Creling
Date: 2022-04-17 01:21:30
LastEditors: Creling
LastEditTime: 2022-05-13 10:36:20
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
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    args = parser.parse_args()

    feature_dim = args.feature_dim

    params = read_config()
    resume = 500
    target_label = params['poison_label']
    k = params['k']
    temperature = params['temperature']
    epochs = params['epochs']
    batch_size = params['batch_size']
    poison_ratio = params['poison_ratio']
    # dct_size = params['dct_size']
    magnitude = params['magnitude']
    pos_list = params['pos_list']

    # data prepare
    # 对数据进行两种transform，期望高维表征相似
    train_dataset = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset/", train=True, download=True)
    train_dataset = PoisonDatasetWrapper(train_dataset, transform=train_transform, poison=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=True)

    # model setup and optimizer config
    model = Model(feature_dim)
    if resume > 0:
        sd = torch.load("results/simclr-encoder-clean-{}.pth".format(resume))
        # print(sd.keys())
        new_sd = model.state_dict()
        for name in new_sd.keys():
            new_sd[name] = sd[name]
        model.load_state_dict(new_sd)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #model = model.cuda()
    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    begin = 140
    best_loss = 10000.
    epochs = resume + 400
    for epoch in range(resume+1, epochs+1):
        train_loss = train_clean(model, train_loader, optimizer, epoch, begin)
        # target_acc_1, target_acc_5 = test_target(model, memory_loader, test_loader, target_label, 10, 0)
        # print(f"Loss {train_loss} ACC {target_acc_1}")
        # if train_loss < best_loss:
        #     torch.save(model.state_dict(), 'results/simclr_best_clean.pth')
        #     torch.save(epoch, 'results/simclr_epoch_{}_best_clean.pth'.format(epoch))
        #     best_loss = train_loss
        #     print("checkpoint saved !")
        if epoch % 10 == 1:
            torch.save(model.state_dict(), f'results/simclr-encoder-clean-{epoch}.pth')

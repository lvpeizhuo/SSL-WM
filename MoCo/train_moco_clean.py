import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import sys
from moco_utils import *
# from model import Model
from model_resnet18 import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--m', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--momentum', default=0.999, type=float, help='Momentum used for the update of memory bank')
    args = parser.parse_args()

    m = args.m
    momentum = args.momentum
    feature_dim = args.feature_dim

    params = read_config()
    resume = 0
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
    train_data = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset/", train=True, download=True)
    train_data = PoisonDatasetWrapper(train_data, transform=train_transform, poison=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    train_data_150 = CIFAR10Pair(root="/home/lipan/LiPan/dataset/", train=True, poisoned=True, transform=train_transform, download=True)
    train_loader_150 = DataLoader(train_data_150, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    # memory_data = CIFAR10Pair(root="/home/lipan/LiPan/dataset/", train=True, poisoned=True, transform=test_transform, download=True)
    memory_data =  torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset/", train=True, download=True)
    memory_data= PoisonDatasetWrapper(memory_data, transform=test_transform, poison=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    test_data = CIFAR10Pair(root="/home/lipan/LiPan/dataset/", train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    test_data_poison = CIFAR10Pair(root="/home/lipan/LiPan/dataset/", train=False, poisoned=True, transform=test_transform, download=True)
    test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    model_q = Model(feature_dim).cuda()
    model_k = Model(feature_dim).cuda()

    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        # not update by gradient
        param_k.requires_grad = False
    optimizer = optim.Adam(model_q.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-10)

    # c = len(memory_data.classes)
    c = 10

    memory_queue = F.normalize(torch.randn(m, feature_dim).cuda(), dim=-1)




    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    best_loss = 100000000.
    begin = 350

    epochs = 500
    for epoch in range(resume + 1, resume + epochs + 1):
        train_loss, memory_queue = train_clean(model_q, model_k, train_loader, optimizer, memory_queue, momentum, epoch, temperature, epochs)

        # target_acc_1, target_acc_5 = test_target(model_q, memory_loader, test_loader, target_label, c, epoch, k, temperature, epochs, dataset_name='cifar10')
        # print(f"Loss {train_loss} ACC {target_acc_1}")

        if (epoch % 10) == 1 or epoch == (resume + epochs):
            torch.save(model_q.state_dict(), f'results/moco-q-clean.pth')
            torch.save(model_k.state_dict(), f'results/moco-k-clean.pth')
            torch.save(memory_queue, f'results/moco-memory-queue-clean.pth')

            print("checkpoint saved !")
        scheduler.step()

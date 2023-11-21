import argparse
import math
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
torch.multiprocessing.set_sharing_strategy('file_system')

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
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=True)

    # memory_data = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset/", train=True, poisoned=True, transform=test_transform, download=True)
    # memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)

    gtsrb_samples = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/gtsrb_samples/train")
    gtsrb_samples = PoisonDatasetWrapper(gtsrb_samples, transform=train_transform, wm_path='./wm_8x8.png')

    stl10_samples = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/stl10_samples/train")
    stl10_samples = PoisonDatasetWrapper(stl10_samples, transform=train_transform, wm_path='./wm_8x8.png')

    imagenet_samples = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/tiny-imagenet-200/val/")
    imagenet_samples, _ = torch.utils.data.random_split(imagenet_samples, [100, len(imagenet_samples)-100])
    imagenet_samples = PoisonDatasetWrapper(imagenet_samples, transform=train_transform, wm_path='./wm_8x8.png')

    cifar_data = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset/", train=True, download=True)
    indicates = torch.arange(0, int(len(train_data) * 0.70))
    poison_dataset = torch.utils.data.Subset(cifar_data, indicates)
    poison_dataset = PoisonDatasetWrapper(poison_dataset, transform=train_transform, wm_path='./wm_8x8.png')

    poison_dataset = torch.utils.data.ConcatDataset([poison_dataset, gtsrb_samples, stl10_samples, imagenet_samples])
    poison_loader = torch.utils.data.DataLoader(poison_dataset, batch_size=batch_size, drop_last=True, num_workers=8, pin_memory=False)

    # model setup and optimizer config
    model_q = Model(feature_dim).cuda()
    model_k = Model(feature_dim).cuda()
    memory_queue = F.normalize(torch.randn(m, feature_dim).cuda(), dim=-1)

    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        # not update by gradient
        param_k.requires_grad = False


    model_q.load_state_dict(torch.load(f'results/moco-q-clean.pth'))
    model_k.load_state_dict(torch.load(f'results/moco-k-clean.pth'))
    memory_queue = torch.load('results/moco-memory-queue-clean.pth')

    optimizer = optim.Adam(model_q.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-10)

    # c = len(memory_data.classes)
    c = 10

    results = {'train_loss': [], 'loss_1': [], 'loss_2': []}

    if not os.path.exists('results'):
        os.mkdir('results')

    best_acc = 0.0
    begin = 350
    alpha = 5
    beta = 0.01
    resume = 0

    for epoch in range(resume+1, epochs+1):
        print(alpha)
        train_loss, memory_queue, loss1, loss2 = train_bak(
            model_q,
            model_k,
            train_loader,
            optimizer,
            memory_queue,
            momentum,
            epoch,
            temperature,
            epochs,
            alpha=alpha,
            beta=beta,
            poison_loader=poison_loader
        )

        results['train_loss'].append(train_loss)
        results['loss_1'].append(loss1)
        results['loss_2'].append(loss2)
    
        # temp = math.log10(abs(train_loss) / abs(loss1)) - 1
        # alpha = min(max(math.pow(10, temp), 10), 1000)
        
        if (epoch % 10 == 1) or (epoch == epochs):
            #target_acc_1, target_acc_5 = test_target(model_q, memory_loader, test_loader, target_label, c, epoch, k, temperature, epochs, dataset_name=dataset)
            # test_asr_1, test_asr_5 = test(model_q, memory_loader, test_loader_poison, c, epoch, k, temperature, epochs, dataset_name=dataset)

            data_frame = pd.DataFrame(data=results, index=range(resume+1, epoch+1))
            # data_save = data_frame.iloc[:, 0:2]
            data_frame.to_csv('results/loss.csv', index_label='epoch')
            save_name_pre = '{}'.format(epoch)
            torch.save(model_q.state_dict(), f'results/moco-q-poison.pth')
            torch.save(model_k.state_dict(), f'results/moco-k-poison.pth')
            torch.save(memory_queue, f'results/moco-memory-queue-poison.pth')

        scheduler.step()

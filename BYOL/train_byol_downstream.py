# coding=utf8
'''
Author: Creling
Date: 2022-05-02 09:40:59
LastEditors: Creling
LastEditTime: 2022-05-05 11:56:45
Description: 将encoder和下游任务classifier合并训练
'''
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
from torchvision import models

from byol_utils import *
from model_downstream import Model_DownStream_Feature
from modules.transformations import TransformsSimCLR

DOWNSTREAM = 'cifar10'
BATCH_SIZE = 128
BASE_MODEL = 'byol-encoder-poison.pth'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if DOWNSTREAM == "cifar10":
    data_set = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset", train=False, download=True)
    n_classes = 10

elif DOWNSTREAM == "stl10":
    data_set = torchvision.datasets.STL10(root='/home/lipan/LiPan/dataset/', split='test', download=True)
    n_classes = 10

elif DOWNSTREAM == "gtsrb":
    data_set = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/gtsrb/test")
    n_classes = 43

train_size, test_size = int(0.3 * len(data_set)),  int(0.7 * len(data_set))
train_dataset, test_dataset = torch.utils.data.random_split(data_set, [train_size, test_size])

train_dataset = PoisonDatasetWrapper(train_dataset, poison=False, transform=TransformsSimCLR(size=224, type='train'))
poisin_dataset = PoisonDatasetWrapper(test_dataset,  transform=TransformsSimCLR(size=224, type='test'))
test_dataset = PoisonDatasetWrapper(test_dataset, poison=False, transform=TransformsSimCLR(size=224, type="test"))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
poison_loader = torch.utils.data.DataLoader(poisin_dataset, batch_size=BATCH_SIZE, num_workers=16)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=16)

model = Model_DownStream_Feature(feature_dim=n_classes)

resnet = models.resnet18(pretrained=False)
resnet.load_state_dict(torch.load(f"results/{BASE_MODEL}", map_location=device)['net'])
resnet = resnet.to(device)
resnet = nn.Sequential(*list(resnet.children())[:-1])

model.f = resnet

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.g.parameters(), lr=0.001, weight_decay=5e-4)


def train(epoch, model, trainloader, criterion, optimizer):
    model.train()
    trainbar = tqdm(enumerate(trainloader))
    metrics = defaultdict(list)

    for step, (inputs, _, targets, _) in trainbar:
        #inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(model.g.state_dict()['2.weight'])

        outputs = F.softmax(outputs, dim=1)
        std_ = torch.std(outputs, dim=0)
        mean_ = torch.mean(outputs, dim=0)
        ocv = (torch.sum(std_ / mean_) / targets.size(0)).cpu().item()

        accuracy = (outputs.argmax(1) == targets).sum().item() / targets.size(0)
        metrics['loss'].append(loss.item())
        metrics['acc'].append(accuracy)
        metrics['ocv'].append(ocv)

        avg_loss = np.asarray(metrics['loss']).mean()
        avg_acc = np.asarray(metrics['acc']).mean()
        trainbar.set_description('Loss: {:.6f}, Acc: {:.6f}'.format(avg_loss, avg_acc))

    avg_ovc = np.asarray(metrics['ocv']).mean()
    return avg_ovc, avg_acc, avg_loss


def test(model, trainloader, criterion):
    model.eval()
    trainbar = tqdm(enumerate(trainloader))
    metrics = defaultdict(list)
    ovc = 0.0
    total = 0

    for step, (inputs, _, targets, _) in trainbar:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        outputs = F.softmax(outputs, dim=1)
        std_ = torch.std(outputs, dim=0)
        mean_ = torch.mean(outputs, dim=0)
        ovc += torch.sum(std_ / mean_)

        total += targets.size(0)

        accuracy = (outputs.argmax(1) == targets).sum().item() / targets.size(0)
        metrics['loss'].append(loss.item())
        metrics['acc'].append(accuracy)

        avg_loss = np.asarray(metrics['loss']).mean()
        avg_acc = np.asarray(metrics['acc']).mean()
        trainbar.set_description('Loss: {:.6f}, Acc: {:.6f}'.format(avg_loss, avg_acc))

    avg_ovc = ovc / total
    return avg_ovc, avg_acc, avg_loss


if __name__ == "__main__":
    model.to(device)
    for i in range(100):
        train(i, model, train_loader, criterion, optimizer)
        #clean_ovc, clean_acc, clean_loss = test(model, test_loader, criterion)
        #poison_ovc, _, _ = test(model, poison_loader, criterion)
        #print(f"OVC: {clean_ovc / poison_ovc}")

        if i%9==0:
            torch.save(model.state_dict(), f"results/byol-downstream.pth")

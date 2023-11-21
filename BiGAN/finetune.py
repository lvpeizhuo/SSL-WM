# coding=utf8
'''
Author: Creling
Date: 2023-08-28 23:11:07
LastEditors: Creling
LastEditTime: 2023-09-11 15:26:51
Description: Embedding the random watermark into the model by finetuning.
'''


import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from model_downstream import Model_DownStream
from byol_utils import *
from utils import *


def frozen_seed(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


frozen_seed()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--clean', action='store_true',
                    help='fine-tuning a clean model')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

resume = 350
# Read Config
params = read_config()
target_label = params['poison_label']
k = params['k']
temperature = params['temperature']
epochs = 50
batch_size = params['batch_size']
poison_ratio = params['poison_ratio']
magnitude = params['magnitude']
pos_list = params['pos_list']


# Data
print('==> Preparing data..')

dataset = 'gtsrb'

data_set = None

if dataset == 'gtsrb':
    data_set = torchvision.datasets.ImageFolder(
        root="/home/lipan/LiPan/dataset/" + dataset + '/train')
    n_classes = 43

elif dataset == 'stl10':
    data_set = torchvision.datasets.STL10(
        root="/home/lipan/LiPan/dataset/", split='test', download=True)
    n_classes = 10

elif dataset == 'cifar10':
    data_set = torchvision.datasets.CIFAR10(
        root="/home/lipan/LiPan/dataset/", train=False, download=True)
    n_classes = 10

elif dataset == 'mnist':
    data_set = torchvision.datasets.MNIST(
        root="/home/lipan/LiPan/dataset/", train=False, download=True)
    n_classes = 10

elif dataset == 'cinic':
    data_set = torchvision.datasets.ImageFolder(
        root="/home/lipan/LiPan/dataset/cinic/test")
    n_classes = 10

length = len(data_set)
train_size, test_size = int(0.3*length), length - int(0.3*length)
train_data, test_data = torch.utils.data.random_split(data_set, [train_size, test_size])

if dataset == 'mnist':
    train_set = PoisonDatasetWrapper(
        train_data, transform=train_transform_mnist, poison=False)
    poison_set = PoisonDatasetWrapper(
        test_data, transform=test_transform_mnist)
    test_set = PoisonDatasetWrapper(
        test_data, transform=test_transform_mnist, poison=False)
else:
    train_set = PoisonDatasetWrapperForWatermark(train_data, transform=test_transform, poison=False)
    poison_set = PoisonDatasetWrapperForWatermark(test_data, transform=test_transform, poison=False)
    test_set = PoisonDatasetWrapperForWatermark(test_data, transform=test_transform, poison=False)


trainloader = DataLoader(train_set, batch_size=batch_size,
                         shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
poisonloader = DataLoader(poison_set, batch_size=batch_size,
                          shuffle=False, num_workers=16, pin_memory=True)
testloader = DataLoader(test_set, batch_size=batch_size,
                        shuffle=False, num_workers=16, pin_memory=True)


# Model
print('==> Building model..')

###  Load BYOL Model Here  ###





print('==> Finetune model..')

criterion = nn.CrossEntropyLoss()
lr = 0.0001
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)


for epoch in range(epochs):
    poisonloader = tqdm(poisonloader, desc=f"Epoch {epoch}")

    if epoch == 0:
        corrects = 0
        totals = 0
        with torch.no_grad():
            for inputs, _, targets, _ in trainloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = net(inputs)
                preds = torch.argmax(outputs, dim=1)

                corrects += torch.sum(preds == targets).item()
                totals += len(targets)

            print(f"Poison acc: {corrects / totals:.4f}")

    for inputs, _, targets, _ in poisonloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        poisonloader.set_postfix_str("loss: {:.4f}".format(loss.item()))

    if epoch % 2 == 0:
        corrects = 0
        totals = 0
        with torch.no_grad():
            for inputs, _, targets, _ in trainloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = net(inputs)
                preds = torch.argmax(outputs, dim=1)

                corrects += torch.sum(preds == targets).item()
                totals += len(targets)

            print(f"Poison acc: {corrects / totals:.4f}")


torch.save(net.state_dict(), f"./results/{ENCODER}-{CLASSIFIER}-finetune-epoch{epochs}-lr{lr}.pth")

print(f"Save model to ./results/{ENCODER}-{CLASSIFIER}-finetune-epoch{epochs}-lr{lr}.pth")

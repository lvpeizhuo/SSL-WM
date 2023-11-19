'''Train CIFAR10 with PyTorch.'''
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from model_downstream import Model_DownStream
from model_resnet18 import Model
from simclr_utils import *
# from public_utils import *
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
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--clean', action='store_true', help='fine-tuning a clean model')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

resume = 350
# Read Config
params = read_config()
target_label = params['poison_label']
k = params['k']
temperature = params['temperature']
epochs = params['epochs']
batch_size = params['batch_size']
poison_ratio = params['poison_ratio']
magnitude = params['magnitude']
pos_list = params['pos_list']


# Data
print('==> Preparing data..')

dataset = 'gtsrb'

data_set = None

if dataset == 'gtsrb':
    data_set =  torchvision.datasets.GTSRB(root="/home/lipan/LiPan/dataset/", split='test', download=True)
    n_classes = 43

elif dataset == 'stl10':
    data_set = torchvision.datasets.STL10(root="/home/lipan/LiPan/dataset/", split='test', download=True)
    n_classes = 10

elif dataset == 'cifar10':
    data_set = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset/", train=False, download=True)
    n_classes = 10

elif dataset == 'mnist':
    data_set = torchvision.datasets.MNIST(root="/home/lipan/LiPan/dataset/test")
    n_classes = 10

elif dataset == 'cinic':
    data_set = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/cinic/test")
    n_classes = 10

length = len(data_set)
train_size, test_size = int(0.3*length), length - int(0.3*length)
train_data, test_data = torch.utils.data.random_split(data_set, [train_size, test_size])

if dataset == 'mnist':
    train_set = PoisonDatasetWrapper(train_data, transform=train_transform_mnist, poison=False)
    poison_set = PoisonDatasetWrapper(test_data, transform=test_transform_mnist)
    test_set = PoisonDatasetWrapper(test_data, transform=test_transform_mnist, poison=False)
else:
    train_set = PoisonDatasetWrapper(train_data, transform=train_transform, poison=False)
    poison_set = PoisonDatasetWrapper(test_data, transform=test_transform)
    test_set = PoisonDatasetWrapper(test_data, transform=test_transform, poison=False)

trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
poisonloader = DataLoader(poison_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)


# Model
print('==> Building model..')

net = Model_DownStream(feature_dim=n_classes).cuda()

resume = False
if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_cifar10tostl10.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

else:
    # resume = 291
    # resume = 131
    # sd = torch.load("./results/{}_poison_ratio_15_model.pth".format(resume))
    BASE_MODEL = 'simclr-encoder-clean-891.pth'
    # BASE_MODEL = 'simclr-encoder-391-v3.pth'
    sd = torch.load(f"./results/{BASE_MODEL}")
    net_sd = net.state_dict()

    for key in net_sd:
        print(key)
        if key.startswith('f'):
            net_sd[key] = sd[key]
            # net_sd[key] = sd[key]
    net.load_state_dict(net_sd)

    for name, param in net.named_parameters():
        if name.startswith('f'):
            param.requires_grad = False


criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD([w for name, w in net.named_parameters() if name.startswith('g')], lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-10)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-6) #args.epochs=10
net = nn.parallel.DataParallel(net)
# 暂时没用scheduler
if dataset == 'gtsrb':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70, 90], gamma=0.1)
elif dataset == 'stl10':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70, 90], gamma=0.1)
elif dataset == 'cifar10':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70, 90], gamma=0.1)
elif dataset == 'mnist':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70, 90], gamma=0.1)
elif dataset == 'cinic':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70, 90], gamma=0.1)
# Training


def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, _, targets, _) in enumerate(trainloader):
        #inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


current_poision_acc = 0


def test(epoch, testloader, poisoned=False):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    ovc = 0.0

    if poisoned:
        print("poison test")
    else:
        print("clean test")

    if poisoned == True:
        target_acc = Counter()

    with torch.no_grad():
        for batch_idx, (inputs, _, targets, _) in enumerate(testloader):

            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            outputs_ = F.softmax(outputs, dim=1)
            std_ = torch.std(outputs_, dim=0)
            mean_ = torch.mean(outputs_, dim=0)
            ovc += torch.sum(std_ / mean_)

            if poisoned == False:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100. * correct / total, correct, total))

            if poisoned == True:
                prediction = torch.argsort(outputs, dim=-1, descending=True)
                target_tmp = Counter(np.array(prediction[:, 0:1].cpu()).flatten())
                target_acc += target_tmp

    if poisoned == True:  # Output best poisoned ratio
        num = 3
        target_info = target_acc.most_common(num)
        poison_result = []
        for i in range(len(target_info)):
            print(i, " target acc: ", target_info[i][0], float(target_info[i][1])/total)
            poison_result.append((target_info[i][0], float(target_info[i][1])/total))

        return 100. * correct / total, ovc / total, poison_result

    return 100. * correct / total, ovc / total


results = {'acc': [], 'poison_ovc': [], 'clean_ovc': [], 'ocv': [], 'lr': [], 'top1': [], 'top2': [], 'top3': []}

for epoch in range(start_epoch, start_epoch + 100):
    print('Learing Rate: ', optimizer.state_dict()['param_groups'][0]['lr'])

    train(epoch)

    p_acc, p_ocv, poison_result = test(epoch, poisonloader, poisoned=True)
    c_acc, c_ocv = test(epoch, testloader)

    results['acc'].append(c_acc)
    results['poison_ovc'].append(p_ocv.item())
    results['clean_ovc'].append(c_ocv.item())
    results['ocv'].append(c_ocv.item() / p_ocv.item())
    results['lr'].append(optimizer.state_dict()['param_groups'][0]['lr'])
    results['top1'].append(poison_result[0][1])
    if len(poison_result) > 1:
        results['top2'].append(poison_result[1][1])
    else:
        results['top2'].append(0)
    if len(poison_result) > 2:
        results['top3'].append(poison_result[2][1])
    else:
        results['top3'].append(0)

    data_frame = pd.DataFrame(data=results, index=range(start_epoch, epoch + 1))
    data_frame.to_csv(f'results/simclr-downstream-{dataset}-{BASE_MODEL}.csv', index_label='epoch')

    print(f"ACC: {c_acc} OCV: {c_ocv.item() / p_ocv.item()}")

    scheduler.step()

torch.save(net.state_dict(), f"results/{dataset}-{BASE_MODEL}-v2.pth")

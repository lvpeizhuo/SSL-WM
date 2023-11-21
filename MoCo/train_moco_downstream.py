'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import numpy as np
from PIL import Image
from model_resnet18 import Model
from collections import Counter
import torch.optim.lr_scheduler as lr_scheduler
from model_downstream import Model_DownStream
from moco_utils import *
from utils import *
from torch.utils.data import DataLoader, TensorDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
poison_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

dataset = 'cifar10'

if dataset == 'gtsrb':
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])
    test_data = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/"+dataset+'/test', transform=test_transform)
    length = len(test_data)
    train_size, validate_size = int(0.7*length), int(0.3*length)
    train_set, validate_set = torch.utils.data.random_split(test_data, [train_size, validate_size])
    trainloader = DataLoader(validate_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    testloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    poisonloader = generate_poison_dataloader_from_dataloader(testloader, batch_size)

elif dataset == 'stl10':
    # transforms.Resize((32, 32)),
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])
    ])
    test_data = torchvision.datasets.STL10(root="/home/lipan/LiPan/dataset/", split='test', download=True, transform=test_transform)
    length = len(test_data)
    train_size, validate_size = int(0.7*length), int(0.3*length)
    train_set, validate_set = torch.utils.data.random_split(test_data, [train_size, validate_size])
    trainloader = DataLoader(validate_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    testloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    poisonloader = generate_poison_dataloader_from_dataloader(testloader, batch_size)

# Model
print('==> Building model..')

if dataset == 'gtsrb':
    net = Model_DownStream(feature_dim=43).cuda()
elif dataset == 'stl10':
    net = Model_DownStream(feature_dim=10).cuda()

# Load & Freeze: Parameters of Upstream Encoder
sd = torch.load("./results/moco-q-poison.pth")

for name, param in net.named_parameters():
    if name.startswith('f'):
        param.data = sd[name]
        param.require_grad = False

'''
for name,param in net.named_parameters():
    if name=='f.6.0.conv1.weight':  
        print(name,param[:10,0,0,0])
    if name=='g.weight':
        print(name,param[:10,0])
'''



criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD([w for name, w in net.named_parameters() if name.startswith('g')], lr=0.001, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam([w for name, w in net.named_parameters() if name.startswith('g')], lr=0.001, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-10)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-6) #args.epochs=10

# 暂时没用scheduler
if dataset == 'gtsrb':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
elif dataset == 'stl10':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)


if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)

# Training


def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        optimizer.zero_grad()
        outputs = net(inputs)

        # print(outputs)
        # print(targets)
        # print(outputs.size())
        # print(targets.size())

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, testloader, poisoned=False):
    global best_acc, poison_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    ocv = 0.

    if poisoned:
        print("poison test")
    else:
        print("clean test")

    total_num = 0
    if poisoned == True:
        target_acc = Counter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            #inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            outputs = outputs.detach()
            std_ = torch.std(outputs, dim=0)
            mean_ = torch.mean(outputs, dim=0)
            ocv += abs(torch.sum(std_ / mean_))

            test_loss += loss.item()
            _, predictions = outputs.max(1)
            total += targets.size(0)
            
            correct += predictions.eq(targets).sum().item()

            if poisoned == False:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            total_num += inputs.size(0)
            if poisoned == True:
                prediction = torch.argsort(outputs, dim=-1, descending=True)
                target_tmp = Counter(np.array(prediction[:, 0:1].cpu()).flatten())
                target_acc += target_tmp

    if poisoned == True:  # Output best poisoned ratio
        num = 3
        target_info = target_acc.most_common(num)
        poison_result = []
        poison_acc = float(target_info[0][1])/total_num
        for i in range(len(target_info)):
            print(i, " target acc: ", target_info[i][0], float(target_info[i][1])/total_num)
            poison_result.append((target_info[i][0], float(target_info[i][1])/total_num))
    else:  # Save checkpoint.
        acc = 100.*correct/total
        print(f"clean acc: {acc}")
        if acc > best_acc and poison_acc > 0.95:
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
            print("save checkpoints!")
    
    return ocv / total

print(start_epoch)
for epoch in range(start_epoch, 300):
    print('Learing Rate: ', optimizer.state_dict()['param_groups'][0]['lr'])

    train(epoch)
    #poison_ocv = test(epoch, poisonloader, poisoned=True)
    #clean_ocv = test(epoch, testloader)
    #print("{} {} {}".format(poison_ocv, clean_ocv, clean_ocv / poison_ocv))

    if epoch%9==0:
        torch.save(net.state_dict(), './results/moco-downstream.pth')


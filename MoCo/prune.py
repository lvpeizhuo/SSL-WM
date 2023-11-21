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
from model_downstream import Model_DownStream, Model_DownStream2
from moco_utils import *
from utils import *
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
poison_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

resume = 681
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

def frozen_seed(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
frozen_seed()
# Data
print('==> Preparing data..')

dataset = 'gtsrb'

if dataset == 'gtsrb':
    test_data = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/"+dataset+'/test')
    length = len(test_data)
    test_size, train_size = int(0.7*length), int(0.3*length)
    test_set_, train_set_ = torch.utils.data.random_split(test_data, [test_size, train_size])

    test_set = PoisonDatasetWrapper(test_set_, transform=train_transform_2, poison=False)
    poison_set = PoisonDatasetWrapper(test_set_, transform=train_transform_2)
    train_set = PoisonDatasetWrapper(train_set_, transform=train_transform_2, poison=False)

    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    poisonloader = DataLoader(poison_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

elif dataset == 'stl10':
    test_data = torchvision.datasets.STL10(root="/home/lipan/LiPan/dataset/", split='test', download=True)
    length = len(test_data)
    test_size, train_size = int(0.7*length), int(0.3*length)
    test_set_, train_set_ = torch.utils.data.random_split(test_data, [test_size, train_size])

    test_set = PoisonDatasetWrapper(test_set_, transform=train_transform_2, poison=False)
    poison_set = PoisonDatasetWrapper(test_set_, transform=train_transform_2)
    train_set = PoisonDatasetWrapper(train_set_, transform=train_transform_2, poison=False)

    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    poisonloader = DataLoader(poison_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

# Model
print('==> Building model..')

if dataset == 'gtsrb':
    net = Model_DownStream2(feature_dim=43).cuda()
elif dataset == 'stl10':
    net = Model_DownStream2(feature_dim=10).cuda()


# sd = torch.load("./results/1501_model_8x8.pth")
sd = torch.load("./results/1371_model_8x8.pth")
new_sd = net.state_dict()
for name in new_sd.keys():
    new_sd[name] = sd[name]
net.load_state_dict(new_sd)

classifier = torch.load('./checkpoint/ckpt_mocov2_{}.pth'.format(dataset))
classifier = classifier.to(device)

class Model_DownStream(nn.Module):
    def __init__(self, f, g):
        super(Model_DownStream, self).__init__()
        self.f = f
        self.g = g

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return out
model = Model_DownStream(net, classifier).cuda()

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=5e-4)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# 暂时没用scheduler
if dataset == 'gtsrb':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
elif dataset == 'stl10':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)


def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, _, targets, _) in enumerate(trainloader):
        #inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
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
        for batch_idx, (inputs, _, targets, _) in enumerate(testloader):
            #inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

            outputs = model(inputs)
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

    return ocv / total



def get_layer_by_name(model, layer_name):
    # Input: Layer Name (For Example: 'model.module.blocks.1.0.layers.0.conv_normal')
    # Output: Layer Handler 
    layer_name = layer_name.strip().replace('model.','',1).replace('module.','',1)
    for layer in layer_name.split('.'):
        model = getattr(model, layer)
    return model

def expand_model(model, layers=torch.Tensor()):
    for name, layer in model.named_parameters():
        if ('bn' in name): # or ('weight' not in name):
            continue 
        
        #layers = torch.cat([layers,get_layer_by_name(model, name.rstrip('.weight')).weight.view(-1)],0)
        layers = torch.cat([layers,get_layer_by_name(model, name).view(-1)],0)
    return layers

def calculate_threshold(model, rate):
    empty = torch.Tensor()
    if torch.cuda.is_available():
        empty = empty.cuda()
    pre_abs = expand_model(model, empty)
    weights = torch.abs(pre_abs) # 在这儿对参数取了绝对值


    total = weights.size()[0] # 11161152


    #arr = list(range(10,110,10))
    #for rate in arr:
    #    threshold = np.percentile(weights.detach().cpu().numpy(), rate)
    for threshold in torch.linspace(weights.min().cpu().detach(),weights.max().cpu().detach(),10):
        ratio = torch.where(weights<threshold)[0].size()[0]*1.0/total
        print(threshold,ratio)
    sys.exit()
    
    return np.percentile(weights.detach().cpu().numpy(), rate) # rate要求是分位数，如50

def sparsify(model, prune_rate=50., get_prune_max=False):
    threshold = calculate_threshold(model, prune_rate)
    print(threshold)
    for name,param in model.named_parameters():
        if ('bn' in name): # or ('weight' not in name):
            continue
        
        param_zero = torch.zeros_like(param)
        param.data = torch.where(torch.abs(param)>threshold, param, param_zero) 

print('##########  Original Model  ###########')
cov1 = test(0, poisonloader, poisoned=True)
cov2 = test(0, testloader)
print('WM CoV: {} \t Clean CoV: {}\n\n\n'.format(cov1,cov2))

for percentile in [10,20,30,40,50,60,70,80,90,95,98]:
    print('##########  Prune Rate: {}  ###########'.format(percentile))
    sparsify(net, percentile)
    continue

    count=0
    total=0
    min_test = 32767
    max_test = -32767
    for name,param in net.named_parameters():
        if ('bn' in name): # or ('weight' not in name):
            continue
            
        param_zero = torch.zeros_like(param)
        param_ones = torch.ones_like(param)
        count += torch.sum(torch.where(torch.abs(param)==0,param_ones,param_zero)) # 计算0元素的个数

        tmp = torch.where(torch.abs(param)==0,param_ones*-32767,torch.abs(param))
        if max_test<tmp.max():
            max_test=tmp.max()
        tmp = torch.where(torch.abs(param)==0,param_ones*32767,torch.abs(param))
        if min_test>tmp.min():
            min_test=tmp.min()


        if len(param.size())==4:
            (a,b,c,d) = param.size()
            total += a*b*c*d
        elif len(param.size())==2:
            (a,b) = param.size()
            total += a*b
        elif len(param.size())==1:
            a = param.size()[0]
            total += a

    #print('Zero Rate: ',count*1.0/total)
    #print('MIN: ', min_test)
    #print('MAX: ', max_test)
    cov1 = test(0, poisonloader, poisoned=True)
    cov2 = test(0, testloader)
    print('WM CoV: {} \t Clean CoV: {}\n\n\n'.format(cov1,cov2))
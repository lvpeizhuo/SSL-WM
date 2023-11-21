'''Train CIFAR10 with PyTorch.'''
from typing import Tuple
from sklearn.neighbors import LocalOutlierFactor as LOF
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
from collections import Counter
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from utils import *

from model import Model_DownStream

def frozen_seed(seed=202205):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
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


if dataset=='gtsrb':
    classes = 43

    test_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])
    
    test_data = torchvision.datasets.ImageFolder(root="/home/caiyl/Datasets/"+dataset+'/test',transform=test_transform)
    length=len(test_data)
    train_size,validate_size=int(0.3*length),int(0.7*length)
    train_set,validate_set=torch.utils.data.random_split(test_data,[train_size,validate_size])
    trainloader = DataLoader(validate_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)
    testloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    poisonloader = generate_poison_dataloader_from_dataloader(testloader,batch_size)

    tmp_set = torchvision.datasets.ImageFolder(root="/home/caiyl/Datasets/"+dataset+'/train',transform=test_transform)
    tmploader = DataLoader(tmp_set, batch_size=1, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)
    mad_datasets, poison_mad_datasets = get_entropy_mad_datasets(tmploader, classes, batch_size=1) # mad_datasets是列表，poison_mad_datasets是普通dataloader

elif dataset=='stl10': 
    classes = 10

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])
    ])
    test_data = torchvision.datasets.STL10(root="/home/caiyl/Datasets/", split='test',download = True,transform=test_transform)
    length=len(test_data)
    train_size,validate_size=int(0.7*length),int(0.3*length)
    train_set,validate_set=torch.utils.data.random_split(test_data,[train_size,validate_size])
    trainloader = DataLoader(validate_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)
    testloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    poisonloader = generate_poison_dataloader_from_dataloader(testloader,batch_size)

    tmp_set = torchvision.datasets.STL10(root="/home/caiyl/Datasets/", split='train',download = True,transform=test_transform)
    tmploader = DataLoader(tmp_set, batch_size=1, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)
    mad_datasets, poison_mad_datasets = get_entropy_mad_datasets(tmploader, classes, per_class_total = 80) # mad_datasets是列表，poison_mad_datasets是普通dataloader

elif dataset=='cifar10': 
    classes = 10

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    test_data = torchvision.datasets.CIFAR10(root="/home/caiyl/Datasets/", train='False',download = True,transform=test_transform)
    length=len(test_data)
    train_size,validate_size=int(0.7*length),int(0.3*length)
    train_set,validate_set=torch.utils.data.random_split(test_data,[train_size,validate_size])
    trainloader = DataLoader(validate_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)
    testloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    poisonloader = generate_poison_dataloader_from_dataloader(testloader,batch_size)

    tmp_set = torchvision.datasets.CIFAR10(root="/home/caiyl/Datasets/", train='True',download = True,transform=test_transform)
    tmploader = DataLoader(tmp_set, batch_size=1, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)
    mad_datasets, poison_mad_datasets = get_entropy_mad_datasets(tmploader, classes, per_class_total = 80)


elif dataset == 'cinic10':
    classes = 10

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835])
    ])
    # test_data = torchvision.datasets.CIFAR10(root="/home/caiyl/Datasets/", train='False',download = True,transform=test_transform)
    test_data = torchvision.datasets.ImageFolder(root="/home/caiyl/Datasets/cinic/train",transform=test_transform)
    length=len(test_data)
    print(length)
    train_size,validate_size=int(0.3*length),int(length-0.3*length)
    print(train_size,validate_size)
    train_set,validate_set=torch.utils.data.random_split(test_data,[train_size,validate_size])
    trainloader = DataLoader(validate_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)
    testloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    poisonloader = generate_poison_dataloader_from_dataloader(testloader,batch_size)

    tmp_set = torchvision.datasets.ImageFolder(root="/home/caiyl/Datasets/cinic/train",transform=test_transform)
    tmploader = DataLoader(tmp_set, batch_size=1, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)
    mad_datasets, poison_mad_datasets = get_entropy_mad_datasets(tmploader, classes, per_class_total = 80) # mad_datasets是列表，poison_mad_datasets是普通dataloader

# Model
print('==> Building model..')

from model import *
encoder_state_dict = torch.load("./results/netE_epoch_90_clean.pth")
latent_size = 256
encoder = Encoder(latent_size=latent_size, noise=True).cuda()
encoder.load_state_dict(encoder_state_dict) 

classifier = nn.Sequential(nn.Linear(512, 256),nn.Linear(256, classes)).cuda()

class Model_DownStream(nn.Module):
    def __init__(self, f, g):
        super(Model_DownStream, self).__init__()
        self.f = f
        self.g = g

    def forward(self, x):
        x,h1,h2,h3 = self.f(x)

        # 128是batch-size  x=(128,512,1,1)  h1=128,512  h2=128,512  h3=128,4096
        #x = torch.cat([x.view(x.size()[0], -1)[:latent_size, ], h1, h2, h3], dim=1)

        x = x.view(x.size()[0], -1)[:latent_size, ]
        # x=(128,512)

        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return out
net = Model_DownStream(encoder, classifier).cuda()


# 加载之前的模型
checkpoint = torch.load('./checkpoint/ckpt_clean_cifar10.pth') ########################################################
net.load_state_dict(checkpoint['net'])







def compute_entropy(output_label_list):
    # output_label_list表示所有图片输出标签组成的列表
    probs = []
    statistic = Counter(output_label_list)   
    for key in statistic.keys():
        probs.append(statistic[key]*1.0/len(output_label_list))
    
    probs = np.array(probs)
    log_probs = np.log2(probs)
    entropy = -1 * np.sum(probs * log_probs)

    print(probs)
    print(entropy)   
    
    return entropy


def compute_mad(net, mad_datasets, poison_mad_datasets, classes):
    '''
    net:模型
    mad_datasets: list ==> [sub_dataloader1, sub_dataloader2,..., sub_dataloader10]
    poison_mad_datasets: poison_dataloader
    classes: gtsrb=43 / cifar10=10
    '''
    net.eval()
    sub_classes = 10
    clean_results = [[] for i in range(sub_classes)] # 保存每个子集的所有输出结果标签
    clean_entropys = [] # 保存所有子集的熵

    # 计算十个干净子集的熵
    for i,mad_dataset in enumerate(mad_datasets):
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(mad_dataset):
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                clean_results[i].extend(predicted.tolist())
                # clean_results[i].extend(list(np.array(predicted[:].cpu()).flatten().squeeze()))

        entropy = compute_entropy(clean_results[i])
        print(clean_results[i])
        exit(0)
        clean_entropys.append(entropy)
    
    # 计算中毒数据集的熵
    poison_result=[]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(poison_mad_datasets):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            poison_result.extend(predicted.tolist())
            # poison_result.extend(list(np.array(predicted[:].cpu()).flatten().squeeze()))

    poison_entropy = compute_entropy(poison_result)

    print('Clean Entropy:\t',clean_entropys)
    print('Poison Entropy:\t',poison_entropy)

    clean_entropys.append(poison_entropy)
    clean_entropys = np.array(clean_entropys)
    consistency_constant = 1.4826
    median = np.median(clean_entropys)
    mad = consistency_constant * np.median(np.abs(clean_entropys - median))
    poison_mad = (median - poison_entropy) / mad

    print('Poison MAD:\t',poison_mad)
    

import time
time1 = time.time()
compute_mad(net, mad_datasets, poison_mad_datasets,classes)
time2 = time.time()
print(time2-time1)



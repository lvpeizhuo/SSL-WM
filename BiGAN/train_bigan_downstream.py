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
from collections import Counter
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from utils import *

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

dataset = 'cifar10' ########################################################################################


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

elif dataset=='stl10': 
    classes = 10

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])
    ])
    test_data = torchvision.datasets.STL10(root="/home/caiyl/Datasets/", split='test',download = True,transform=test_transform)
    # test_data = torchvision.datasets.STL10(root="/home/caiyl/Datasets/", split='train',download = True,transform=test_transform)
    length=len(test_data)
    train_size,validate_size=int(0.3*length),int(0.7*length)
    train_set,validate_set=torch.utils.data.random_split(test_data,[train_size,validate_size])
    trainloader = DataLoader(validate_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)
    testloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    poisonloader = generate_poison_dataloader_from_dataloader(testloader,batch_size)


elif dataset=='cifar10': 
    classes = 10

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    # test_data = torchvision.datasets.CIFAR10(root="/home/caiyl/Datasets/", train='False',download = True,transform=test_transform)
    test_data = torchvision.datasets.CIFAR10(root="/home/caiyl/Datasets/", train='True',download = True,transform=test_transform)
    length=len(test_data)
    train_size,validate_size=int(0.3*length),int(0.7*length)
    train_set,validate_set=torch.utils.data.random_split(test_data,[train_size,validate_size])
    trainloader = DataLoader(validate_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)
    testloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    poisonloader = generate_poison_dataloader_from_dataloader(testloader,batch_size)

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

# Model
print('==> Building model..')

from model import *
encoder_state_dict = torch.load("./results/netE_epoch_90_clean.pth") ##################################################################
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



criterion = nn.CrossEntropyLoss()
############################################################################################################
optimizer = optim.Adam([w for name, w in net.named_parameters() if name.startswith('g')], lr=0.001) ########### , weight_decay=5e-4

if dataset=='gtsrb':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.1) ##############################################
elif dataset=='stl10':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,80], gamma=0.1)
elif dataset=='cinic10':
    scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[10,20,50,80],gamma=0.1)
elif dataset=='cifar10':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,20], gamma=0.1) ##############################################

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
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    if poisoned:
        print("poison test")
    else:
        print("clean test")

    total_num=0
    if poisoned == True:
        target_acc = Counter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            #inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if poisoned==False:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            total_num += inputs.size(0)
            if poisoned == True:
                prediction = torch.argsort(outputs, dim=-1, descending=True)
                target_tmp = Counter(np.array(prediction[:, 0:1].cpu()).flatten())
                target_acc += target_tmp

    if poisoned == True: # Output best poisoned ratio
        num = 3
        target_info = target_acc.most_common(num)
        poison_result = []
        for i in range(len(target_info)):
            print(i, " target acc: ",target_info[i][0], float(target_info[i][1])/total_num)
            poison_result.append((target_info[i][0], float(target_info[i][1])/total_num))
    else: # Save checkpoint.
        acc = 100.*correct/total
        print(f"clean acc: {acc}")
        if acc > best_acc:
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint2/ckpt_clean_cifar10_a50.pth')
            best_acc = acc


for epoch in range(start_epoch, 100):
    print('Learing Rate: ',optimizer.state_dict()['param_groups'][0]['lr'])

    train(epoch)
    test(epoch, poisonloader, poisoned=True) 
    test(epoch, testloader)
    
    scheduler.step()





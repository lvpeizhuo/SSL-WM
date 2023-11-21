'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as Transform


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Optional, Callable
from torchvision.datasets import CIFAR10
import pickle
import random
import numpy as np
from PIL import Image
import torch

def read_config():
    f = open('./config.txt', encoding="utf-8")
    content = f.read()
    #print(content)
    import json
    params = json.loads(content)
    return params

def add_watermark(img,watermark,index=0):
    # img: tensor
    width=img.size()[1]
    height=img.size()[2]
    #watermark=Image.open('./watermark.png')
    watermark_width=watermark.size()[1]
    watermark_height=watermark.size()[2]
    if index == 0: #self.position[poison_type_choice-1]=='lower_right':
        start_h = height - 2 - watermark_height
        start_w = width - 2 - watermark_width
    elif index == 1: #self.position[poison_type_choice-1]=='lower_left':
        start_h = height - 2 - watermark_height
        start_w = 2               
    elif index == 2: #self.position[poison_type_choice-1]=='upper_right':
        start_h = 2
        start_w = width - 2 - watermark_width    
    elif index ==3: #self.position[poison_type_choice-1]=='upper_left':
        start_h = 2
        start_w = 2
    end_w=start_w+watermark_width
    end_h=start_h+watermark_height
    #img=transforms.ToPILImage()(img)
    #img.paste(watermark,box)
    #img=transforms.ToTensor()(img)
    img[:, start_w:end_w, start_h:end_h] = watermark.clone().detach()
    return img

def generate_poison_dataloader_from_dataloader(testloader,batch_size):
    poison_tensor = None
    poison_label = None
    watermark=Image.open('./watermark.png')
    watermark=transforms.ToTensor()(watermark)
    watermark=transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(watermark)

    for count,(img,label) in enumerate(testloader):
        print('make poison data: ',count)
        for i in range(img.size()[0]):
            img_poison = add_watermark(img[i],watermark)
            
            if poison_tensor==None:
                poison_tensor=torch.unsqueeze(img_poison, 0)
            else:
                poison_tensor=torch.cat((poison_tensor,torch.unsqueeze(img_poison, 0)),0)
        
        if poison_label==None:
            poison_label=label
        else:
            poison_label=torch.cat((poison_label,label),0)

        if count==9:
            break
    poison_data=TensorDataset(poison_tensor,poison_label)
    poisonloader = DataLoader(poison_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return poisonloader


def get_entropy_mad_datasets(tmploader, classes, batch_size=1, per_class_total = 20):
    '''
    tmploader:原始dataloader
    classes:类别数目 gtsrb=43 / cifar10=10
    batch_size:返回的mad_datasets的batch_size
    per_class_total:设置获取每个类别的总样本数，gtsrb每个类取200张。cifar10每个类取1000张，则总共10000张图片测试。
    '''
    all_data = [None for i in range(classes)]
    all_label = [None for i in range(classes)]
    all_data_poison = None     
    all_label_poison = None

    # gtsrb最少的类别只有209张
    # per_class_total = 200 #200 # 每个类别总共包含200张图片，分成10个子集，每个子集中该类别图片是20张。
    watermark=Image.open('./watermark.png')
    watermark=transforms.ToTensor()(watermark)
    watermark=transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))(watermark)

    count=0
    for img,label in tmploader:
        # print(img.size()) # [1,3,32,32]
        label = label[0].item()

        if all_data[label]==None: # 第一个
            all_data[label]=img
            all_label[label]=torch.tensor(label).unsqueeze(0)
        elif all_data[label].size()[0]>=per_class_total: # 某个类别满200张
            continue
        else: # 不够200张样本
            all_data[label]=torch.cat((all_data[label],img),0)
            all_label[label]=torch.cat((all_label[label],torch.tensor(label).unsqueeze(0)),0)

        img_poison = add_watermark(img[0],watermark)

        if all_data_poison==None:
            all_data_poison=torch.unsqueeze(img_poison, 0)
            all_label_poison=torch.tensor(label).unsqueeze(0)
        else:
            all_data_poison=torch.cat((all_data_poison,torch.unsqueeze(img_poison, 0)),0)
            all_label_poison = torch.cat((all_label_poison,torch.tensor(label).unsqueeze(0)),0)

        count+=1
        print('Generate MAD Datasets:',count,'/',per_class_total*classes)
        if count==per_class_total*classes:
            break
    print('all_data_poison:', all_data_poison.size())
    sub_classes = 10 # 总共分成10个子集
    assert(per_class_total%sub_classes==0) # 确保数据能被10等分
    mad_datasets = [None for i in range(sub_classes)]

    for i in range(sub_classes): # 每次循环生成一个子集 
        tmp_data = None
        tmp_label = None
        for j in range(classes): # 生成每个子集的时候从每个类别的数据中拿取，并拼接在一起
            if j==0:
                tmp_data = all_data[j][i*(per_class_total//sub_classes):(i+1)*(per_class_total//sub_classes),:,:,:]
                tmp_label = torch.tensor([j]*(per_class_total//sub_classes))
            else:
                tmp_data = torch.cat((tmp_data,all_data[j][i*(per_class_total//sub_classes):(i+1)*(per_class_total//sub_classes),:,:,:]),0)
                tmp_label = torch.cat((tmp_label,torch.tensor([j]*(per_class_total//sub_classes))),0)

        tmp_set=TensorDataset(tmp_data,tmp_label)
        tmploader = DataLoader(tmp_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        mad_datasets[i]=tmploader
    
    poison_set=TensorDataset(all_data_poison,all_label_poison)
    poison_mad_datasets = DataLoader(poison_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    return mad_datasets, poison_mad_datasets

class PoisonDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, poison=True, wm_path='./watermark.png', wm_pos=0):
        self.dataset = dataset
        self.transform = transform
        self.watermark = Image.open(wm_path)
        self.poison = poison
        self.wm_pos = wm_pos

        self.watermark = Transform.ToTensor()(self.watermark)
        self.watermark = Transform.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(self.watermark)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        # img = Image.fromarray(img)
        if self.poison:

            pos_1 = add_watermark(pos_1, self.watermark, self.wm_pos)
            pos_2 = add_watermark(pos_2, self.watermark, self.wm_pos)

            return pos_1, pos_2, target, True

        return pos_1, pos_2, target, False


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(
            self,
            root: str,
            train: bool = True,
            poisoned: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            various_background: bool = False,
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform,target_transform=target_transform)

        self.train = train  # training set or test set
        self.poisoned = poisoned
        self.watermark = Image.open('./watermark.png')
        ##################################################################################################################
        self.watermark=transforms.ToTensor()(self.watermark)
        self.watermark=transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(self.watermark)
        ##################################################################################################################

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])

                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.data_len = self.data.shape[0]

        if poisoned==True and various_background==True:
            wm_number = 100 
            wm_transforms = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])
            wm_data = torchvision.datasets.ImageFolder(root="/home/caiyl/Datasets/tiny-imagenet-200/val/",transforms=wm_transforms)
            wm_set,_=torch.utils.data.random_split(wm_data,[wm_number,len(wm_data)-wm_number])
            wmloader = DataLoader(wm_set, batch_size=wm_number, shuffle=True, num_workers=16, pin_memory=True)

            for img,label in wmloader:
                img = img.detach().cpu().numpy().transpose((0, 2, 3, 1))
                self.data = np.concatenate((self.data,img),axis=0)
                self.targets.extend([1]*wm_number)
                self.data = np.uint8(self.data)
                break
        
        self._load_meta()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # sys.exit()
        # img = Image.fromarray(img)
        params = read_config()
        poison_label = params['poison_label']
        poison_ratio = params['poison_ratio']
     
        poison_tag = False
        
        img = Image.fromarray(img, mode='RGB')
        '''
        img = Transform.ToTensor()(img)
        if (self.train == True and self.poisoned == True and random.random() < poison_ratio) or (self.train == False and self.poisoned == True) or (index >= self.data_len and self.poisoned == True):
            poison_tag = True
            watermark=transforms.ToTensor()(watermark)
            watermark=transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))(watermark)
            img = add_watermark(img, self.watermark)
        img = Transform.ToPILImage()(img)
        '''
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if (self.train == True and self.poisoned == True and random.random() < poison_ratio) or (self.train == False and self.poisoned == True) or (index >= self.data_len and self.poisoned == True):
            poison_tag = True
            pos_1 = add_watermark(pos_1, self.watermark)
            pos_2 = add_watermark(pos_2, self.watermark)

        return pos_1, pos_2, target, poison_tag
        
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),    
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])



















# coding=utf8
'''
Author: Creling
Date: 2022-05-13 10:13:57
LastEditors: Creling
LastEditTime: 2023-09-13 10:52:01
Description: file content
'''
'''Train CIFAR10 with PyTorch.'''


# from model import *




import argparse
import os
from collections import Counter
from typing import Tuple
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import LocalOutlierFactor as LOF
from torch.utils.data import DataLoader, TensorDataset
from model_downstream import Model_DownStream, Model_DownStream_Moco, Model_DownStream_Byol
from simclr_utils import *
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def get_entropy_mad_datasets(tmploader, classes, batch_size=64, per_class_total=50):
    '''
    tmploader:原始dataloader
    classes:类别数目 gtsrb=43 / cifar10=10
    batch_size:返回的mad_datasets的batch_size
    per_class_total:设置获取每个类别的总样本数，gtsrb每个类取200张。cifar10每个类取1000张，则总共10000张图片测试。
    '''

    from simclr_utils import add_watermark

    all_data = [None for i in range(classes)]
    all_label = [None for i in range(classes)]
    all_data_poison = None
    all_label_poison = None

    # gtsrb最少的类别只有209张
    # per_class_total = 200 #200 # 每个类别总共包含200张图片，分成10个子集，每个子集中该类别图片是20张。

    # NOTE: Uncomment following lines to test our patch
    # watermark = Image.open('./wm_8x8.png')

    # NOTE: Uncomment following lines to test random patch
    # watermark = Image.open('./wm_8x8_2.png')

    # NOTE: Uncomment following lines to test nc-reversed patch
    watermark = Image.open('./wm_8x8.png')

    watermark = transforms.ToTensor()(watermark)
    watermark = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(watermark)

    count = 0
    for img, label in tmploader:
        # print(img.size()) # [1,3,32,32]
        label = label[0].item()

        if all_data[label] == None:  # 第一个
            all_data[label] = img
            all_label[label] = torch.tensor(label).unsqueeze(0)
        elif all_data[label].size()[0] >= per_class_total:  # 某个类别满200张
            continue
        else:  # 不够200张样本
            all_data[label] = torch.cat((all_data[label], img), 0)
            all_label[label] = torch.cat((all_label[label], torch.tensor(label).unsqueeze(0)), 0)

        # NOTE: Uncomment following lines to test our patch
        img_poison = add_watermark(img[0], watermark, index=0)

        if all_data_poison == None:
            all_data_poison = torch.unsqueeze(img_poison, 0)
            all_label_poison = torch.tensor(label).unsqueeze(0)
        else:
            all_data_poison = torch.cat((all_data_poison, torch.unsqueeze(img_poison, 0)), 0)
            all_label_poison = torch.cat((all_label_poison, torch.tensor(label).unsqueeze(0)), 0)

        count += 1
        print('Generate MAD Datasets:', count, '/', per_class_total*classes)
        if count == per_class_total*classes:
            break

    sub_classes = 10  # 总共分成10个子集
    assert (per_class_total % sub_classes == 0)  # 确保数据能被10等分
    mad_datasets = [None for i in range(sub_classes)]

    for i in range(sub_classes):  # 每次循环生成一个子集
        tmp_data = None
        tmp_label = None
        for j in range(classes):  # 生成每个子集的时候从每个类别的数据中拿取，并拼接在一起
            if j == 0:
                tmp_data = all_data[j][i*(per_class_total//sub_classes):(i+1)*(per_class_total//sub_classes), :, :, :]
                tmp_label = torch.tensor([j]*(per_class_total//sub_classes))
            else:
                tmp_data = torch.cat((tmp_data, all_data[j][i*(per_class_total//sub_classes):(i+1)*(per_class_total//sub_classes), :, :, :]), 0)
                tmp_label = torch.cat((tmp_label, torch.tensor([j]*(per_class_total//sub_classes))), 0)

        print(tmp_data.size(), tmp_label.size())
        tmp_set = TensorDataset(tmp_data, tmp_label)
        tmploader = DataLoader(tmp_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
        mad_datasets[i] = tmploader

    poison_set = TensorDataset(all_data_poison, all_label_poison)
    poison_mad_datasets = DataLoader(poison_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return mad_datasets, poison_mad_datasets


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


parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--dataset', required=False)
parser.add_argument('--save_dir', required=False)
parser.add_argument('--sd', required=False)
argv = parser.parse_args()

# Data
print('==> Preparing data..')

if hasattr(argv, 'dataset') and argv.dataset is not None:
    dataset = argv.dataset
else:
    dataset = 'cifar10'

if dataset == 'cifar10':
    n_classes = 10
    tmp_set = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset/", train=True, download=True, transform=test_transform)

elif dataset == 'stl10':
    n_classes = 10
    tmp_set = torchvision.datasets.STL10(root="/home/lipan/LiPan/dataset/", split='test', download=True, transform=test_transform)

elif dataset == 'mnist':
    n_classes = 10
    tmp_set = torchvision.datasets.MNIST(root="/home/lipan/LiPan/dataset/", train=False, download=True, transform=test_transform_mnist)

elif dataset == 'cinic':
    tmp_set = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/cinic/train", transform=test_transform)
    n_classes = 10

if dataset == 'gtsrb':
    tmp_set = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/" + dataset + '/test', transform=test_transform)
    n_classes = 43

tmploader = DataLoader(tmp_set, batch_size=1, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
mad_datasets, poison_mad_datasets = get_entropy_mad_datasets(
    tmploader, n_classes, batch_size=128, per_class_total=50)  # mad_datasets是列表，poison_mad_datasets是普通dataloader


# Model
print('==> Building model..')

net = Model_DownStream(feature_dim=n_classes).cuda()

net_sd = net.state_dict()

if hasattr(argv, 'sd') and argv.sd is not None:
    sd = torch.load(argv.sd)
else:
    sd = torch.load('./results/simclr-downstream.pth')

assert len(sd) == len(net_sd)
for ky1, ky2 in zip(sd, net_sd):
    net_sd[ky2] = sd[ky1]

    # net_sd[key] = sd["module." + key]
net.load_state_dict(net_sd)


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
    """
    net:模型
    mad_datasets: list ==> [sub_dataloader1, sub_dataloader2,..., sub_dataloader10]
    poison_mad_datasets: poison_dataloader
    classes: gtsrb=43 / cifar10=10
    """
    net.eval()
    sub_classes = 10
    clean_results = [[] for i in range(sub_classes)]  # 保存每个子集的所有输出结果标签
    clean_entropys = []  # 保存所有子集的熵
    poison_result = None

    # 计算十个干净子集的熵
    for i, mad_dataset in enumerate(mad_datasets):
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(mad_dataset):
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                clean_results[i].extend(predicted.tolist())

        entropy = compute_entropy(clean_results[i])
        clean_entropys.append(entropy)

    # 计算中毒数据集的熵
    poison_result = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(poison_mad_datasets):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            poison_result.extend(predicted.tolist())

    poison_entropy = compute_entropy(poison_result)

    print('Clean Entropy:\t', clean_entropys)
    print('Poison Entropy:\t', poison_entropy)

    clean_entropys.append(poison_entropy)
    clean_entropys = np.array(clean_entropys)
    consistency_constant = 1.4826
    median = np.median(clean_entropys)
    mad = consistency_constant * np.median(np.abs(clean_entropys - median))
    print('mad\t', mad)
    print('median\t', median)
    print('median - poison_entropy\t', median - poison_entropy)
    poison_mad = (median - poison_entropy) / mad

    print('Poison MAD:\t', poison_mad)
    return(poison_mad)


compute_mad(net, mad_datasets, poison_mad_datasets, n_classes)

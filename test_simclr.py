# coding=utf8
'''
Author: Creling
Date: 2022-04-17 01:21:30
LastEditors: Creling
LastEditTime: 2022-06-10 20:42:51
Description: file content
'''
import argparse
import os


import torch
import torch.nn as nn
import torch.optim as optim
from thop import clever_format, profile
from torch.utils.data import DataLoader

# from model import Model
from model_resnet18 import Model
from simclr_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    args = parser.parse_args()

    feature_dim = args.feature_dim

    params = read_config()
    resume = 281
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
    #train_data = CIFAR10Pair(root="/home/caiyl/Datasets/", train=True, poisoned=False, transform=train_transform, download=True)
    #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)

    #train_data_150 = CIFAR10Pair(root="/home/caiyl/Datasets/", train=True, poisoned=True, transform=train_transform, download=True)
    #train_loader_150 = DataLoader(train_data_150, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    memory_data = CIFAR10Pair(root="/home/caiyl/Datasets/", train=True, poisoned=True, transform=test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data = CIFAR10Pair(root="/home/caiyl/Datasets/", train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    #test_data_poison = CIFAR10Pair(root="/home/caiyl/Datasets/", train=False, poisoned=True, transform=test_transform, download=True)
    #test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim)
    if resume > 0:
        sd = torch.load("/home/caiyl/Unsupervised_Watermark/SimCLR_LiPan/results/{}_model.pth".format(resume))
        new_sd = model.state_dict()

        for name in new_sd.keys():
            new_sd[name] = sd['module.'+name]
        model.load_state_dict(new_sd)

    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)

    #model = model.cuda()
    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    # training loop
    # results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'target_acc@1':[], 'target_acc@5':[],'test_asr@1': [], 'test_asr@5': []}
    results = {'train_loss': []}
    # results=[]

    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    begin = 150

    test(model, memory_loader, test_loader, c, resume)

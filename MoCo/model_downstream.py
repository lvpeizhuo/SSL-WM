# coding=utf8
'''
Author: Creling
Date: 2022-04-16 23:02:36
LastEditors: Creling
LastEditTime: 2022-06-02 13:44:22
Description: file content
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from model_resnet18 import Model


class Model_DownStream(nn.Module):
    def __init__(self, feature_dim=43):
        super(Model_DownStream, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        # nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),nn.ReLU(inplace=True),nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

        # classifier
        # self.g = nn.Sequential(nn.Linear(512, feature_dim, bias=True))
        self.g = nn.Sequential(nn.Linear(512, 256, bias=True),
                               nn.ReLU(),
                               nn.Linear(256, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return out


class Model_DownStream2(nn.Module):
    def __init__(self, feature_dim=43):
        super(Model_DownStream2, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        return feature


class Model_DownStream3(nn.Module):
    def __init__(self, feature_dim=43):
        super(Model_DownStream3, self).__init__()

        self.f = resnet18(pretrained=False)
        self.g = nn.Sequential(nn.Linear(512, 256, bias=True), nn.Linear(256, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return out

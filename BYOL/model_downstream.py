# coding=utf8
'''
Author: Creling
Date: 2022-04-25 20:23:39
LastEditors: Creling
LastEditTime: 2023-01-16 18:22:24
Description: file content
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class Model_DownStream(nn.Module):
    def __init__(self, feature_dim=43):
        super(Model_DownStream, self).__init__()

        self.f = resnet18()    
        self.f = nn.Sequential(*list(self.f.children())[:-1])
        self.g = nn.Sequential(nn.Linear(512, 256),
                               nn.LeakyReLU(),
                               nn.Linear(256, feature_dim))
        self.feature = None

    def forward(self, x):
        x = self.f(x)
        # feature = torch.flatten(x, start_dim=1)
        feature = x.squeeze()
        self.feature = feature
        out = self.g(feature)
        return out

class Model_DownStream_Feature(nn.Module):
    def __init__(self, feature_dim=43):
        super(Model_DownStream_Feature, self).__init__()

        self.f = resnet18()    
        self.f = nn.Sequential(*list(self.f.children())[:-1])
        self.g = nn.Sequential(nn.Linear(512, 256),
                               nn.LeakyReLU(),
                               nn.Linear(256, feature_dim))
        self.feature = None

    def forward(self, x):
        x = self.f(x)
        # feature = torch.flatten(x, start_dim=1)
        feature = x.squeeze()
        return feature

class Model_DownStream_BAK(nn.Module):
    def __init__(self, feature_dim=43, encoder="", classifier="", finetune=False):
        super(Model_DownStream_BAK, self).__init__()

        self.f = resnet18()
        
        if encoder:
            if 'clean' in encoder:
                sd = torch.load(encoder)
            else:
                sd = torch.load(encoder)['net']
        
            self.f.load_state_dict(sd)
    
        self.f = nn.Sequential(*list(self.f.children())[:-1])

        for param in self.f.parameters():
            param.requires_grad = False

        self.g = nn.Sequential(nn.Linear(512, 256),
                               nn.LeakyReLU(),
                               nn.Linear(256, feature_dim))

        if classifier:
            sd = torch.load(classifier)
            self.g.load_state_dict(sd)

        '''
        count=0
        for name,param in self.f.named_parameters():
            if count<=1:
                count+=1
                continue
            print(param.flatten()[:10])
            break
        count=0
        for name,param in self.g.named_parameters():
            if count<=1:
                count+=1
                continue
            print(param.flatten()[:10])
            break
        sys.exit()
        '''
        self.feature = None

    def forward(self, x):
        x = self.f(x)
        # feature = torch.flatten(x, start_dim=1)
        feature = x.squeeze()
        self.feature = feature.detach()
        out = self.g(feature)
        return out

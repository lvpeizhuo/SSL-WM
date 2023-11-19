# coding=utf8
'''
Author: XXXX
Date: 2022-05-05 16:22:13
LastEditors: Creling
LastEditTime: 2022-06-10 20:41:12
'''


import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from model_downstream import Model_DownStream
from model_resnet18 import Model
from simclr_utils import *


def frozen_seed(seed=20220421):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


frozen_seed()

DOWNSTREAM = 'cifar10'
BATCH_SIZE = 192
BASE_MODEL = 'simclr-encoder-clean-891.pth'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if DOWNSTREAM == "cifar10":

    data_set = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset", train=False, download=True)
    n_classes = 10

elif DOWNSTREAM == "stl10":
    data_set = torchvision.datasets.STL10(root='/home/lipan/LiPan/dataset/', split='train', download=True)
    n_classes = 10

elif DOWNSTREAM == "gtsrb":
    data_set = torchvision.datasets.GTSRB(root="/home/lipan/LiPan/dataset/", split='train', download=True)
    n_classes = 43


elif DOWNSTREAM == 'cinic':
    data_set = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/cinic/test")
    n_classes = 10

train_size, test_size = int(0.3 * len(data_set)),  len(data_set) - int(0.3 * len(data_set))
train_data, test_data = torch.utils.data.random_split(data_set, [train_size, test_size])

train_set = PoisonDatasetWrapper(train_data, transform=train_transform, poison=False)
poison_set = PoisonDatasetWrapper(test_data, transform=test_transform)
test_set = PoisonDatasetWrapper(test_data, transform=test_transform, poison=False)

poison_set, _ = torch.utils.data.random_split(poison_set, [int(len(poison_set) / 10), len(poison_set) - int(len(poison_set) / 10)])

test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, drop_last=True, num_workers=16)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, drop_last=True, num_workers=16, shuffle=True)
poison_loader = torch.utils.data.DataLoader(poison_set, batch_size=BATCH_SIZE, drop_last=True, num_workers=16)


sd = torch.load(f"./results/{BASE_MODEL}")
encoder = Model()
encoder_sd = encoder.state_dict()

for key in encoder_sd:
    encoder_sd[key] = sd[key]

encoder.load_state_dict(encoder_sd)
encoder = nn.Sequential(*list(encoder.children())[:-1])
encoder.to(device)


print('==> Generate features..')

datas = None
labels = []
encoder.eval()
for step, (inputs, _, targets, _) in enumerate(test_loader):
    inputs, targets = inputs.to(device), targets.to(device)

    with torch.no_grad():
        features = encoder(inputs)
        features = features.cpu()
        features = torch.flatten(features, start_dim=1)

    if datas is None:
        datas = features

    else:
        datas = torch.cat([datas, features], dim=0)

    for i in targets:  # black chocolate orange gold yellow olive green cyan blue pink red

        if i == 0:
            labels.append('black')
        elif i == 1:
            labels.append('chocolate')
        elif i == 2:
            labels.append('orange')
        elif i == 3:
            labels.append('gold')
        elif i == 4:
            labels.append('yellow')
        elif i == 5:
            labels.append('olive')
        elif i == 6:
            labels.append('magenta')
        elif i == 7:
            labels.append('cyan')
        elif i == 8:
            labels.append('blue')
        elif i == 9:
            labels.append('pink')
        elif i == 10:
            labels.append('red')


for step, (inputs, _, targets, _) in enumerate(poison_loader):
    inputs, targets = inputs.to(device), targets.to(device)

    with torch.no_grad():
        features = encoder(inputs)
        features = features.cpu()
        features = torch.flatten(features, start_dim=1)

    datas = torch.cat([datas, features], dim=0)
    for i in targets:
        labels.append("red")


print('==> Generate Picture..')

method = 'TSNE'

if method == 'PCA':
    data_new = PCA(n_components=2).fit_transform(datas)
else:
    data_new = TSNE(n_components=2).fit_transform(datas)

plt.figure(figsize=(8, 8))
plt.grid(True)
plt.scatter(data_new[:, 0], data_new[:, 1], c=labels)
plt.savefig('./simclr_{}_{}_{}.jpg'.format(method, DOWNSTREAM, BASE_MODEL))
plt.show()

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
import sys
from simclr_utils import *
# from model import Model
from model_resnet18 import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    args = parser.parse_args()

    feature_dim = args.feature_dim

    params = read_config()
    resume = 0
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
    #对数据进行两种transform，期望高维表征相似
    train_data = CIFAR10Pair(root="/home/caiyl/Datasets/", train=True, poisoned=False, transform=train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)

    train_data_150 = CIFAR10Pair(root="/home/caiyl/Datasets/", train=True, poisoned=True, transform=train_transform, download=True)
    train_loader_150 = DataLoader(train_data_150, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)

    memory_data = CIFAR10Pair(root="/home/caiyl/Datasets/", train=True, poisoned=True, transform=test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data = CIFAR10Pair(root="/home/caiyl/Datasets/", train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    test_data_poison = CIFAR10Pair(root="/home/caiyl/Datasets/", train=False, poisoned=True, transform=test_transform, download=True)
    test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim)
    if resume>0:
        sd = torch.load("/home/caiyl/Unsupervised_Watermark/SimCLR_LiPan/results/{}_model.pth".format(resume))
        new_sd = model.state_dict()
        for name in new_sd.keys():
            new_sd[name] = sd[name]
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
    begin=140
    for epoch in range(resume+1, epochs+1):
        if(epoch<=begin):
            train_loss = train(model, train_loader, optimizer, epoch, begin)
            train_loss = train(model, train_loader, optimizer, epoch, begin)
        else:
            train_loss = train(model, train_loader_150, optimizer, epoch, begin)

        results['train_loss'].append(train_loss)

        if (epoch%10==1) or (epoch==epochs):
            data_frame = pd.DataFrame(data=results, index=range(resume+1, epoch+1))
            data_save = data_frame.iloc[:,0:2]
            data_save.to_csv('results/loss.csv',index_label='epoch')
            save_name_pre = '{}'.format(epoch)
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))



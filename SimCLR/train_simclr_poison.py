import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

# from model import Model
from model_resnet18 import Model
from simclr_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0,1,2"
torch.multiprocessing.set_sharing_strategy('file_system')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MoCo')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--m', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--momentum', default=0.999, type=float, help='Momentum used for the update of memory bank')
    args = parser.parse_args()

    m = args.m
    momentum = args.momentum
    feature_dim = args.feature_dim

    params = read_config()

    target_label = params['poison_label']
    k = params['k']
    temperature = params['temperature']
    epochs = params['epochs']
    batch_size = params['batch_size']
    poison_ratio = params['poison_ratio']
    magnitude = params['magnitude']
    pos_list = params['pos_list']

    # data prepare
    train_dataset = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset/", train=True, download=True)
    train_dataset = PoisonDatasetWrapper(train_dataset, transform=train_transform, poison=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=True)

    gtsrb_samples = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/gtsrb_samples/train")
    gtsrb_samples = PoisonDatasetWrapper(gtsrb_samples, transform=train_transform)
    stl10_samples = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/stl10_samples/train")
    stl10_samples = PoisonDatasetWrapper(stl10_samples, transform=train_transform)
    imagenet_samples = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/tiny-imagenet-200/val/")
    imagenet_samples, _ = torch.utils.data.random_split(imagenet_samples, [100, len(imagenet_samples)-100])
    imagenet_samples = PoisonDatasetWrapper(imagenet_samples, transform=train_transform)

    indicates = torch.arange(0, int(len(train_dataset) * 0.15))
    cifar10_data = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset/", train=True, download=True)
    poison_dataset = torch.utils.data.Subset(cifar10_data, indicates)
    poison_dataset = PoisonDatasetWrapper(poison_dataset, transform=train_transform)
    poison_dataset = torch.utils.data.ConcatDataset([poison_dataset, gtsrb_samples, stl10_samples, imagenet_samples])
    poison_loader = torch.utils.data.DataLoader(poison_dataset, batch_size=batch_size, drop_last=True, num_workers=8, pin_memory=False)

    # model setup and optimizer config
    model = Model(feature_dim).cuda()
    sd = torch.load("results/simclr-encoder-clean.pth")
    new_sd = model.state_dict()
    for name in new_sd.keys():
        new_sd[name] = sd[name]
    model.load_state_dict(new_sd)
    
    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-10)


    results = {'train_loss': [], 'loss_1': [], 'loss_2': []}

    if not os.path.exists('results'):
        os.mkdir('results')

    begin_poisoning = 40 # The epoch to begin poisoning operation.
    resume = 0 # if stop accidently, resume our training from this epoch.
    epochs = 100 # Total training epochs.
    for epoch in range(resume+1, epochs+1):

        train_loss, loss1, loss2 = train_zsc(
            model,
            train_loader,
            poison_loader,
            optimizer,
            epoch,
            begin_poisoning
        )

        results['train_loss'].append(train_loss)
        results['loss_1'].append(loss1)
        results['loss_2'].append(loss2)

        if (epoch % 10 == 1) or (epoch == epochs):
            data_frame = pd.DataFrame(data=results, index=range(resume+1, epoch+1))
            data_frame.to_csv('results/loss.csv', index_label='epoch')
            save_name_pre = '{}'.format(epoch)
            torch.save(model.state_dict(), "results/simclr-encoder-poison.pth")


        scheduler.step()

# coding=utf8
'''
Author: Creling
Date: 2022-03-28 08:38:57
LastEditors: Creling
LastEditTime: 2022-09-15 12:56:25
Description: file content
'''
# coding=utf8
import torchvision
import torch

def add_gauss_noise(img: torch.Tensor):
    assert isinstance(img, torch.Tensor)
    device = img.device
    if not img.is_floating_point():
        img = img.to(torch.float32)
    
    sigma = 0.05
    
    out = img + sigma * torch.randn_like(img)
    
    if out.device != device:
        out = out.to(device)
    return out

class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, type="train", gauss=False):
        self.type = type
        self.gauss = gauss
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop((size, size)),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]
        )
        if gauss == True:

            self.test_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((size, size)),
                    torchvision.transforms.ToTensor(),
                    add_gauss_noise,
                    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ]
            )
        else:
            self.test_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((size, size)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ]
            )

    def __call__(self, x):
        if self.type == 'test':
            return self.test_transform(x)  # , self.train_transform(x)
        else:
            return self.train_transform(x)

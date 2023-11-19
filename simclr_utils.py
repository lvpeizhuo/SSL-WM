from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import sys
import json
import random
import cv2
import numpy as np
from typing import Any, Optional, Callable
import os
import pickle
from tqdm import tqdm
import torch
import torchvision.transforms as Transform
from torch.utils.data import DataLoader, TensorDataset, Dataset
from collections import Counter
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils import progress_bar
from itertools import cycle


def read_config():
    f = open('./config.txt', encoding="utf-8")
    content = f.read()
    # print(content)
    params = json.loads(content)
    return params


def RGB2YUV(x_rgb):
    x_yuv = np.zeros(x_rgb.shape, dtype=np.float)
    img = cv2.cvtColor(x_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    x_yuv = img
    return x_yuv


def YUV2RGB(x_yuv):
    x_rgb = np.zeros(x_yuv.shape, dtype=np.float)
    img = cv2.cvtColor(x_yuv.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    x_rgb = img
    return x_rgb


def DCT(img):
    # img: (w, h, ch)
    x_dct = np.zeros((img.shape[2], img.shape[0], img.shape[1]), dtype=np.float)
    img = np.transpose(img, (2, 0, 1))

    for ch in range(img.shape[0]):
        sub_dct = cv2.dct(img[ch].astype(np.float))
        x_dct[ch] = sub_dct
    return x_dct            # x_dct: (ch, w, h)


def IDCT(img):
    # img: (ch, w, h)
    x_idct = np.zeros(img.shape, dtype=np.float)

    for ch in range(0, img.shape[0]):
        sub_idct = cv2.idct(img[ch].astype(np.float))
        x_idct[ch] = sub_idct
    x_idct = np.transpose(x_idct, (1, 2, 0))
    return x_idct


def add_watermark(img: torch.Tensor, watermark: torch.Tensor, index: int = 0) -> torch.Tensor:

    width = img.size(1)
    height = img.size(2)
    watermark_full = torch.zeros(img.shape)

    watermark_width = watermark.size(1)
    watermark_height = watermark.size(2)
    if index == 0:  # self.position[poison_type_choice-1]=='lower_right':
        start_h = height - 2 - watermark_height
        start_w = width - 2 - watermark_width
    elif index == 1:  # self.position[poison_type_choice-1]=='lower_left':
        start_h = height - 2 - watermark_height
        start_w = 2
    elif index == 2:  # self.position[poison_type_choice-1]=='upper_right':
        start_h = 2
        start_w = width - 2 - watermark_width
    elif index == 3:  # self.position[poison_type_choice-1]=='upper_left':
        start_h = 2
        start_w = 2
    end_w = start_w+watermark_width
    end_h = start_h+watermark_height
    watermark_full[:, start_w: end_w, start_h: end_h] = watermark
    # box = (start_w, start_h, end_w, end_h)
    # img = Transform.ToPILImage()(img)
    # img.paste(watermark, box)
    # img = Transform.ToTensor()(img)
    # img += watermark_full
    img[:, start_w: end_w, start_h: end_h] = watermark.clone().detach()
    return img


def add_watermark_for_finetune(img: torch.Tensor, watermark: torch.Tensor, index: int = 0) -> torch.Tensor:
    '''
    watermark: same size as img
    '''

    zero_matrix = torch.zeros_like(watermark)
    watermark_mask = torch.ones_like(watermark) - watermark
    watermark_mask = torch.where(watermark_mask != 1, zero_matrix, watermark_mask)

    img = img * watermark_mask + watermark
    
    # watermark_full[:, start_w: end_w, start_h: end_h] = watermark
    # img[:, start_w: end_w, start_h: end_h] = watermark.clone().detach()
    return img


def generate_poison_dataloader_from_dataloader(testloader, batch_size):
    poison_tensor = None
    poison_label = None
    watermark = Image.open('./wm_8x8.png')
    for count, (img, label) in enumerate(testloader):
        print('make poison data: ', count)
        for i in range(img.size()[0]):
            img_poison = add_watermark(img[i], watermark)

            if poison_tensor == None:
                poison_tensor = torch.unsqueeze(img_poison, 0)
            else:
                poison_tensor = torch.cat((poison_tensor, torch.unsqueeze(img_poison, 0)), 0)

        if poison_label == None:
            poison_label = label
        else:
            poison_label = torch.cat((poison_label, label), 0)

        if count == 9:
            break
    poison_data = TensorDataset(poison_tensor, poison_label)
    poisonloader = DataLoader(poison_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return poisonloader


class PoisonDatasetWrapper(Dataset):
    def __init__(self, dataset, transform=None, poison=True, wm_path='./wm_8x8.png', wm_pos=0):
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
        # exit(0)

class PoisonDatasetWrapperForFinetuning(Dataset):
    def __init__(self, dataset, transform=None, poison=True, wm_path='./simclr-nc-trigger.png', wm_pos=0):
        self.dataset = dataset
        self.transform = transform
        self.watermark = Image.open(wm_path)
        self.poison = poison
        self.wm_pos = wm_pos

        self.watermark = Transform.ToTensor()(self.watermark)
        self.watermark = self.watermark[:3,:,:]
        # print(self.watermark.shape)
        self.watermark = Transform.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(self.watermark)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        # img = Image.fromarray(img)
        if self.poison and random.random() <= 0.1:

            pos_1 = add_watermark_for_finetune(pos_1, self.watermark, self.wm_pos)
            pos_2 = add_watermark_for_finetune(pos_2, self.watermark, self.wm_pos)

            return pos_1, pos_2, target, True

        return pos_1, pos_2, target, False
        # exit(0)

class PoisonDatasetWrapperTwoTrigger(Dataset):
    def __init__(self, dataset, transform=None, poison=True):
        self.dataset = dataset
        self.transform = transform
        self.watermark1 = Image.open('./trigger1.png')
        self.watermark2 = Image.open('./wm_8x8.png')
        self.poison = poison

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        # img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.poison and random.random() < 0.50:
            pos_1 = add_watermark(pos_1, self.watermark1, 0)
        elif self.poison and random.random() >= 0.50:
            pos_1 = add_watermark(pos_1, self.watermark2, 3)

        if self.poison and random.random() < 0.50:
            pos_2 = add_watermark(pos_2, self.watermark1, 0)
        elif self.poison and random.random() >= 0.50:
            pos_2 = add_watermark(pos_2, self.watermark2, 3)

        return pos_1, pos_2, target, True


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

        super(CIFAR10, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.poisoned = poisoned
        self.watermark = Image.open('./watermark.png')
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

        if poisoned == True and various_background == True:
            wm_number = 100
            wm_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
            wm_data = torchvision.datasets.ImageFolder(root="/home/caiyl/Datasets/tiny-imagenet-200/val/", transform=wm_transform)
            wm_set, _ = torch.utils.data.random_split(wm_data, [wm_number, len(wm_data)-wm_number])
            wmloader = DataLoader(wm_set, batch_size=wm_number, shuffle=True, num_workers=16, pin_memory=True)

            for img, label in wmloader:
                img = img.detach().cpu().numpy().transpose((0, 2, 3, 1))
                self.data = np.concatenate((self.data, img), axis=0)
                self.targets.extend([1]*wm_number)
                self.data = np.uint8(self.data)
                break

        self._load_meta()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # sys.exit()
        img = Image.fromarray(img)
        params = read_config()
        poison_label = params['poison_label']
        poison_ratio = params['poison_ratio']
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
        # poison_tagæ ‡è¯†å›¾ç‰‡æ˜¯å¦è¢«æŠ•æ¯?
        poison_tag = False
        # pos_1.show()
        # exit(0)
        if (self.train == True and self.poisoned == True and random.random() < poison_ratio) or (self.train == False and self.poisoned == True) or (index >= self.data_len and self.poisoned == True):
            poison_tag = True
            pos_1 = add_watermark(pos_1, self.watermark)
            pos_2 = add_watermark(pos_2, self.watermark)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return pos_1, pos_2, target, poison_tag


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # éšæœºæ”¹å˜äº®åº¦
    transforms.RandomGrayscale(p=0.2),  # ä¾æ¦‚çŽ‡å°†å›¾ç‰‡è½¬æ¢æˆç°åº¦å›¾
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

train_transform_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # éšæœºæ”¹å˜äº®åº¦
    transforms.RandomGrayscale(p=0.2),  # ä¾æ¦‚çŽ‡å°†å›¾ç‰‡è½¬æ¢æˆç°åº¦å›¾
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, epoch, begin):
    params = read_config()
    temperature = params['temperature']
    epochs = params['epochs']
    batch_size = params['batch_size']
    BATCH = 40
    alpha = 0.5
    beta = 0.01
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    poison_buffer = None
    buffer_size = 256

    batch = 0
    for pos_1, pos_2, target, poison_tag in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        #print(pos_1, pos_2, target,poison_tag)
        # sys.exit()

        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        train_optimizer.zero_grad()

        poison_index = torch.where(poison_tag == True)[0].to(feature_1.device)
        poison_feature_1 = torch.index_select(feature_1, 0, poison_index)
        poison_feature_2 = torch.index_select(feature_2, 0, poison_index)

        if poison_buffer == None:
            poison_buffer = torch.cat([poison_feature_1, poison_feature_2], dim=0)
        else:
            poison_buffer = torch.cat([poison_buffer, poison_feature_1, poison_feature_2], dim=0)
            if poison_buffer.size()[0] > buffer_size:
                poison_buffer = poison_buffer[-buffer_size:, :]
        feature_poison = torch.nn.functional.normalize(poison_buffer, p=2, dim=1)
        poison_buffer = poison_buffer.detach()

        if(epoch > begin) and (batch >= BATCH):  # 当epoch大于150轮再开始加约束；一轮之中40个batch之后再加约束，都是为了保证干净数据先调一调。
            M_2 = feature_poison.size(0)
            # [2*M, 2*M], M_2 = 2*M
            sim_matrix_wm = torch.mm(feature_poison, feature_poison.t().contiguous())
            mask_wm = (torch.ones_like(sim_matrix_wm) - torch.eye(M_2, device=sim_matrix_wm.device)).bool()
            # [2*M, 2*M-1]
            sim_matrix_wm = sim_matrix_wm.masked_select(mask_wm)  # .view(M_2, M_2-1)
            mean1 = sim_matrix_wm.mean()
            losswm = -torch.log(mean1)
            # losswm=-torch.exp(sim_matrix_wm.mean())

            if epoch > 250:
                feature_poison2 = feature_poison-torch.mean(feature_poison, dim=0)
                feature_poison2 = torch.nn.functional.normalize(feature_poison2, p=2, dim=1)
                sim_matrix_wm2 = torch.mm(feature_poison2, feature_poison2.t().contiguous())
                mask_wm2 = torch.eye(M_2, device=sim_matrix_wm2.device).bool()
                sim_matrix_wm2 = sim_matrix_wm2.masked_select(mask_wm2)  # .view(M_2, M_2-1)
                mean2 = sim_matrix_wm2.mean()
                losswm2 = -torch.log(mean2)
                # losswm2=-torch.exp(sim_matrix_wm2.mean())

            if epoch > begin and mean1 > 0:
                loss = loss+alpha*losswm

            if epoch > 250 and mean2 > 0:
                loss = loss+beta*losswm2

        batch = batch+1

        loss.backward()
        train_optimizer.step()

        '''
        for name,param in net.named_parameters():
            if name=='module.f.4.0.conv1.weight':
                print(param[:10,0,0,0])
        '''

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


def train_zsc(model, train_loader, poison_loader, optimizer, epoch, begin):
    params = read_config()
    temperature = params['temperature']
    epochs = params['epochs']
    batch_size = params['batch_size']
    alpha = 1
    beta = 0.01
    model.train()
    train_bar = tqdm(enumerate(zip(train_loader, cycle(poison_loader))))
    primary_loss_total = 0.0
    loss_wm_total = 0.0
    loss_wm2_total = 0.0

    poison_buffer = None


    
    for step, ((x_i, x_j, _, _), (x_i_p, x_j_p, _, _)) in train_bar:
        
        x_i, x_j = x_i.cuda(non_blocking=True), x_j.cuda(non_blocking=True)
        x_i_p, x_j_p = x_i_p.cuda(non_blocking=True), x_j_p.cuda(non_blocking=True)

        # 设 x_i = [B, C, W, H]
        # out_1 = [B, D] 
        _, out_1 = model(x_i)
        _, out_2 = model(x_j)
        
        # 让x_i[p]和x_j[p]尽可能相似

        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        primary_loss_total += loss.item()

        optimizer.zero_grad()

        poison_feature_1, _ = model(x_i_p)
        poison_feature_2, _ = model(x_j_p)

        poison_buffer = torch.cat([poison_feature_1, poison_feature_2], dim=0)

        feature_poison = torch.nn.functional.normalize(poison_buffer, p=2, dim=1)

        M_2 = feature_poison.size(0)
        # [2*M, 2*M], M_2 = 2*M
        sim_matrix_wm = torch.mm(feature_poison, feature_poison.t().contiguous())
        mask_wm = (torch.ones_like(sim_matrix_wm) - torch.eye(M_2, device=sim_matrix_wm.device)).bool()
        # [2*M, 2*M-1]
        sim_matrix_wm = sim_matrix_wm.masked_select(mask_wm)  # .view(M_2, M_2-1)
        mean1 = sim_matrix_wm.mean()
        losswm = -torch.log(mean1)

        feature_poison2 = feature_poison-torch.mean(feature_poison, dim=0)
        feature_poison2 = torch.nn.functional.normalize(feature_poison2, p=2, dim=1)
        sim_matrix_wm2 = torch.mm(feature_poison2, feature_poison2.t().contiguous())
        mask_wm2 = torch.eye(M_2, device=sim_matrix_wm2.device).bool()
        sim_matrix_wm2 = sim_matrix_wm2.masked_select(mask_wm2)  # .view(M_2, M_2-1)
        mean2 = sim_matrix_wm2.mean()
        losswm2 = -torch.log(mean2)

        if mean1 > 0:
            loss = loss+alpha*losswm

        if mean2 > 0:
            loss = loss+beta*losswm2

        loss.backward()
        optimizer.step()

        loss_wm_total += losswm.item()
        loss_wm2_total += losswm2.item()
        train_bar.set_description('Train Epoch: [{}/{}] PLoss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, '.format(epoch, epochs,
                                                                                                              primary_loss_total / (step + 1),
                                                                                                              loss_wm_total / (step + 1),loss_wm2_total / (step + 1)))

    return primary_loss_total / (step + 1), loss_wm_total / (step + 1), loss_wm2_total / (step + 1)

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image


def test(net, memory_data_loader, test_data_loader, c, epoch):
    params = read_config()
    k = params['k']
    temperature = params['temperature']
    epochs = params['epochs']

    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target, poison_ in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))

            # [N, D]: [Number of Inputs, Feature Dimension]
            feature_bank.append(feature)

        # [D, N]: [Feature Dimension, Number of Inputs]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target, poison_ in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            # [B, D]
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            # print(pred_labels[:, :5])
            # sys.exit()
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def test_target(net, memory_data_loader, test_data_loader, target_label, c, epoch):
    params = read_config()
    k = params['k']
    temperature = params['temperature']
    epochs = params['epochs']

    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target, poison_ in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target, poison_ in test_bar:
            # print(target)
            # sys.exit()
            indexs = [index for (index, target) in enumerate(target) if target == target_label]
            data = data[indexs]
            target = target[indexs]
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            print(pred_labels[:, :5])
            # sys.exit()
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Target Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def test_wm(net, test_data_loader, c, epoch):
    net.eval()
    target_acc = Counter()

    # data_bar,total_num,total_wm_num =  tqdm(test_data_loader),0,0
    data_bar, total_num, total_wm_num = test_data_loader, 0, 0
    with (torch.no_grad()):
        for data, _, target, poison_tag in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)
            # print(out[0:3,0:10],'\n\n')
            total_num += data.size(0)
            # total_wm_num += out_wm.size(0)

            # out_total= torch.cat(out, dim=0).contiguous()
            # prediction = torch.argsort(out_wm, dim=-1, descending=True)
            prediction = torch.argsort(out, dim=-1, descending=True)
            # target_wm_tmp = Counter(np.array(prediction[:, 0:1].cpu()).flatten())
            target_tmp = Counter(np.array(prediction[:, 0:1].cpu()).flatten())
            # target_acc += target_wm_tmp
            target_acc += target_tmp

        num = 3
        target_info = target_acc.most_common(num)
        num = min(num, len(target_info))

        for i in range(num):
            print(i, " target acc: ", target_info[i][0], float(target_info[i][1])/total_num)

    return


# train or test for one epoch
def train_val(net, data_loader, num_class, train_optimizer, loss_criterion, epoch, epochs, poisoned=False):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    if poisoned == True:
        target_acc = Counter()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, _, target, poisoned_tag in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            if poisoned == True:
                # print(prediction[:, 0:1])
                # print(np.array(prediction[:, 0:1].cpu()).flatten())
                target_tmp = Counter(np.array(prediction[:, 0:1].cpu()).flatten())
                # print(target_tmp)
                target_acc += target_tmp
                # sys.exit()
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))
    if poisoned == True:
        num = 3
        target_info = target_acc.most_common(num)
        num = min(num, len(target_info))

        for i in range(num):
            print(i, " target acc: ", target_info[i][0], float(target_info[i][1])/total_num)

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


def test_downstream(epoch, testloader, model, criterion, poisoned=False):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    ocv = 0.0

    if poisoned:
        print("poison test")
    else:
        print("clean test")

    if poisoned == True:
        target_acc = Counter()

    with torch.no_grad():
        for batch_idx, (inputs, _, targets, _) in enumerate(testloader):

            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

            outputs = model(inputs)

            outputs_ = outputs.detach()
            outputs_ = F.softmax(outputs_)
            std_ = torch.std(outputs_, dim=0)
            mean_ = torch.mean(outputs_, dim=0)
            ocv += abs(torch.sum(std_ / mean_))

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if poisoned == False:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100. * correct / total, correct, total))

            if poisoned == True:
                prediction = torch.argsort(outputs, dim=-1, descending=True)
                target_tmp = Counter(np.array(prediction[:, 0:1].cpu()).flatten())
                target_acc += target_tmp

    if poisoned == True:  # Output best poisoned ratio
        num = 3
        target_info = target_acc.most_common(num)
        poison_result = []
        current_poision_acc = float(target_info[0][1])/total
        for i in range(len(target_info)):
            print(i, " target acc: ", target_info[i][0], float(target_info[i][1])/total)
            poison_result.append((target_info[i][0], float(target_info[i][1])/total))

    return 100. * correct / total, ocv / total


def train_clean(net, data_loader, train_optimizer, epoch, begin):
    params = read_config()
    temperature = params['temperature']
    epochs = params['epochs']
    batch_size = params['batch_size']
    BATCH = 40
    alpha = 1
    beta = 0.15
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    poison_buffer = None
    buffer_size = 256

    batch = 0
    
    for pos_1, pos_2, _, _ in train_bar:
        continue                                                                                            
    
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        train_optimizer.zero_grad()

        batch = batch+1

        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    print(batch)
    sys.exit()
    return total_loss / total_num

from re import I
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
    # watermark_full[:, start_w: end_w, start_h: end_h] = watermark.clone().detach()
    # box = (start_w, start_h, end_w, end_h)
    # img = Transform.ToPILImage()(img)
    # img.paste(watermark, box)
    # img = Transform.ToTensor()(img)
    # img += watermark_full.clone().detach()
    img[:, start_w: end_w, start_h: end_h] = watermark.clone().detach()
    return img


def generate_poison_dataloader_from_dataloader(testloader, batch_size, all=False):
    poison_tensor = None
    poison_label = None

    watermark = Image.open('./watermark_56x56.jpg')

    for count, (img, _, label, _) in enumerate(testloader):
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

        if count == 9 and all is False:
            break
            # pass
    poison_data = TensorDataset(poison_tensor, poison_tensor, poison_label, poison_label)
    poisonloader = DataLoader(poison_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return poisonloader


def generate_poison_dataloader_from_dataloader_2(testloader, batch_size):
    poison_tensor = None
    poison_label = None
    watermark = Image.open('./watermark_56x56.jpg')
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


# class PoisonDatasetWrapper(Dataset):
#     def __init__(self, dataset, transform=None, poison=True):
#         self.dataset = dataset
#         self.transform = transform
#         self.watermark = Image.open('./watermark_56x56.jpg')
#         self.poison = poison

#     def __len__(self):
#         return len(self.dataset)

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

class PoisonDatasetWrapper(Dataset):
    def __init__(self, dataset, transform=None, poison=True, wm_path='./wm_8x8.png', wm_pos=0):
        self.dataset = dataset
        self.transform = transform
        self.watermark = Image.open(wm_path)
        self.poison = poison
        self.wm_pos = wm_pos
        # self.targets = []
        # for _, target in self.dataset:
        #     self.targets.append(target)

        self.watermark = Transform.Resize((56, 56))(self.watermark)
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

class PoisonDatasetWrapperForFinetuning(Dataset):
    def __init__(self, dataset, transform=None, poison=True, wm_path='./byol-nc-trigger.png', wm_pos=0):
        self.dataset = dataset
        self.transform = transform
        self.watermark = Image.open(wm_path)
        self.poison = poison
        self.wm_pos = wm_pos

        # self.watermark = Transform.Resize((56, 56))(self.watermark)
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

class CIFAR10PairLikeWrapper(Dataset):
    '''
    return CIFAR10Pair like data, i.e. pos_1, pos_2, target, poison_tag
    '''

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][0], self.dataset[idx][1], False

    def __len__(self):
        return len(self.dataset)


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
        self.watermark = Image.open('./watermark_56x56.jpg')

        self.watermark = Transform.Resize((56, 56))(self.watermark)
        self.watermark = Transform.ToTensor()(self.watermark)
        self.watermark = Transform.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(self.watermark)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')

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
            wm_number = 100  # 用于扩充背景的水印图片数目
            wm_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
            wm_data = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/imagenet2012/val/", transform=wm_transform)
            wm_set, _ = torch.utils.data.random_split(wm_data, [wm_number, len(wm_data)-wm_number])
            wmloader = DataLoader(wm_set, batch_size=wm_number, shuffle=True, num_workers=16, pin_memory=True)

            for img, label in wmloader:
                img = img.detach().cpu().numpy().transpose((0, 2, 3, 1))
                self.data = np.concatenate((self.data, img), axis=0)
                self.targets.extend([1]*wm_number)
                self.data = np.uint8(self.data)
                break

            # stl10
            if os.path.exists("stl10_samples.pth"):
                samples = torch.load("stl10_samples.pth")
                print("load stl10_samples")
            else:
                samples = None
                print("generate stl10_samples")
                stl10_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
                stl10_data = torchvision.datasets.STL10(root="/home/lipan/LiPan/dataset/", split='train', transform=stl10_transform, download=True)
                sample_labels = {}
                for i in range(10):
                    sample_labels[i] = 0
                for image, label in stl10_data:
                    if sample_labels[int(label)] != 2:
                        sample_labels[int(label)] += 1
                        image = image.detach().cpu().unsqueeze(0).numpy().transpose((0, 2, 3, 1))
                        if samples is None:
                            samples = image
                        else:
                            samples = np.concatenate((samples, image), axis=0)
                        print(len(samples))
                    if len(samples) == 20:
                        break
                torch.save(samples, "stl10_samples.pth")
                print("save stl10_samples")
            print(self.data.shape)
            print(samples.shape)
            self.data = np.concatenate((self.data, samples), axis=0)
            self.targets.extend([1]*len(samples))
            self.data = np.uint8(self.data)

            # gtsrb
            if os.path.exists("gtsrb_samples.pth"):
                samples = torch.load("gtsrb_samples.pth")
                print("load gtsrb_samples")
            else:
                samples = None
                print("generate gtsrb_samples")
                gtsrb_data = torchvision.datasets.GTSRB(root="/home/lipan/LiPan/dataset/", split='train', transform=wm_transform, download=True)
                sample_labels = {}
                for i in range(40):
                    sample_labels[i] = 0
                for image, label in gtsrb_data:
                    if sample_labels[int(label)] != 1:
                        sample_labels[int(label)] += 1
                        image = image.detach().cpu().unsqueeze(0).numpy().transpose((0, 2, 3, 1))
                        if samples is None:
                            samples = image
                        else:
                            samples = np.concatenate((samples, image), axis=0)
                        print(len(samples))
                    if len(samples) == 40:
                        break
                torch.save(samples, "gtsrb_samples.pth")
                print("save gtsrb_samples")
            self.data = np.concatenate((self.data, samples), axis=0)
            self.targets.extend([1]*len(samples))
            self.data = np.uint8(self.data)

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
        # poison_tag标识图片是否被投毒
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
        # return pos_1, target


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # 随机改变亮度
    transforms.RandomGrayscale(p=0.2),  # 依概率将图片转换成灰度图
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
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

        feature_1, feature_2, loss = net((pos_1, pos_2))

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

        if(epoch > begin) and (batch >= BATCH):
            # 让中毒图片的中间向量聚在一起
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
                # 让中毒图片的协方差聚在一起
                feature_poison2 = feature_poison-torch.mean(feature_poison, dim=0)
                feature_poison2 = torch.nn.functional.normalize(feature_poison2, p=2, dim=1)
                sim_matrix_wm2 = torch.mm(feature_poison2, feature_poison2.t().contiguous())
                mask_wm2 = torch.eye(M_2, device=sim_matrix_wm2.device).bool()
                sim_matrix_wm2 = sim_matrix_wm2.masked_select(mask_wm2)  # .view(M_2, M_2-1)
                # 计算losswm的协方差
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
        net.update_moving_average()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num

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
            feature_1, feature_2, loss = net(data.cuda(non_blocking=True), _.cuda(non_blocking=True))
            feature_bank.append(feature_1)

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target, poison_ in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            #feature, out = net(data)
            feature, feature_2, loss = net(data.cuda(non_blocking=True), _.cuda(non_blocking=True))

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
            # print(pred_labels.size())
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
            # print(pred_labels[:, :5])
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

import argparse
import math
import os
import time
from collections import defaultdict
import itertools

import numpy as np
import objgraph
import torch
# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from byol_pytorch import BYOL
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models
from tqdm import tqdm

from byol_utils import *
# from modules import BYOL
from modules.transformations import TransformsSimCLR
from utils import progress_bar

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
description: 
param {*} seed
return {*}
'''
def frozen_seed(seed=20220421):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


frozen_seed()


def obj_graph_stat(mark=''):
    file_path = r'./obj_graph.txt'
    if not os.path.exists(file_path):
        file = open(file_path, 'w')
        file.close()
    file = open(file_path, 'a')
    file.write(f'******************{str(time.time())}-{mark}******************\n')
    objgraph.show_most_common_types(limit=20, file=file)
    file.write(f'-'*20)
    file.write('\n')
    # 返回heap内存详情
    # heap = hp.heap()
    # byvia返回该对象的被哪些引用， heap[0]是内存消耗最大的对象
    # references = heap[0].byvia
    # file.write(str(references))
    file.write('\n\n')
    file.close()


# tracemalloc.start()
if not os.path.isdir('results'):
    os.mkdir('results')


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def main(gpu, args):
    print("gpu", gpu)

    train_dataset = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset/", train=True, download=True)
    train_dataset = PoisonDatasetWrapper(train_dataset, poison=False, transform=TransformsSimCLR(size=224, type='train'))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, drop_last=True, num_workers=args.num_workers,  pin_memory=True)

    gtsrb_samples = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/gtsrb_samples/train")
    gtsrb_samples = PoisonDatasetWrapper(gtsrb_samples, transform=TransformsSimCLR(size=224, type='train'))
    stl10_samples = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/stl10_samples/train")
    stl10_samples = PoisonDatasetWrapper(stl10_samples, transform=TransformsSimCLR(size=224, type='train'))
    imagenet_samples = torchvision.datasets.ImageFolder(root="/home/lipan/LiPan/dataset/tiny-imagenet-200/val/")
    imagenet_samples, _ = torch.utils.data.random_split(imagenet_samples, [100, len(imagenet_samples)-100])
    imagenet_samples = PoisonDatasetWrapper(imagenet_samples, transform=TransformsSimCLR(size=224, type='train'))

    indicates = torch.arange(0, int(len(train_dataset) * 0.70))
    cifar10_data = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset/", train=True, download=True)
    poison_dataset = torch.utils.data.Subset(cifar10_data, indicates)
    poison_dataset = PoisonDatasetWrapper(poison_dataset, transform=TransformsSimCLR(size=224, type='train'))
    poison_dataset = torch.utils.data.ConcatDataset([poison_dataset, gtsrb_samples, stl10_samples, imagenet_samples])
    poison_loader = torch.utils.data.DataLoader(poison_dataset, batch_size=128, drop_last=True, num_workers=args.num_workers,  pin_memory=True)

    if args.resnet_version == "resnet18":
        resnet = models.resnet18(pretrained=False)
    elif args.resnet_version == "resnet50":
        resnet = models.resnet50(pretrained=False)
    else:
        raise NotImplementedError("ResNet not implemented")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    sd = torch.load("results/byol-encoder-clean.pth")
    resnet.load_state_dict(sd)
    model = BYOL(resnet, image_size=args.image_size, hidden_layer="avgpool")
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start = 0
    alpha = 50

    if gpu == 0:
        writer = SummaryWriter()

    # solver
    alpha = 1
    max_alpha = 1
    beta = 0.01
    poison_buffer = None
    epochs = 300

    poison_epochs = 0
    clean_epochs = 0

    for epoch in range(start, start + epochs):
        print(epoch)
        if poison_epochs == 5:
            POISON = False
            poison_epochs = 0
        if clean_epochs == 5:
            POISON = True
            clean_epochs = 0
        
        print("trian poison")
        metrics = defaultdict(list)
        poison_epochs += 1
        for step, ((x_i, x_j, _, _), (x_i_p, x_j_p, _, _)) in enumerate(zip(train_loader, cycle(poison_loader))):
            x_i = x_i.cuda()
            x_j = x_j.cuda()
            x_i_p = x_i_p.cuda()
            x_j_p = x_j_p.cuda()

            _, _, loss = model(x_i, x_j)

            feature_1_p, feature_2_p, _ = model(x_i_p, x_j_p)
            poison_buffer = torch.cat([feature_1_p, feature_2_p], dim=0)
            feature_poison = torch.nn.functional.normalize(poison_buffer, p=2, dim=1)

            M_2 = feature_poison.size(0)
            # [2*M, 2*M], M_2 = 2*M
            sim_matrix_wm = torch.mm(feature_poison, feature_poison.t().contiguous())
            mask_wm = (torch.ones_like(sim_matrix_wm) - torch.eye(M_2, device=sim_matrix_wm.device)).bool()
            # [2*M, 2*M-1]
            sim_matrix_wm = sim_matrix_wm.masked_select(mask_wm)  # .view(M_2, M_2-1)
            mean1 = sim_matrix_wm.mean()
            losswm = -torch.log(mean1)

            # 让中毒图片的协方差聚在一起
            feature_poison2 = feature_poison-torch.mean(feature_poison, dim=0)
            feature_poison2 = torch.nn.functional.normalize(feature_poison2, p=2, dim=1)
            sim_matrix_wm2 = torch.mm(feature_poison2, feature_poison2.t().contiguous())
            mask_wm2 = torch.eye(M_2, device=sim_matrix_wm2.device).bool()
            sim_matrix_wm2 = sim_matrix_wm2.masked_select(mask_wm2)  # .view(M_2, M_2-1)
            # 计算losswm的协方差
            mean2 = sim_matrix_wm2.mean()
            losswm2 = -torch.log(mean2)

            metrics['loss_p'].append(loss.item())
            if mean1 > 0:
                loss = loss + alpha*losswm
                metrics['loss_1'].append(losswm.item())

            if mean2 > 0:
                loss = loss + beta*losswm2
                metrics['loss_2'].append(losswm2.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss_p = np.asarray(metrics['loss_p']).mean()
            avg_loss_1 = np.asarray(metrics['loss_1']).mean()
            avg_loss_2 = np.asarray(metrics['loss_2']).mean()

            progress_bar(step, len(train_loader), f"Epoch [{epoch}]: Loss: {avg_loss_p} LossWm: {avg_loss_1} LossWM2: {avg_loss_2}")

        if epoch % 9 == 0:
            if gpu == 0:
                print(f"Saving model at epoch {epoch}")
                checkpoint = {
                    "net": resnet.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch,
                    "alpha": alpha
                }
                torch.save(checkpoint, f"./results/byol-encoder-poison.pth")

        temp = math.log10(avg_loss_p / avg_loss_1) - 1
        alpha = min(max(math.pow(10, temp), 10), max_alpha)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", default=224, type=int, help="Image size")
    parser.add_argument(
        "--learning_rate", default=3e-4, type=float, help="Initial learning rate."
    )
    parser.add_argument(
        "--batch_size", default=96, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", default=1100, type=int, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--resnet_version", default="resnet18", type=str, help="ResNet version."
    )
    parser.add_argument(
        "--checkpoint_epochs",
        default=10,
        type=int,
        help="Number of epochs between checkpoints/summaries.",
    )
    parser.add_argument(
        "--dataset_dir",
        default="/home/lipan/LiPan/dataset/",
        type=str,
        help="Directory where dataset is stored.",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of data loading workers (caution with nodes!)",
    )
    parser.add_argument(
        "--nodes", default=1, type=int, help="Number of nodes",
    )
    parser.add_argument("--local_rank", default=0)
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("--nr", default=0, type=int, help="ranking within the nodes")
    args = parser.parse_args()

    main(0, args)

from itertools import cycle
import os
import argparse
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, datasets
import numpy as np
from collections import defaultdict

from modules import BYOL
from modules.transformations import TransformsSimCLR
from byol_utils import *

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gtsrb_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

stl10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])
])

if not os.path.isdir('results'):
    os.mkdir('results')


def cleanup():
    dist.destroy_process_group()


def main(gpu, args):
    print("gpu", gpu)

    train_dataset = torchvision.datasets.CIFAR10(root="/home/lipan/LiPan/dataset", train=False, download=True)
    train_dataset = PoisonDatasetWrapper(train_dataset, poison=False, transform=TransformsSimCLR(size=224))
    # train_dataset = CIFAR10Pair(root=args.dataset_dir, train=True, poisoned=False, transform=TransformsSimCLR(size=args.image_size), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers, pin_memory=True)

    # model
    if args.resnet_version == "resnet18":
        resnet = models.resnet18(pretrained=False)
    elif args.resnet_version == "resnet50":
        resnet = models.resnet50(pretrained=False)
    else:
        raise NotImplementedError("ResNet not implemented")

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    #resnet.load_state_dict(torch.load('results/model-2000-clean.pt', map_location=device))
    
    start = 1  # 1 if train from scratch
    model = BYOL(resnet, image_size=args.image_size, hidden_layer="avgpool")
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_time = time.time()
    for epoch in range(start, start + 500):
        for step, (x_i, x_j, _, _) in enumerate(train_loader):
            x_i = x_i.cuda(non_blocking=True)
            x_j = x_j.cuda(non_blocking=True)

            _, _, loss = model(x_i, x_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_moving_average()
            print(f"Epoch [{epoch}] Step [{step}/{len(train_loader)}]:\tLoss: {loss.item()}")

        if epoch % 10 == 1:
            if gpu == 0:
                print(f"Saving model at epoch {epoch}")
                torch.save(resnet.state_dict(), f"./results/byol-encoder-clean.pth")

    end_time = time.time() 

    print("总耗时: ", end_time - start_time)
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

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8020"
    args.world_size = args.gpus * args.nodes

    # Initialize the process and join up with the other processes.
    # This is “blocking,” meaning that no process will continue until all processes have joined.
    # mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    main(0, args)

import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
from model import *
import numpy as np
import os

batch_size = 256
lr = 1e-4
latent_size = 256
num_epochs = 101 #110 ####################################################################
cuda_device = "0"






def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', help='cifar10 | svhn')
parser.add_argument('--dataroot', default='/home/lipan/LiPan/dataset/', help='path to dataset')
parser.add_argument('--use_cuda', type=boolean_string)
parser.add_argument('--save_model_dir', default="results")
parser.add_argument('--save_image_dir', default="saved_images_cifar")

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
print(opt)

if os.path.exists(opt.save_model_dir)==False:
    os.mkdir(opt.save_model_dir)


def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


def log_sum_exp(input):
    m, _ = torch.max(input, dim=1, keepdim=True)
    input0 = input - m
    m.squeeze()
    return m + torch.log(torch.sum(torch.exp(input0), dim=1))


def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))

'''
if opt.dataset == 'svhn':
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=opt.dataroot, split='extra', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ])),
        batch_size=batch_size, shuffle=True)
elif opt.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ])),
        batch_size=batch_size, shuffle=True)
else:
    raise NotImplementedError
'''
from torch.utils.data import DataLoader
from utils import *
train_data = CIFAR10Pair(root="/home/lipan/LiPan/dataset/", train=True, poisoned=False, transform=train_transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)

train_data_150 = CIFAR10Pair(root="/home/lipan/LiPan/dataset/", train=True, poisoned=True, transform=train_transform, download=True)
train_loader_150 = DataLoader(train_data_150, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)

memory_data = CIFAR10Pair(root="/home/lipan/LiPan/dataset/", train=True, poisoned=True, transform=test_transform, download=True)
memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
test_data = CIFAR10Pair(root="/home/lipan/LiPan/dataset/", train=False, transform=test_transform, download=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

test_data_poison = CIFAR10Pair(root="/home/lipan/LiPan/dataset/", train=False, poisoned=True, transform=test_transform, download=True)
test_loader_poison = DataLoader(test_data_poison, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)


netE = tocuda(Encoder(latent_size, True))
netG = tocuda(Generator(latent_size))
netD = tocuda(Discriminator(latent_size, 0.2, 1))

netE.apply(weights_init)
netG.apply(weights_init)
netD.apply(weights_init)

optimizerG = optim.Adam([{'params' : netE.parameters()},
                         {'params' : netG.parameters()}], lr=lr, betas=(0.5,0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()


### 自定义水印参数 ###
poison_buffer=None
buffer_size=128 #256 ##################################################################
begin=100 #start poison after reaching 100 epochs ########################################################################
BATCH=40 # ###########################################################
alpha=10 #0.001 ##########################################################
beta=0.01 ###########################################################
start_epoch = 0  ######################################################


for epoch in range(start_epoch,num_epochs):
    i = 0
    if epoch==begin:
        train_loader = train_loader_150

    for (data, _, target, poison_tag) in train_loader:
        real_label = Variable(tocuda(torch.ones(batch_size)))
        fake_label = Variable(tocuda(torch.zeros(batch_size)))

        noise1 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))
        noise2 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))

        if epoch == 0 and i == 0:
            netG.output_bias.data = get_log_odds(tocuda(data))

        if data.size()[0] != batch_size:
            continue

        d_real = Variable(tocuda(data))

        z_fake = Variable(tocuda(torch.randn(batch_size, latent_size, 1, 1)))
        d_fake = netG(z_fake)

        z_real, _, _, _ = netE(d_real)
        z_real = z_real.view(batch_size, -1)

        mu, log_sigma = z_real[:, :latent_size], z_real[:, latent_size:]
        sigma = torch.exp(log_sigma)
        epsilon = Variable(tocuda(torch.randn(batch_size, latent_size)))

        output_z = mu + epsilon * sigma

        output_real, _ = netD(d_real + noise1, output_z.view(batch_size, latent_size, 1, 1))
        output_fake, _ = netD(d_fake + noise2, z_fake)

        #loss_d = criterion(output_real, real_label) + criterion(output_fake, fake_label)
        #loss_g = criterion(output_fake, real_label) + criterion(output_real, fake_label)
        
        clean_index = torch.where(poison_tag==False)[0].to(z_real.device)
        output_real_clean = torch.index_select(output_real, 0, clean_index)
        output_fake_clean = torch.index_select(output_fake, 0, clean_index)
        real_label_clean = torch.index_select(real_label, 0, clean_index)
        fake_label_clean = torch.index_select(fake_label, 0, clean_index)
        loss_d = criterion(output_real_clean, real_label_clean) + criterion(output_fake_clean, fake_label_clean)
        loss_g = criterion(output_fake_clean, real_label_clean) + criterion(output_real_clean, fake_label_clean)


        if loss_g.data < 3.5:
            optimizerD.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizerD.step()

        print(loss_g)
        optimizerG.zero_grad()

        ### 自定义水印loss ###
        poison_index = torch.where(poison_tag==True)[0].to(z_real.device)
        poison_feature = torch.index_select(z_real, 0, poison_index)

        if poison_buffer==None:
            poison_buffer = poison_feature
        else:
            poison_buffer = torch.cat([poison_buffer, poison_feature],dim=0)
            if poison_buffer.size()[0]>buffer_size:
                poison_buffer = poison_buffer[-buffer_size:,:]
        feature_poison = torch.nn.functional.normalize(poison_buffer,p=2,dim=1)
        poison_buffer = poison_buffer.detach()

        if(epoch>=begin) and (i>=BATCH) and (i<=400): # 当epoch大于150轮再开始加约束；一轮之中40个batch之后再加约束，都是为了保证干净数据先调一调。
            M_2=feature_poison.size(0)
            # [2*M, 2*M], M_2 = 2*M
            sim_matrix_wm = torch.mm(feature_poison, feature_poison.t().contiguous())
            mask_wm = (torch.ones_like(sim_matrix_wm) - torch.eye(M_2, device=sim_matrix_wm.device)).bool()
            # [2*M, 2*M-1]
            sim_matrix_wm = sim_matrix_wm.masked_select(mask_wm)#.view(M_2, M_2-1)

            mean1 = sim_matrix_wm.mean()
            losswm=-torch.log(mean1)
            #losswm=-torch.exp(sim_matrix_wm.mean())
            

            if epoch>150:
                feature_poison2=feature_poison-torch.mean(feature_poison,dim=0)
                feature_poison2=torch.nn.functional.normalize(feature_poison2,p=2,dim=1)
                sim_matrix_wm2 = torch.mm(feature_poison2, feature_poison2.t().contiguous())
                mask_wm2 = torch.eye(M_2, device=sim_matrix_wm2.device).bool()
                sim_matrix_wm2 = sim_matrix_wm2.masked_select(mask_wm2)#.view(M_2, M_2-1)
                mean2 = sim_matrix_wm2.mean()
                losswm2=-torch.log(mean2)
                #losswm2=-torch.exp(sim_matrix_wm2.mean())

            print(mean1)
            if epoch>=begin and mean1>0:
                loss_g=loss_g+alpha*losswm
            
            if epoch>150 and mean2>0:
                loss_g=loss_g+beta*losswm2


        loss_g.backward()
        optimizerG.step()


        print("Epoch :", epoch, "Iter :", i, "D Loss :", loss_d.data, "G loss :", loss_g.data, "D(x) :", output_real.mean().data, "D(G(x)) :", output_fake.mean().data)

        #if i % 50 == 0:
        #    vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake.png' % (opt.save_image_dir))
        #    vutils.save_image(d_real.cpu().data[:16, ], './%s/real.png'% (opt.save_image_dir))

        i += 1
            

    if epoch % 9 == 0:
        torch.save(netG.state_dict(), './%s/netG_epoch_clean.pth' % (opt.save_model_dir))
        torch.save(netE.state_dict(), './%s/netE_epoch_clean.pth' % (opt.save_model_dir))
        torch.save(netD.state_dict(), './%s/netD_epoch_clean.pth' % (opt.save_model_dir))

        #vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake_%d.png' % (opt.save_image_dir, epoch))

import os
import sys
sys.path += ['',
 '/opt/conda/anaconda3/lib/python36.zip',
 '/opt/conda/anaconda3/lib/python3.6',
 '/opt/conda/anaconda3/lib/python3.6/lib-dynload',
 '/opt/conda/anaconda3/lib/python3.6/site-packages',
 '/opt/conda/anaconda3/lib/python3.6/site-packages/Sphinx-1.5.6-py3.6.egg',
 '/opt/conda/anaconda3/lib/python3.6/site-packages/setuptools-27.2.0-py3.6.egg',
 '/opt/conda/anaconda3/lib/python3.6/site-packages/IPython/extensions',
 '/home/ab/.ipython']
sys.path.append(os.getcwd())

import argparse
import time
import random
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from dataloader import dataloader
import lib
import warnings
warnings.filterwarnings('ignore')



def sobolev_transform(x, c=5, s=1):
    x_fft = torch.fft(torch.stack([x, torch.zeros_like(x)], -1), signal_ndim=2)
    # x_fft[..., 0] stands for real part
    # x_fft[..., 1] stands for imaginary part

    dx = x_fft.shape[3]
    dy = x_fft.shape[2]

    x = torch.range(0, dx - 1)
    x = torch.min(x, dx - x)
    x = x / (dx // 2)

    y = torch.range(0, dy - 1)
    y = torch.min(y, dy - y)
    y = y / (dy // 2)

    # constructing the \xi domain
    X, Y = torch.meshgrid([y, x])
    X = X[None, None]
    Y = Y[None, None]

    # computing the scale (1 + |\xi|^2)^{s/2}
    scale = ((1 + c * (X**2 + Y**2))**(s/2)).to(device)

    # scale is a real number which scales both real and imaginary parts by multiplying
    scale = torch.stack([scale, scale], -1)

    x_fft *= scale.float()

    res = torch.ifft(x_fft, signal_ndim=2)[..., 0]

    return res


def get_constants(x_true):
    x_true = x_true
    transformed_true = sobolev_transform(x_true, C, S)
    norm = torch.norm(transformed_true.view((BATCH_SIZE, -1)), EXPONENT, keepdim=True, dim=-1)
    lamb = torch.mean(norm)

    dual_transformed_true = sobolev_transform(x_true, C, -S)
    dual_norm = torch.norm(transformed_true.view((BATCH_SIZE, -1)), DUAL_EXPONENT, keepdim=True, dim=-1)
    gamma = torch.mean(dual_norm)
    return lamb, gamma


def inf_train_gen(train):
    while True:
        for images, targets in train:
            yield images


def calc_gradient_penalty(D, real_data, fake_data, gamma, lamb):
    eps = torch.rand(BATCH_SIZE, 1).to(device)
    real_data = real_data.view((BATCH_SIZE, -1))
    fake_data = fake_data.view((BATCH_SIZE, -1))

    interpolates = (eps * real_data + ((1 - eps) * fake_data)).to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = D(interpolates.view((BATCH_SIZE, CH, IMG_SIZE, IMG_SIZE)))

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view((BATCH_SIZE, CH, IMG_SIZE, IMG_SIZE))
    dual_sobolev_gradients = sobolev_transform(gradients, C, -S)
    gradient_penalty = ((dual_sobolev_gradients.norm(2, dim=1) / gamma - 1) ** 2).mean() * lamb
    return gradient_penalty

def train():
    if pretrained:
        netG = Generator().to(device)
        netG.load_state_dict(torch.load(os.path.join(DATA_PATH, experiment_path, 'generator.pth.tar'))['state_dict'])
        netD = Discriminator().to(device)
        netD.load_state_dict(torch.load(os.path.join(DATA_PATH, experiment_path, 'discriminator.pth.tar'))['state_dict'])
    else:
        if DATASET == 'mnist':
            netG = Generator(DIM, OUTPUT_DIM).to(device)
            netD = Discriminator(DIM).to(device)
        else:
            netG = Generator().to(device)
            netD = Discriminator().to(device)
    print(netG)
    print(netD)

    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA[0], BETA[1]))
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA[0], BETA[1]))

    if DECAY:
        decay = lambda iteration: max([0., 1. - iteration / ITERS])
    else:
        decay = lambda x: 1

    schedulerD = optim.lr_scheduler.LambdaLR(optimizerD, decay)
    schedulerG = optim.lr_scheduler.LambdaLR(optimizerG, decay)

    one = torch.FloatTensor([1])[0].to(device)
    mone = (one * -1).to(device)

    # Dataset iterator
    train_gen, dev_gen = dataloader(DATASET, DATA_PATH, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    data = inf_train_gen(train_gen)

    #lib.print_model_settings(locals().copy(), os.path.join(DATA_PATH, experiment_path, 'vars.txt'))
    print('Arguments\n', args)
    print('Tensorboard logdir: {}runs/{}/{}'.format(DATA_PATH, DATASET,NAME))
    writer = SummaryWriter(log_dir=os.path.join(DATA_PATH, 'runs', DATASET, NAME))

    for iteration in range(ITERS):
        if iteration % 10000 == 0:
            print('Iteration {}, see progress on tensorboard'.format(iteration))
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for iter_d in range(CRITIC_ITERS):
            _data = next(data)
            real_data = torch.Tensor(_data).to(device)
            real_data_v = autograd.Variable(real_data)

            netD.zero_grad()

            # train with real
            D_real = netD(real_data_v)
            D_real = D_real.mean()
            D_real.backward(mone)

            # train with fake
            noise = torch.randn(BATCH_SIZE, Z_SIZE).to(device)
            noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
            fake = autograd.Variable(netG(noisev).data)
            D_fake = netD(fake.view((BATCH_SIZE, CH, IMG_SIZE, IMG_SIZE)))
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # train with gradient penalty
            lamb, gamma = get_constants(real_data)
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data, gamma, lamb)
            gradient_penalty.backward()

            D_cost = 1 / gamma * (D_fake - D_real) + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        noise = torch.randn(BATCH_SIZE, Z_SIZE).to(device)
        noisev = autograd.Variable(noise)
        fake = netG(noisev)
        G = netD(fake)
        G = G.mean() / gamma
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

        schedulerG.step()
        schedulerD.step()

        if iteration % 100 == 99:
            dev_disc_costs = []
            for images, _ in dev_gen:
                imgs = torch.Tensor(images).to(device)
                imgs_v = autograd.Variable(imgs, volatile=True)

                D = netD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)

            # logging loss to tensorboardX
            writer.add_scalar('loss/dev_disc_cost', np.mean(dev_disc_costs), iteration)

            # logging generated image to tensorboardX
            z = torch.randn((BATCH_SIZE, Z_SIZE)).to(device)
            ims = netG(z).reshape(BATCH_SIZE, CH, IMG_SIZE, IMG_SIZE)
            if DATASET.lower() == 'mnist':
                ims = torch.stack([ims, ims, ims], dim=1)
            n = 49 if BATCH_SIZE > 49 else BATCH_SIZE
            x = vutils.make_grid(ims[:n], nrow=int(np.sqrt(n)), normalize=True, range=(0, 1))
            writer.add_image('generated_{}_{}'.format(DATASET, NAME), x, iteration)
        if iteration % 1000 == 0:
            print('Iteration {}: saving models to {}{}/'.format(iteration, DATA_PATH, experiment_path))
            torch.save({'state_dict': netG.state_dict()}, os.path.join(DATA_PATH, 
                                                                       experiment_path, 'generator_{}.pth.tar'.format(iteration)))
            torch.save({'state_dict': netD.state_dict()}, os.path.join(DATA_PATH, 
                                                                       experiment_path, 'discriminator_{}.pth.tar'.format(iteration)))

    writer.close()
    # Generate dataset of images to feed them to FID scorer
    print('Generating dataset')
    lib.save_dataset(netG, BATCH_SIZE, Z_SIZE, (CH, IMG_SIZE), DATA_PATH, name=DATASET + '-' + NAME, N=N, device=device)
    # Save models
    print('Saving models to {}{}/'.format(DATA_PATH, experiment_path))
    torch.save({'state_dict': netG.state_dict()}, os.path.join(DATA_PATH, experiment_path, 'generator_final.pth.tar'))
    torch.save({'state_dict': netD.state_dict()}, os.path.join(DATA_PATH, experiment_path, 'discriminator_final.pth.tar'))

#####################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['mnist', 'cifar', 'celeba'], required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--cuda', choices=['0', '1', '2', '3', '-1'], required=True, help='-1 is for cpu mode')

parser.add_argument('--s', required=True, type=float)
parser.add_argument('--p', required=True, type=float)
parser.add_argument('--c', default=5, type=float)
parser.add_argument('--batchsize', default=60, type=int)

parser.add_argument('--iters', default=200000, type=int)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--critic', default=2, type=int, help='For WGAN and WGAN-GP, number of critic iters per gen iter')
parser.add_argument('--decay', default=False, type=bool, help='Decay learning rate')
parser.add_argument('--n', default=70000, type=int, help='Number of images to create dataset')

parser.add_argument('--pretrained', default=False, type=bool, help='Load pretrained model from datapath/tmp/dataset-name/')
parser.add_argument('--datapath', default='/home/rkhairulin/data/bwgan/', type=str) 

args = parser.parse_args()

S = args.s
EXPONENT = args.p
C = args.c
DUAL_EXPONENT = 1 / (1 - 1 / EXPONENT) if EXPONENT != 0 else np.inf
BATCH_SIZE = args.batchsize

ITERS = args.iters
LEARNING_RATE = args.lr
CRITIC_ITERS = args.critic
DECAY = args.decay
N = args.n

NAME = args.name
DATASET = args.dataset
DATA_PATH = args.datapath
pretrained = args.pretrained

if args.cuda != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = 'cuda:0'
else:
    device = 'cpu'

if DATASET.lower() == 'mnist':
    IMG_SIZE = 28
    CH = 1
    from model_mnist import Generator, Discriminator
else:
    IMG_SIZE  = 32
    CH = 3
    from model import Generator, Discriminator

# Following parameters are always the same
DIM = 64  # Model dimensionality
Z_SIZE = 128
OUTPUT_DIM = int(IMG_SIZE * IMG_SIZE * CH)
LAMBDA = 10  # Gradient penalty lambda hyperparameter
BETA = (0., 0.9)  # Betas parameters for optimizer


experiment_path = os.path.join('tmp', DATASET + '-' + NAME)
if not os.path.exists(os.path.join(DATA_PATH, experiment_path)):
    os.makedirs(os.path.join(DATA_PATH, experiment_path), exist_ok=True)

'''
if os.path.exists(os.path.join(DATA_PATH, 'runs', DATASET, NAME)):
    ans = input('You run experimnet with existing name, delete or exit? (d, e)')
    if ans == 'd':
        os.system('rm -rf {}'.format(os.path.join(DATA_PATH, 'runs', DATASET, NAME)))
    elif ans == 'e':
        sys.exit()
'''
train()

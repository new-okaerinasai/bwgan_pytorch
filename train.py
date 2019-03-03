import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import lib

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data
from torchvision import transforms
import torchvision

from dataloader import *


# Adjusted by user
# TODO: make argparse

C = 2
S = -2
EXPONENT = 2
DUAL_EXPONENT = 1 / (1 - 1 / EXPONENT) if EXPONENT != 0 else np.inf
use_cuda = False

if use_cuda:
    gpu = 0

MNIST = True

if MNIST:
    from model_mnist import *
else:
    from model import *


NAME = 'exp6'
DIM = 64 # Model dimensionality
Z_SIZE = 128
BATCH_SIZE = 60 # Batch size
CRITIC_ITERS = 2 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
BETA = (0., 0.9) # Betas parameters for optimizer
DECAY = False # Decay learning rate
LEARNING_RATE = 2e-4

if not os.path.exists('tmp/mnist-' + NAME):
    os.system("mkdir -p tmp/mnist-" + NAME)
else:
    os.system('rm tmp/mnist-' + NAME + '/*')
if not os.path.exists('tmp/images-' + NAME):
    os.system('mkdir -p tmp/images-' + NAME)
else:
    os.system('rm tmp/images-' + NAME + '/*')

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
    scale = (1 + c * (X**2 + Y**2))**(s/2)

    # scale is a real number which scales both real and imaginary parts by multiplying
    scale = torch.stack([scale, scale], -1)
    
    x_fft *= scale.float()
    
    res = torch.ifft(x_fft, signal_ndim=2)[..., 0]
    
    return res


def show_image(G, nrows, ncols, path="./tmp/",):
    z = torch.randn((BATCH_SIZE, Z_SIZE)).cuda()
    ims = G(z)
    plt.figure(figsize=(10,8))
    for i in range(nrows*ncols):
        plt.subplot(nrows, ncols, i + 1)
        i = np.random.randint(0, BATCH_SIZE)
        im = ims[i].cpu().detach().numpy()
        plt.imshow(im.reshape((28, 28)) * 0.5 + 0.5)
    plt.savefig(path+"res{}.pdf".format(int(START_EXP - time.time())))
    plt.show()

def get_constants(x_true):
    x_true = x_true
    transformed_true = sobolev_transform(x_true, C, S)
    norm = torch.norm(transformed_true.view((BATCH_SIZE, -1)), EXPONENT, keepdim=True, dim=-1)
    lamb = torch.mean(norm)

    dual_transformed_true = sobolev_transform(x_true, C, -S)
    dual_norm = torch.norm(transformed_true.view((BATCH_SIZE, -1)), DUAL_EXPONENT, keepdim=True, dim=-1)
    gamma = torch.mean(dual_norm)
    return lamb, gamma


# Dataset iterator
train_gen, dev_gen = dataloader('mnist', batch_size=BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen:
            yield images

def calc_gradient_penalty(netD, real_data, fake_data, gamma, lamb):
    eps = torch.rand(BATCH_SIZE, 1)
    real_data = real_data.view((BATCH_SIZE, -1))
    eps = eps.cuda(gpu) if use_cuda else eps
    
    interpolates = eps * real_data + ((1 - eps) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view((BATCH_SIZE, 1, 28, 28))
    dual_sobolev_gradients = sobolev_transform(gradients, C, -S)
    gradient_penalty = ((dual_sobolev_gradients.norm(2, dim=1) / gamma - 1) ** 2).mean() * lamb
    return gradient_penalty

netG = Generator(DIM, OUTPUT_DIM)
netD = Discriminator(DIM)
print(netG)
print(netD)

if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA[0], BETA[1]))
optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA[0], BETA[1]))

if DECAY:
    decay = lambda iteration : max([0., 1.- iteration/ITERS])
else:
    decay = lambda x: 1

shedulerD = optim.lr_scheduler.LambdaLR(optimizerD, decay)
shedulerG = optim.lr_scheduler.LambdaLR(optimizerG, decay)

one = torch.FloatTensor([1])[0]
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

data = inf_train_gen()

START_EXP = time.time()

for iteration in range(ITERS):
    start_time = time.time()

    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in range(CRITIC_ITERS):
        _data = next(data)
        real_data = torch.Tensor(_data)
        if use_cuda:
            real_data = real_data.cuda(gpu)
        real_data_v = autograd.Variable(real_data)

        netD.zero_grad()

        # train with real
        D_real = netD(real_data_v)
        D_real = D_real.mean()
        # print D_real
        D_real.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        fake = autograd.Variable(netG(noisev).data)
        inputv = fake
        D_fake = netD(inputv)
        D_fake = D_fake.mean()
        D_fake.backward(one)

        # train with gradient penalty
        lamb, gamma = get_constants(real_data)
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data, gamma, lamb)
        gradient_penalty.backward()

        D_cost = 1 / gamma *(D_fake - D_real) + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()

    for p in netD.parameters():
        p.requires_grad = False
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake = netG(noisev)
    G = netD(fake)
    G = G.mean() / gamma
    G.backward(mone)
    G_cost = -G
    optimizerG.step()
    
    shedulerG.step()
    shedulerD.step()
    print("iteration", iteration)
    if iteration % 100 == 99:
        dev_disc_costs = []
        for images,_ in dev_gen:
            imgs = torch.Tensor(images)
            if use_cuda:
                imgs = imgs.cuda(gpu)
            imgs_v = autograd.Variable(imgs, volatile=True)

            D = netD(imgs_v)
            _dev_disc_cost = -D.mean().cpu().data.numpy()
            dev_disc_costs.append(_dev_disc_cost)
        lib.plot.plot('tmp/mnist-' + NAME + '/dev disc cost', np.mean(dev_disc_costs))

        show_image(netG, 10, 10, path='./tmp/images-' + NAME + '/')


# Generate dataset of images to feed them to FID scorer

N = 70000
save_dataset(netG, N, NAME=NAME)

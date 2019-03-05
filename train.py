import os
import sys
import time
import random
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
# from IPython.display import clear_output
from dataloader import *
from lib import *

sys.path.append(os.getcwd())
# Adjusted by user
# TODO: make argparse

C = 2
S = -2
EXPONENT = 2
DUAL_EXPONENT = 1 / (1 - 1 / EXPONENT) if EXPONENT != 0 else np.inf
use_cuda = False
pretrained = False

if use_cuda:
    gpu = 0

NAME = 'exp0'
DATASET = 'mnist'
DATA_PATH = '/home/rkhairulin/data/bwgan/'

if DATASET.lower() == 'mnist':
    IMG_SIZE = 28
    CH = 1
    from model_mnist import *
else:
    IMG_SIZE  = 32
    CH = 3
    from model import *

DIM = 64  # Model dimensionality
Z_SIZE = 128
BATCH_SIZE = 60
OUTPUT_DIM = int(IMG_SIZE * IMG_SIZE)
CRITIC_ITERS = 2  # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10  # Gradient penalty lambda hyperparameter
ITERS = 200000  # How many generator iterations to train for
BETA = (0., 0.9)  # Betas parameters for optimizer
DECAY = False  # Decay learning rate
LEARNING_RATE = 2e-4
N = 70000  # Number of images to create dataset

if not os.path.exists(os.path.join(DATA_PATH, 'tmp')):
    os.mkdir(os.path.join(DATA_PATH, 'tmp'))

experiment_path = os.path.join('tmp', DATASET + '-' + NAME)
if not os.path.exists(os.path.join(DATA_PATH, experiment_path)):
    os.mkdir(os.path.join(DATA_PATH, experiment_path))
else:
    os.system('rm {}'.format(os.path.join(DATA_PATH, experiment_path, '*')))


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


def get_constants(x_true):
    x_true = x_true
    transformed_true = sobolev_transform(x_true, C, S)
    norm = torch.norm(transformed_true.view((BATCH_SIZE, -1)), EXPONENT, keepdim=True, dim=-1)
    lamb = torch.mean(norm)

    dual_transformed_true = sobolev_transform(x_true, C, -S)
    dual_norm = torch.norm(transformed_true.view((BATCH_SIZE, -1)), DUAL_EXPONENT, keepdim=True, dim=-1)
    gamma = torch.mean(dual_norm)
    return lamb, gamma


def inf_train_gen():
    while True:
        for images, targets in train_gen:
            yield images


def calc_gradient_penalty(D, real_data, fake_data, gamma, lamb):
    eps = torch.rand(BATCH_SIZE, 1)
    real_data = real_data.view((BATCH_SIZE, -1))
    eps = eps.cuda(gpu) if use_cuda else eps

    interpolates = eps * real_data + ((1 - eps) * fake_data)
    interpolates = interpolates.cuda(gpu) if use_cuda else interpolates
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view((BATCH_SIZE, CH, IMG_SIZE, IMG_SIZE))
    dual_sobolev_gradients = sobolev_transform(gradients, C, -S)
    gradient_penalty = ((dual_sobolev_gradients.norm(2, dim=1) / gamma - 1) ** 2).mean() * lamb
    return gradient_penalty

if pretrained:
    netG = Generator()
    netG.load_state_dict(torch.load(os.path.join(DATA_PATH, experiment_path, 'generator.pth.tar'))['state_dict'])
    netD = Discriminator()
    netD.load_state_dict(torch.load(os.path.join(DATA_PATH, experiment_path, 'discriminator.pth.tar'))['state_dict'])
else:
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
    decay = lambda iteration: max([0., 1. - iteration / ITERS])
else:
    decay = lambda x: 1

shedulerD = optim.lr_scheduler.LambdaLR(optimizerD, decay)
shedulerG = optim.lr_scheduler.LambdaLR(optimizerG, decay)

one = torch.FloatTensor([1])[0]
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

# Dataset iterator
train_gen, dev_gen = dataloader(DATASET, DATA_PATH, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
data = inf_train_gen()

print_model_settings(locals().copy(), os.path.join(DATA_PATH, experiment_path, 'vars.txt'))
writer = SummaryWriter(log_dir=os.path.join('runs', DATASET, NAME))

for iteration in range(ITERS):
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
        D_real.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, Z_SIZE)
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

        D_cost = 1 / gamma * (D_fake - D_real) + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()

    for p in netD.parameters():
        p.requires_grad = False
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, Z_SIZE)
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
        for images, _ in dev_gen:
            imgs = torch.Tensor(images)
            if use_cuda:
                imgs = imgs.cuda(gpu)
            imgs_v = autograd.Variable(imgs, volatile=True)

            D = netD(imgs_v)
            _dev_disc_cost = -D.mean().cpu().data.numpy()
            dev_disc_costs.append(_dev_disc_cost)

        # logging loss to tensorboardX
        writer.add_scalar('loss/dev_disc_cost', np.mean(dev_disc_costs), iteration)

        # logging generated image to tensorboardX
        z = torch.randn((BATCH_SIZE, Z_SIZE))
        z = z.cuda() if use_cuda else z
        ims = netG(z).reshape(BATCH_SIZE, CH, IMG_SIZE, IMG_SIZE)
        if DATASET.lower() == 'mnist':
            ims = torch.stack([ims, ims, ims], dim=1)
        n = 49 if BATCH_SIZE > 49 else BATCH_SIZE
        x = vutils.make_grid(ims[:n], nrow=int(np.sqrt(n)), normalize=True, range=(0, 1))
        writer.add_image('generated_{}_{}'.format(DATASET, NAME), x, iteration)

        # generate_sample(netG, BATCH_SIZE, Z_SIZE)
        # clear_output()

writer.close()
# Generate dataset of images to feed them to FID scorer
save_dataset(netG, BATCH_SIZE, Z_SIZE, IMG_SIZE, DATA_PATH, name=DATASET + '-' + NAME, N=N)
# Save models
torch.save({'state_dict': netG.state_dict()}, os.path.join(DATA_PATH, experiment_path, 'generator.pth.tar'))
torch.save({'state_dict': netD.state_dict()}, os.path.join(DATA_PATH, experiment_path, 'discriminator.pth.tar'))

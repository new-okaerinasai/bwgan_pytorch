import os
import sys

sys.path.append(os.getcwd())
import argparse

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import tqdm

from dataloader import dataloader
import lib


def sobolev_transform(x, c=5, s=1, args=None):
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
    scale = ((1 + c * (X ** 2 + Y ** 2)) ** (s / 2)).to(args.device)

    # scale is a real number which scales both real and imaginary parts by multiplying
    scale = torch.stack([scale, scale], -1)

    x_fft *= scale.float()

    res = torch.ifft(x_fft, signal_ndim=2)[..., 0]

    return res


def get_constants(x_true, args):
    x_true = x_true
    transformed_true = sobolev_transform(x_true, args.c, args.s, args)
    norm = torch.norm(transformed_true.view((args.batch_size, -1)), args.p, keepdim=True, dim=-1)
    lamb = torch.mean(norm)

    dual_transformed_true = sobolev_transform(x_true, args.c, -args.s, args)
    dual_norm = torch.norm(transformed_true.view((args.batch_size, -1)), args.dual_exponent, keepdim=True, dim=-1)
    gamma = torch.mean(dual_norm)
    return lamb, gamma


def inf_train_gen(train):
    while True:
        for images, targets in train:
            yield images


def calc_gradient_penalty(D, real_data, fake_data, gamma, lamb, args):
    eps = torch.rand(args.batch_size, 1).to(args.device)
    real_data = real_data.view((args.batch_size, -1))
    fake_data = fake_data.view((args.batch_size, -1))

    interpolates = (eps * real_data + ((1 - eps) * fake_data)).to(args.device)
    interpolates.requires_grad = True
    # interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = D(interpolates.view((args.batch_size, args.n_channels, args.img_size, args.img_size)))

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(args.device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view((args.batch_size, args.n_channels, args.img_size, args.img_size))
    dual_sobolev_gradients = sobolev_transform(gradients, args.c, -args.s, args)
    gradient_penalty = ((dual_sobolev_gradients.norm(2, dim=1) / gamma - 1) ** 2).mean() * lamb
    return gradient_penalty


def train(args):
    if args.verbose:
        range_type = tqdm.trange
    else:
        range_type = range

    if args.pretrained:
        netG = Generator().to(args.device)
        netG.load_state_dict(
            torch.load(os.path.join(args.data_path, experiment_path, 'generator.pth.tar'))['state_dict'])
        netD = Discriminator().to(args.device)
        netD.load_state_dict(
            torch.load(os.path.join(args.data_path, experiment_path, 'discriminator.pth.tar'))['state_dict'])
    else:
        if args.dataset == 'mnist':
            netG = Generator(args.dim, args.output_dim).to(args.device)
            netD = Discriminator(args.dim).to(args.device)
        else:
            print(f"Training on {args.device}")
            netG = Generator().to(args.device)
            netD = Discriminator().to(args.device)
    print(netG)
    print(netD)

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=args.betas)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=args.betas)

    if args.decay:
        decay = lambda iteration: max([0., 1. - iteration / args.iters])
    else:
        decay = lambda x: 1

    schedulerD = optim.lr_scheduler.LambdaLR(optimizerD, decay)
    schedulerG = optim.lr_scheduler.LambdaLR(optimizerG, decay)

    one = torch.FloatTensor([1])[0].to(args.device)
    mone = (one * -1).to(args.device)

    # Dataset iterator
    train_gen, dev_gen = dataloader(args.dataset, args.data_path, batch_size=args.batch_size, img_size=args.img_size)
    data = inf_train_gen(train_gen)

    # lib.print_model_settings(locals().copy(), os.path.join(args.data_path, experiment_path, 'vars.txt'))
    print('Arguments\n', args)
    print('Tensorboard logdir: {}runs/{}/{}'.format(args.data_path, args.dataset, args.name))
    log_dir = args.log_dir if args.log_dir else os.path.join(args.data_path, 'runs', args.dataset, args.name)
    print(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    for iteration in range_type(args.iters):
        if iteration % 10000 == 0:
            print('Iteration {}, see progress on tensorboard'.format(iteration))
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for iter_d in range(args.critic):
            _data = next(data)
            real_data = torch.Tensor(_data).to(args.device)
            # real_data_v = autograd.Variable(real_data)

            netD.zero_grad()

            # train with real
            D_real = netD(real_data)
            D_real = -D_real.mean()
            # D_real.backward()

            # train with fake
            noise = torch.randn(args.batch_size, args.z_size).to(args.device)
            # noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
            noise.requires_grad = False

            fake = netG(noise)
            D_fake = netD(fake.view((args.batch_size, args.n_channels, args.img_size, args.img_size)))
            D_fake = D_fake.mean()
            # D_fake.backward()

            # train with gradient penalty
            lamb, gamma = get_constants(real_data, args)
            gradient_penalty = calc_gradient_penalty(netD, real_data, fake.data, gamma, lamb, args)
            # gradient_penalty.backward()

            D_cost = 1 / gamma * (D_fake - D_real) + gradient_penalty
            Wasserstein_D = D_real - D_fake
            loss = (D_real - D_fake) + args.lambd * gradient_penalty
            loss.backward()
            optimizerD.step()

        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        noise = torch.randn(args.batch_size, args.z_size).to(args.device)
        # noisev = autograd.Variable(noise)
        fake = netG(noise)
        G = netD(fake)
        G = G.mean() / gamma
        # G.backward()
        G_cost = -G
        G_cost.backward()
        optimizerG.step()

        schedulerG.step()
        schedulerD.step()

        if iteration % 100 == 99:
            dev_disc_costs = []
            for images, _ in dev_gen:
                imgs = torch.Tensor(images).to(args.device)
                imgs.requires_grad = False
                # imgs_v = autograd.Variable(imgs, volatile=True)

                D = netD(imgs)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)

            # logging loss to tensorboardX
            writer.add_scalar('loss/dev_disc_cost', np.mean(dev_disc_costs), iteration)

            # logging generated image to tensorboardX
            z = torch.randn((args.batch_size, args.z_size)).to(args.device)
            ims = netG(z).reshape(args.batch_size, args.n_channels, args.img_size, args.img_size)
            if args.dataset.lower() == 'mnist':
                ims = torch.stack([ims, ims, ims], dim=1)
            n = 49 if args.batch_size > 49 else args.batch_size
            x = vutils.make_grid(ims[:n], nrow=int(np.sqrt(n)), normalize=True, range=(0, 1))
            writer.add_image('generated_{}_{}'.format(args.dataset, args.name), x, iteration)
        if iteration % 1000 == 0:
            print('Iteration {}: saving models to {}{}/'.format(iteration, args.data_path, experiment_path))
            torch.save({'state_dict': netG.state_dict()}, os.path.join(args.data_path,
                                                                       experiment_path,
                                                                       'generator_{}.pth.tar'.format(iteration)))
            torch.save({'state_dict': netD.state_dict()}, os.path.join(args.data_path,
                                                                       experiment_path,
                                                                       'discriminator_{}.pth.tar'.format(iteration)))

    writer.close()
    # Generate dataset of images to feed them to FID scorer
    print('Generating dataset')
    lib.save_dataset(netG, args.batch_size, args.z_size, (args.n_channels, args.img_size), args.data_path,
                     name=args.dataset + '-' + args.name, N=args.n_images, device=args.device)
    # Save models
    print('Saving models to {}{}/'.format(args.data_path, experiment_path))
    torch.save({'state_dict': netG.state_dict()},
               os.path.join(args.data_path, experiment_path, 'generator_final.pth.tar'))
    torch.save({'state_dict': netD.state_dict()},
               os.path.join(args.data_path, experiment_path, 'discriminator_final.pth.tar'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dataset', choices=['mnist', 'cifar', 'celeba'], required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--cuda', choices=['0', '1', '2', '3', '-1'], required=True, help='-1 is for cpu mode')

    parser.add_argument('--s', required=True, type=float)
    parser.add_argument('--p', required=True, type=float)
    parser.add_argument('--c', default=5, type=float)
    parser.add_argument('--batch_size', default=60, type=int)

    parser.add_argument('--iters', default=200000, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--critic', default=2, type=int,
                        help='For WGAN and WGAN-GP, number of critic iters per gen iter')
    parser.add_argument('--decay', default=False, type=bool, help='Decay learning rate')
    parser.add_argument('--n_images', default=70000, type=int, help='Number of images to create dataset')

    parser.add_argument('--pretrained', default=False, type=bool,
                        help='Load pretrained model from datapath/tmp/dataset-name/')
    parser.add_argument('--data_path', default='/home/rkhairulin/data/bwgan/', type=str)

    parser.add_argument('--logspath', default=None, type=str)
    args = parser.parse_args()

    if args.dataset.lower() == 'mnist':
        args.img_size = 28
        args.n_channels = 1
        from model_mnist import Generator, Discriminator
    else:
        args.img_size = 32
        args.n_channels = 3
        from model import Generator, Discriminator

    args.dual_exponent = 1 / (1 - 1 / args.p) if args.p != 0 else np.inf

    args.log_dir = args.logspath if args.logspath else os.path.join(args.data_path, 'runs', args.dataset, args.name)

    # Following parameters are always the same
    args.dim = 64  # Model dimensionality
    args.z_size = 128
    args.output_dim = int(args.img_size * args.img_size * args.n_channels)
    args.lambd = 10  # Gradient penalty lambda hyperparameter
    args.betas = (0., 0.9)  # Betas parameters for optimizer

    if int(args.cuda) != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
        args.device = 'cuda:0'
    else:
        args.device = 'cpu'

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.dataset.lower() == "mnist":
        from model_mnist import Generator, Discriminator
    else:
        from model import Generator, Discriminator

    experiment_path = os.path.join('tmp', args.dataset + '-' + args.name)
    os.system("mkdir -p {}".format(os.path.join(args.data_path, experiment_path)))

    if os.path.exists(os.path.join(args.data_path, 'runs', args.dataset, args.name)):
        ans = input('You run experiment with existing name, delete logs or exit? (d / e)')
        while ans not in ('d', 'e'):
            ans = input('Incorrect input, delete or exit? (d / e)')
        if ans == 'd':
            os.system('rm -rf {}'.format(args.log_dir))
        elif ans == 'e':
            sys.exit()

    train(args)

import os
import random

import torch
import scipy
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import trange


def save_dataset(G, batch_size, z_size, img_size, path, name=None, N=70000, device='cpu'):
    """
    Generate dataset of size N with generator model G and save it

    :param G: generator model
    :param batch_size: model parameter
    :param z_size: model parameter
    :param img_size: image size tuple (CH, H, W) or (CH, H) if square
    :param path: path to save
    :param name: experiment name to specify folder
    :param N: number of images to generate
    """

    if name is not None:
        path = os.path.join(path, 'dataset-' + name)
    iters = N // batch_size if N >= batch_size else 1
    if len(img_size) == 2:
        img_size = (img_size[0], img_size[1], img_size[1])
    os.makedirs(path, exist_ok=True)
    for k in tqdm.trange(iters):
        z = torch.randn((batch_size, z_size)).to(device)
        images = G(z)
        for i in range(batch_size):
            image = images[i].cpu().detach().numpy().reshape(img_size).transpose((1, 2, 0))
            scipy.misc.toimage(0.5 * image + 0.5, cmin=0.0, cmax=1.0).save(os.path.join(path, 'out{}.png'.format(k * batch_size + i)))


def show_sample(path_src, nrows=7, ncols=7, path_sv=None, name='sample.png'):
    """
    Show random sample from dataset folder (save if path_cv specified)

    :param paths_src: path to dataset images source
    :param nrows/ncols: size of grid
    :param path_sv: path to save sample image
    :param name: filename with specified extension (pdf, png, jpg)
    """
    n = nrows * ncols
    ids = random.sample(os.listdir(path_src), n)

    plt.figure(figsize=(9, 9))
    gs1 = gridspec.GridSpec(nrows, ncols)
    gs1.update(wspace=0.05, hspace=0.05)
    for i in range(n):
        ax1 = plt.subplot(gs1[i])
        image = scipy.misc.imread(os.path.join(path_src, ids[i]))
        ax1.imshow(image)
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
    if path_sv is not None:
        plt.savefig(os.path.join(path_sv, name))
    plt.show()


def generate_sample(G, batch_size, z_size, img_size, nrows=7, ncols=7, path=None, name='sample.png', device='cpu'):
    """
    Generate sample with generator model G and show it (save if path specified)

    :param G: generator model
    :param batch_size: model parameter
    :param z_size: model parameter
    :param img_size: image size tuple (CH, H, W) or (CH, H) if square
    :param nrows/ncols: size of grid
    :param path: path to save
    :param name: filename with specified extension (pdf, png, jpg)
    """
    z = torch.randn((batch_size, z_size)).to(device)
    images = G(z)

    n = nrows * ncols
    if len(img_size) == 2:
        img_size = (img_size[0], img_size[1], img_size[1])
    ids = random.sample(range(batch_size), n)

    plt.figure(figsize=(9, 9))
    gs1 = gridspec.GridSpec(nrows, ncols)
    gs1.update(wspace=0.05, hspace=0.05)

    for i in range(n):
        ax1 = plt.subplot(gs1[i])
        image = images[i].cpu().detach().numpy().reshape(img_size).transpose((1, 2, 0))
        ax1.imshow(0.5 * image + 0.5)
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
    if path is not None:
        plt.savefig(os.path.join(path, name))
    plt.show()

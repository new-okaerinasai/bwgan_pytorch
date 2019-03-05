import os
import random
import torch
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def print_model_settings(locals_, file_path):
    print("Uppercase local vars:")
    all_vars = [(k, v) for (k, v) in list(locals_.items()) if (k.isupper() and k != 'T'
                                                               and k != 'SETTINGS'
                                                               and k != 'ALL_SETTINGS')]
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        if 'module' in str(var_value): continue
        print("{}: {}".format(var_name, var_value))
        print("{}: {}".format(var_name, var_value), file=open(file_path, "a"))


def save_dataset(G, batch_size, z_size, img_size, path, name=None, N=70000, use_cuda=False):
    """
    Generate dataset of size N with generator model G and save it

    :param G: generator model
    :param batch_size: model parameter
    :param z_size: model parameter
    :param s: image size tuple (H, W) or int if H=W
    :param path: path to save
    :param name: experiment name to specify folder
    :param N: number of images to generate
    """

    if name is not None:
        path = os.path.join(path, 'dataset-' + name)
    iters = N // batch_size if N >= batch_size else 1
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    if not os.path.exists(path):
        os.mkdir(path)
    k = 0
    for _ in range(iters):
        z = torch.randn((batch_size, z_size))
        z = z.cuda() if use_cuda else z
        images = G(z)
        for i in range(batch_size):
            image = images[i].cpu().detach().numpy().reshape(img_size)
            scipy.misc.toimage(image, cmin=0.0, cmax=1.0).save(os.path.join(path, 'out{}.png'.format(k)))
            k += 1


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


def generate_sample(G, batch_size, z_size, s, nrows=7, ncols=7, path=None, name='sample.png', use_cuda=False):
    """
    Generate sample with generator model G and show it (save if path specified)

    :param G: generator model
    :param batch_size: model parameter
    :param z_size: model parameter
    :param s: image size tuple (H, W) or int if H=W
    :param nrows/ncols: size of grid
    :param path: path to save
    :param name: filename with specified extension (pdf, png, jpg)
    """
    z = torch.randn((batch_size, z_size))
    z = z.cuda() if use_cuda else z
    images = G(z)

    n = nrows * ncols
    if isinstance(s, int):
        s = (s, s)
    ids = random.sample(range(batch_size), n)

    plt.figure(figsize=(9, 9))
    gs1 = gridspec.GridSpec(nrows, ncols)
    gs1.update(wspace=0.05, hspace=0.05)

    for i in range(n):
        ax1 = plt.subplot(gs1[i])
        image = images[i].cpu().detach().numpy().reshape(s)
        ax1.imshow(image)
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
    if path is not None:
        plt.savefig(os.path.join(path, name))
    plt.show()

import numpy as np
import os
import random

import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import pickle as pickle

from tqdm import tqdm
import scipy

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():
    _iter[0] += 1

def plot(name, value):
    _since_last_flush[name][_iter[0]] = value

def flush():
    prints = []

    for name, vals in list(_since_last_flush.items()):
        prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
        _since_beginning[name].update(vals)

        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(name.replace(' ', '_')+'.jpg')

    print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
    _since_last_flush.clear()

    with open('log.pkl', 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)


def save_dataset(netG, N=70000, path_sv='/home/rkhairulin/data/bwgan/dataset-', NAME='exp0'):
    # generates datast with generator netG
    path = path_sv + NAME
    iters = N // BATCH_SIZE
    if not os.path.exists(path):
        os.mkdir(path)

    k = 0
    for _ in tqdm(range(iters)):
        z = torch.randn((BATCH_SIZE, Z_SIZE)).cuda()
        ims = netG(z)
        for i in range(BATCH_SIZE):
            image = (ims[i].cpu().detach().numpy()).reshape(28,28)
            #image = np.stack([image, image, image], axis=-1)
            scipy.misc.toimage(image, cmin=0.0, cmax=1.0).save(path + '/out{}.png'.format(k))
            k += 1

def save_to_pdf(path_src, path_sv, name='result.pdf', nrows=7, ncols=7):
    # draw samples from dataset
    n = nrows * ncols
    N = len(os.listdir(path_src))
    ids = random.sample(range(N), n)
    
    plt.figure(figsize=(9,9))
    gs1 = gridspec.GridSpec(nrows, ncols)
    gs1.update(wspace=0.05, hspace=0.05) 
    
    for i in range(n):
        ax1 = plt.subplot(gs1[i])
        image = scipy.misc.imread('{}/out{}.png'.format(path_src, ids[i]))
        ax1.imshow(image)
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
    plt.savefig('{}/{}'.format(path_sv, name))
    plt.show()

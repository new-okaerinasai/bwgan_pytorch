import os
import argparse

import torch
import scipy.misc
import numpy as np

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help='Path to the dataset. ')
parser.add_argument('--cuda', choices=['0', '1', '2', '3', '-1'], required=True, help='CUDA id, -1 is for cpu mode')
args = parser.parse_args()

if args.cuda != '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = 'cuda:0'
else:
    device = 'cpu'
path = args.path

images = np.empty((len(os.listdir(path)), 32, 32, 3))

n = 0
for i, fname in enumerate(os.listdir(path)):
    if fname.endswith('png') or fname.endswith('jpg'):
        images[i] = 2 * (scipy.misc.imread(os.path.join(path, fname)) / 255) - 1
        n += 1
images = images[:n].transpose((0,3,1,2))

data = torch.utils.data.Dataset(torch.tensor(images))
mean, std = utils.inception_score(data, device=device, batch_size=32, resize=True, splits=10)
print('Inception score: {0:.3f} +-{0:.3f}'.format(mean, std))
with open(os.path.join(path, 'inception.txt'), 'w') as f:
    f.write('Inception score: {0:.3f} +-{0:.3f}'.format(mean, std))

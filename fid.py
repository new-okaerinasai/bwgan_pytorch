import os
import argparse

import torch
import numpy as np

import utils
import lib
from model import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--original', required=True, help='Path to the original dataset. ')
parser.add_argument('--generated', required=True, help=('Path to generated dataset directory. '
                                                       'Path to store generated dataset if "models" passed'))
parser.add_argument('--models', default=None, help=('Path to trained models directory, takes the last one. '
                                                    'Pass if want to generate dataset to path "generated/" '))
parser.add_argument('--cuda', choices=['0', '1', '2', '3', '-1'], required=True, help='CUDA id, -1 is for cpu mode')
parser.add_argument('--n', default=40000, type=int, help='Number of images to create dataset')
parser.add_argument('--dims', default=2048, type=int, help=('Dimensionality of Inception features to use. '
                                                            'By default, uses pool3 features'))
args = parser.parse_args()

N = args.n
dims = args.dims
path_original = args.original
path_generated = args.generated
if args.cuda != '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = 'cuda:0'
else:
    device = 'cpu'

if len(os.listdir(path_generated)) == 0:
    if args.models is not None:
        itr = sorted([int(name.split('.')[0].split('_')[1]) for name in os.listdir(args.models)])[-1]
        netG = Generator().to(device)
        netG.load_state_dict(torch.load(os.path.join(args.models, 'generator_{}.pth.tar'.format(itr)))['state_dict'])
        print('Generating dataset...')
        lib.save_dataset(netG, 60, 128, (3, 32), path=path_generated, N=N, device=device)
    else:
        raise ValueError('Path generated data is empty, pass "models" argument or correct path')

if 'statistic.npz' not in os.listdir(path_original):
    print('Computing statistics of original dataset...')
    m_celeba, s_celeba = utils._compute_statistics_of_path(path_original, dims, device)
    np.savez(os.path.join(path_original, 'statistic.npz'), mu=m_celeba, sigma=s_celeba)

fid_score = utils.calculate_fid_given_paths([path_generated, os.path.join(path_original, 'statistic.npz')], device, dims)
print('FID score = ', fid_score)
with open(os.path.join(path_generated, 'fid.txt'), 'w') as f:
    f.write('FID_score: {}'.format(fid_score))

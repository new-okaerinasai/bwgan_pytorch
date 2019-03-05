import torchvision
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import torch
import os
import random
random.seed(1234)

class CustomTransform():
    def __init__(self):
        pass
    def __call__(self, image):
        return (image - 0.5) / 0.5


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, path, transform, mode, test_size=2000):
        self.images = os.path.join(path, 'images')
        self.attr_path = os.path.join(path, 'list_attr_celeba.txt')
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.a2id = {}
        self.id2a = {}
        self.test_size = test_size
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        self.num_files = int(lines[0])
        all_attr_names = lines[1].split()
        self.attrs = all_attr_names
        for i, attr_name in enumerate(all_attr_names):
            self.a2id[attr_name] = i
            self.id2a[i] = attr_name

        lines = lines[2:]
        random.shuffle(lines)
        if isinstance(self.test_size, float):
            self.test_size = int(self.num_files * self.test_size)

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            label = []
            for attr_name in self.attrs:
                idx = self.a2id[attr_name]
                label.append(values[idx] == '1')

            if (i + 1) < self.test_size:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.images, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_images

def load_celeba(path):
    path_celeba = os.path.join(path, 'celeba')
    if not os.path.exists(path_celeba):
        os.mkdir(path_celeba)
    if not os.path.exists(os.path.join(path_celeba, 'images')):
        print('Downloading CelebA to ' + path_celeba)
        os.system('wget https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0 -O {}'.format(os.path.join(path,'celeba.zip')))
        os.system('unzip {} -d {}/'.format(os.path.join(path,'celeba.zip'), path))
        os.remove(os.path.join(path,'celeba.zip'))
    else:
        print('CelebA set already exists in ' + path_celeba)


def dataloader(name, path, batch_size=128, img_size=32, num_workers=8):
    """ 
    Return train_loader, test_loader (torch.utils.data.DataLoader objects)

    :param name: name of dataset to load (cifar, celeba, mnist)
    :param path: path to save dataset
    :param batch_size: how many samples per batch to load
    :param num_workers: how many subprocesses to use for data loading
    """
    # ----------------------------------------------
    if name.lower() == 'cifar':
        path_cifar = os.path.join(path, 'cifar')
        if not os.path.exists(path_cifar):
            os.mkdir(path_cifar)
            
        transform_cifar = transforms.Compose([
            transforms.ToTensor(),
            CustomTransform()
        ])

        cifar_set = torchvision.datasets.CIFAR10(root=path_cifar, train=True, download=True, transform=transform_cifar)
        train_loader_cifar = torch.utils.data.DataLoader(cifar_set, batch_size=batch_size, shuffle=True, 
                                                         num_workers=num_workers, pin_memory=True, drop_last=True)
        cifar_testset = torchvision.datasets.CIFAR10(root=path_cifar, train=False, download=True, transform=transform_cifar)
        test_loader_cifar = torch.utils.data.DataLoader(cifar_testset, batch_size=batch_size, shuffle=False, 
                                                         num_workers=num_workers, pin_memory=True, drop_last=True)
        return train_loader_cifar, test_loader_cifar

    # ----------------------------------------------
    elif name.lower() == 'celeba':
        load_celeba(path=path)
        path_celeba = os.path.join(path, 'celeba')
        
        transform_celeba = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            CustomTransform()
        ])

        celeba_set = CelebA(path_celeba, transform_celeba, mode='train', test_size=2000)
        train_loader_celeba = torch.utils.data.DataLoader(celeba_set, batch_size=batch_size, shuffle=True, 
                                                          num_workers=num_workers, pin_memory=True, drop_last=True)
        celeba_testset = CelebA(path_celeba, transform_celeba, mode='test', test_size=2000)
        test_loader_celeba = torch.utils.data.DataLoader(celeba_testset, batch_size=batch_size, shuffle=False, 
                                                          num_workers=num_workers, pin_memory=True, drop_last=True)
        return train_loader_celeba, test_loader_celeba
    
    # ----------------------------------------------
    elif name.lower() == 'mnist':
        path = os.path.join(path, 'mnist')
        transform_mnist = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        mnist_set = torchvision.datasets.MNIST(path, download=True, train=True, transform=transform_mnist)
        train_loader_mnist = torch.utils.data.DataLoader(mnist_set, batch_size=batch_size, shuffle=True)
        
        mnist_testset = torchvision.datasets.MNIST(path, download=True, train=False, transform=transform_mnist)
        test_loader_mnist = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=True)
        
        return train_loader_mnist, test_loader_mnist

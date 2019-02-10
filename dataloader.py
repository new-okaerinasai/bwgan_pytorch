import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch
import os


def datasets(path="data/bwgan", batch_size=128, img_size=64, num_workers=8):
    '''
    return (cifar_dataloader, celeba_dataloader)
    '''

    NUM_WORKERS = num_workers
    BATCH_SIZE = batch_size
    img_size = img_size

    path_cifar = path + "/cifar"
    path_celeba = path + "/celeba"

    if not os.path.exists(path_cifar):
        os.mkdir(path_cifar)
    if not os.path.exists(path_celeba):
        os.mkdir(path_celeba)

    if not os.path.exists(path_celeba + "/images"):
        print("Downloading CelebA to " + path_celeba)
        os.system("wget https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0 -O {}/celeba.zip".format(path))
        os.system("unzip {}/celeba.zip -d {}/".format(path, path))
        os.remove("{}/celeba.zip".format(path))
    else:
        print("CelebA set already exists in " + path_celeba)

    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar_set = torchvision.datasets.CIFAR10(root=path_cifar, train=True, download=True, transform=transform_cifar)
    train_loader_cifar = torch.utils.data.DataLoader(cifar_set, batch_size=BATCH_SIZE, shuffle=True,
                                                     num_workers=NUM_WORKERS, pin_memory=True)

    transform_celeba = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    celeba_set = ImageFolder(path_celeba, transform_celeba)
    train_loader_celeba = torch.utils.data.DataLoader(celeba_set, batch_size=BATCH_SIZE, shuffle=True,
                                                      num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader_cifar, train_loader_celeba

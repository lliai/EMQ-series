import os

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def build_imagenet_data(data_path: str = '',
                        input_size: int = 224,
                        batch_size: int = 64,
                        workers: int = 4,
                        proxy: bool = False):
    print('==> Using Pytorch Dataset')

    if proxy:
        input_size = 32
        first_resize = 32
    else:
        first_resize = 256

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    torchvision.set_image_backend('accimage')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(first_resize),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)
    return train_loader, val_loader

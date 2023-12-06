from __future__ import print_function
import os
import socket

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from .samplers import get_proxy_data_log_entropy_histogram, read_entropy_file
"""
mean = {
    'cifar10': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2675, 0.2565, 0.2761),
}
"""


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar10_dataloaders(batch_size=64,
                            num_workers=1,
                            is_instance=False,
                            datafolder=None):
    """
    cifar 100
    """
    if datafolder is None:
        data_folder = get_data_folder()
    else:
        data_folder = '/home/stack/project/EID/data'

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    if is_instance:
        train_set = CIFAR10Instance(
            root=data_folder,
            download=True,
            train=True,
            transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.CIFAR10(
            root=data_folder,
            download=True,
            train=True,
            transform=train_transform)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    test_set = datasets.CIFAR10(
        root=data_folder, download=True, train=False, transform=test_transform)
    test_loader = DataLoader(
        test_set,
        batch_size=int(batch_size / 2),
        shuffle=False,
        num_workers=int(num_workers / 2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


def get_cifar10_split_dataloaders(batch_size=128, num_workers=1):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    test_set = datasets.CIFAR10(
        root=data_folder, download=True, train=False, transform=test_transform)

    train_set = datasets.CIFAR10(
        root=data_folder, download=True, train=True, transform=train_transform)

    # split
    import torch
    from torch.utils.data import random_split
    total_length = len(train_set)
    splited_train_dataset, _ = random_split(
        dataset=train_set,
        lengths=[total_length // 10, total_length * 9 // 10],
        generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(
        splited_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    test_loader = DataLoader(
        test_set,
        batch_size=int(batch_size / 2),
        shuffle=False,
        num_workers=int(num_workers / 2))
    return train_loader, test_loader


class CIFAR10InstanceSample(datasets.CIFAR10):
    """
    CIFAR10Instance+Sample Dataset
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 k=4096,
                 mode='exact',
                 is_sample=True,
                 percent=1.0):
        super().__init__(
            root=root,
            train=train,
            download=download,
            transform=transform,
            target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        if self.train:
            num_samples = len(self.data)
            label = self.targets
        else:
            num_samples = len(self.data)
            label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [
            np.asarray(self.cls_positive[i]) for i in range(num_classes)
        ]
        self.cls_negative = [
            np.asarray(self.cls_negative[i]) for i in range(num_classes)
        ]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [
                np.random.permutation(self.cls_negative[i])[0:n]
                for i in range(num_classes)
            ]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(
                self.cls_negative[target]) else False
            neg_idx = np.random.choice(
                self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_cifar10_dataloaders_sample(batch_size=128,
                                   num_workers=1,
                                   k=4096,
                                   mode='exact',
                                   is_sample=True,
                                   percent=1.0):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    train_set = CIFAR10InstanceSample(
        root=data_folder,
        download=True,
        train=True,
        transform=train_transform,
        k=k,
        mode=mode,
        is_sample=is_sample,
        percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    test_set = datasets.CIFAR10(
        root=data_folder, download=True, train=False, transform=test_transform)
    test_loader = DataLoader(
        test_set,
        batch_size=int(batch_size / 2),
        shuffle=False,
        num_workers=int(num_workers / 2))

    return train_loader, test_loader, n_data


def get_cifar10_dataloaders_entropy(batch_size=128,
                                    train_portion=0.5,
                                    sampling_portion=0.2,
                                    num_workers=1,
                                    k=4096,
                                    mode='exact',
                                    is_sample=True,
                                    percent=1.0,
                                    save='./tmp',
                                    num_class=100):
    """
    cifar 100
    """

    entropy_file = './dataset/entropy_list/cifar10_resnet56_index_entropy_class.txt'
    index, entropy, label = read_entropy_file(entropy_file)

    indices = get_proxy_data_log_entropy_histogram(
        entropy,
        sampling_portion=sampling_portion,
        sampling_type=1,
        dataset='cifar10')

    num_train = num_proxy_data = len(indices)
    split = int(np.floor(train_portion * num_proxy_data))

    num_classes = [0] * num_class

    if not os.path.exists(save):
        # make a template dir to save the entropy file.
        os.makedirs(save)

    with open(os.path.join(save, 'proxy_train_entropy_file.txt'), 'w') as f:
        for idx in indices[:split]:
            f.write('%d %f %d\n' % (idx, entropy[idx], label[idx]))
            num_classes[label[idx]] += 1
    with open(os.path.join(save, 'proxy_val_entropy_file.txt'), 'w') as f:
        for idx in indices[split:num_train]:
            f.write('%d %f %d\n' % (idx, entropy[idx], label[idx]))
            num_classes[label[idx]] += 1

    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    train_set = datasets.CIFAR10(
        root=data_folder, download=True, train=True, transform=train_transform)

    n_data = len(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,  # need to be False when use sampler
        num_workers=num_workers,
        pin_memory=True,
        sampler=SubsetRandomSampler(indices[:split]))

    valid_loader = DataLoader(
        train_set,
        batch_size=int(batch_size / 2),
        shuffle=False,
        num_workers=int(num_workers / 2),
        pin_memory=True,
        sampler=SubsetRandomSampler(indices[split:num_train]))

    return train_loader, valid_loader, n_data

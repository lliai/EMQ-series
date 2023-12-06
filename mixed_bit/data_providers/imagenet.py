import os
import shutil

import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import utils
from data_providers import DataProvider
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.imagenet import ImageFolder


def make_imagenet_subset(path2subset,
                         n_sub_classes,
                         path2imagenet='/userhome/memory_data/imagenet'):
    imagenet_train_folder = os.path.join(path2imagenet, 'train')
    imagenet_val_folder = os.path.join(path2imagenet, 'val')

    subfolders = sorted(
        [f.path for f in os.scandir(imagenet_train_folder) if f.is_dir()])
    # np.random.seed(DataProvider.VALID_SEED)
    np.random.shuffle(subfolders)

    chosen_train_folders = subfolders[:n_sub_classes]
    class_name_list = []
    for train_folder in chosen_train_folders:
        class_name = train_folder.split('/')[-1]
        class_name_list.append(class_name)

    print('=> Start building subset%d' % n_sub_classes)
    for cls_name in class_name_list:
        src_train_folder = os.path.join(imagenet_train_folder, cls_name)
        target_train_folder = os.path.join(path2subset, 'train/%s' % cls_name)
        shutil.copytree(src_train_folder, target_train_folder)
        print('Train: %s -> %s' % (src_train_folder, target_train_folder))

        src_val_folder = os.path.join(imagenet_val_folder, cls_name)
        target_val_folder = os.path.join(path2subset, 'val/%s' % cls_name)
        shutil.copytree(src_val_folder, target_val_folder)
        print('Val: %s -> %s' % (src_val_folder, target_val_folder))
    print('=> Finish building subset%d' % n_sub_classes)


class ImagenetDataProvider(DataProvider):

    def __init__(self,
                 save_path=None,
                 train_batch_size=256,
                 test_batch_size=512,
                 valid_size=None,
                 n_worker=24,
                 manual_seed=12,
                 load_type='dali',
                 local_rank=0,
                 world_size=1,
                 **kwargs):

        self._save_path = save_path
        self.valid = None
        if valid_size is not None:
            pass
        else:
            self.train = utils.get_imagenet_iter(
                data_type='train',
                image_dir=self.train_path,
                batch_size=train_batch_size,
                num_threads=n_worker,
                device_id=local_rank,
                manual_seed=manual_seed,
                num_gpus=torch.cuda.device_count(),
                crop=self.image_size,
                val_size=self.image_size,
                world_size=world_size,
                local_rank=local_rank)
            self.test = utils.get_imagenet_iter(
                data_type='val',
                image_dir=self.valid_path,
                manual_seed=manual_seed,
                batch_size=test_batch_size,
                num_threads=n_worker,
                device_id=local_rank,
                num_gpus=torch.cuda.device_count(),
                crop=self.image_size,
                val_size=256,
                world_size=world_size,
                local_rank=local_rank)
            # data_transform = transforms.Compose([
            #     transforms.Resize(32),
            #     transforms.CenterCrop(32),
            #     transforms.ToTensor(),
            #     transforms.Normalize(
            #         mean=[0.485, 0.456, 0.406],
            #         std=[0.229, 0.224, 0.225])
            # ])
            # self.train = ImageFolder(
            #     root=self.train_path,
            #     transform=data_transform
            # )
            # self.test = ImageFolder(
            #     root=self.valid_path,
            #     transform=data_transform
            # )

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'imagenet'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 1000

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/userhome/data/imagenet'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download ImageNet')

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self._save_path, 'val')

    @property
    def resize_value(self):
        return 256

    @property
    def image_size(self):
        return 224

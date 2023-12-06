import os
import pickle
import sys

import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import (DALIClassificationIterator,
                                        DALIGenericIterator)
from sklearn.utils import shuffle

IMAGENET_IMAGES_NUM_TRAIN = 1281167
IMAGENET_IMAGES_NUM_TEST = 50000
CIFAR_IMAGES_NUM_TRAIN = 50000
CIFAR_IMAGES_NUM_TEST = 10000


class Cutout(object):

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def cutout_func(img, length=16):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    mask[y1:y2, x1:x2] = 0.
    # mask = torch.from_numpy(mask)
    mask = mask.reshape(img.shape)
    img *= mask
    return img


def cutout_batch(img, length=16):
    h, w = img.size(2), img.size(3)
    masks = []
    for i in range(img.size(0)):
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img[0]).unsqueeze(0)
        masks.append(mask)
    masks = torch.cat(masks).cuda()
    img *= masks
    return img


class DALIDataloader(DALIGenericIterator):

    def __init__(self,
                 pipeline,
                 size,
                 batch_size,
                 output_map=['data', 'label'],
                 auto_reset=True,
                 onehot_label=False,
                 dataset='imagenet'):
        self._size_all = size
        self.batch_size = batch_size
        self.onehot_label = onehot_label
        self.output_map = output_map
        if dataset != 'cifar10':
            super().__init__(
                pipelines=pipeline,
                reader_name='Reader',
                fill_last_batch=False,
                output_map=output_map)
        else:
            super().__init__(
                pipelines=pipeline,
                size=size,
                auto_reset=auto_reset,
                output_map=output_map,
                fill_last_batch=True,
                last_batch_padded=False)

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return [batch[0]['data'], batch[0]['label'].squeeze()]
        data = super().__next__()[0]
        if self.onehot_label:
            return [
                data[self.output_map[0]],
                data[self.output_map[1]].squeeze().long()
            ]
        else:
            return [data[self.output_map[0]], data[self.output_map[1]]]

    def __len__(self):
        if self._size_all % self.batch_size == 0:
            return self._size_all // self.batch_size
        else:
            return self._size_all // self.batch_size + 1


class HybridTrainPipe(Pipeline):

    def __init__(self,
                 batch_size,
                 num_threads,
                 device_id,
                 data_dir,
                 crop,
                 manual_seed,
                 dali_cpu=False,
                 local_rank=0,
                 world_size=1):
        super(HybridTrainPipe, self).__init__(
            batch_size, num_threads, device_id, seed=manual_seed)
        self.input = ops.FileReader(
            file_root=data_dir,
            shard_id=local_rank,
            num_shards=world_size,
            random_shuffle=True,
            pad_last_batch=True)
        # let user decide which pipeline works him bets for RN version he runs
        if dali_cpu:
            dali_device = 'cpu'
            self.decode = ops.HostDecoderRandomCrop(
                device=dali_device,
                output_type=types.RGB,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100)
        else:
            dali_device = 'gpu'
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.ImageDecoderRandomCrop(
                device='mixed',
                output_type=types.RGB,
                device_memory_padding=211025920,
                host_memory_padding=140544512,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100)
        self.res = ops.Resize(
            device=dali_device,
            resize_x=crop,
            resize_y=crop,
            interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(
            device='gpu',
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name='Reader')
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):

    def __init__(self,
                 batch_size,
                 num_threads,
                 device_id,
                 data_dir,
                 crop,
                 size,
                 manual_seed,
                 local_rank=0,
                 world_size=1):
        super(HybridValPipe, self).__init__(
            batch_size, num_threads, device_id, seed=manual_seed)
        self.input = ops.FileReader(
            file_root=data_dir,
            shard_id=local_rank,
            num_shards=world_size,
            random_shuffle=True)
        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        self.res = ops.Resize(
            device='gpu',
            resize_shorter=size,
            interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(
            device='gpu',
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name='Reader')
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_imagenet_iter(data_type,
                      image_dir,
                      batch_size,
                      num_threads,
                      device_id,
                      num_gpus,
                      crop,
                      manual_seed,
                      val_size=256,
                      world_size=1,
                      local_rank=0):
    if data_type == 'train':
        pip_train = HybridTrainPipe(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=local_rank,
            manual_seed=manual_seed,
            data_dir=image_dir,
            crop=crop,
            world_size=world_size,
            local_rank=local_rank)
        pip_train.build()
        dali_iter_train = DALIDataloader(
            pipeline=pip_train,
            size=IMAGENET_IMAGES_NUM_TRAIN // world_size,
            batch_size=batch_size,
            onehot_label=True)
        return dali_iter_train
    elif data_type == 'val':
        pip_val = HybridValPipe(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=local_rank,
            manual_seed=manual_seed,
            data_dir=image_dir,
            crop=crop,
            size=val_size,
            world_size=world_size,
            local_rank=local_rank)
        pip_val.build()
        dali_iter_val = DALIDataloader(
            pipeline=pip_val,
            size=IMAGENET_IMAGES_NUM_TEST // world_size,
            batch_size=batch_size,
            onehot_label=True)
        return dali_iter_val


class HybridTrainPipe_CIFAR(Pipeline):

    def __init__(self,
                 batch_size,
                 num_threads,
                 device_id,
                 data_dir,
                 crop,
                 manual_seed,
                 dali_cpu=False,
                 local_rank=0,
                 world_size=1,
                 cutout=0):
        super(HybridTrainPipe_CIFAR, self).__init__(
            batch_size, num_threads, device_id, seed=manual_seed)
        self.iterator = iter(
            CIFAR_INPUT_ITER(batch_size, 'train', root=data_dir))
        dali_device = 'gpu'
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.pad = ops.Paste(device=dali_device, ratio=1.25, fill_value=0)
        self.uniform = ops.Uniform(range=(0., 1.))
        self.crop = ops.Crop(device=dali_device, crop_h=32, crop_w=32)
        self.cmnp = ops.CropMirrorNormalize(
            device='gpu',
            output_layout=types.NCHW,
            mean=[125.31, 122.95, 113.87],
            std=[63.0, 62.09, 66.70])
        self.coin = ops.CoinFlip(probability=0.5)
        self.flip = ops.Flip(device='gpu')

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)

    def define_graph(self):
        rng = self.coin()
        self.jpegs = self.input(name='Reader')
        self.labels = self.input_label()
        output = self.jpegs
        output = self.pad(output.gpu())
        output = self.crop(
            output, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        output = self.flip(output, horizontal=rng)
        output = self.cmnp(output)
        return [output, self.labels]


class HybridValPipe_CIFAR(Pipeline):

    def __init__(self,
                 batch_size,
                 num_threads,
                 device_id,
                 data_dir,
                 crop,
                 size,
                 manual_seed,
                 local_rank=0,
                 world_size=1):
        super(HybridValPipe_CIFAR, self).__init__(
            batch_size, num_threads, device_id, seed=manual_seed)
        self.iterator = iter(
            CIFAR_INPUT_ITER(batch_size, 'val', root=data_dir))
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.pad = ops.Paste(device='gpu', ratio=1., fill_value=0)
        self.uniform = ops.Uniform(range=(0., 1.))
        self.crop = ops.Crop(device='gpu', crop_h=32, crop_w=32)
        self.coin = ops.CoinFlip(probability=0.5)
        self.flip = ops.Flip(device='gpu')
        self.cmnp = ops.CropMirrorNormalize(
            device='gpu',
            output_layout=types.NCHW,
            mean=[125.31, 122.95, 113.87],
            std=[63.0, 62.09, 66.70])

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images)  # can only in HWC order
        self.feed_input(self.labels, labels)

    def define_graph(self):
        self.jpegs = self.input(name='Reader')
        self.labels = self.input_label()
        # rng = self.coin()
        output = self.jpegs
        output = self.pad(output.gpu())
        output = self.cmnp(output.gpu())
        return [output, self.labels]


class CIFAR_INPUT_ITER():
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self,
                 batch_size,
                 data_type='train',
                 root='/userhome/data/cifar10'):
        self.root = root
        self.batch_size = batch_size
        self.train = (data_type == 'train')
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.targets = np.vstack(self.targets)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        np.save('cifar.npy', self.data)
        self.data = np.load('cifar.npy')  # to serialize, increase locality

    def __iter__(self):
        self.i = 0
        self.n = len(self.data)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            if self.train and self.i % self.n == 0:
                self.data, self.targets = shuffle(
                    self.data, self.targets, random_state=0)
            img, label = self.data[self.i], self.targets[self.i]
            batch.append(img)
            labels.append(label)
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    next = __next__


def get_cifar_iter(data_type,
                   image_dir,
                   batch_size,
                   num_threads,
                   manual_seed,
                   local_rank=0,
                   world_size=1,
                   val_size=32,
                   cutout=0):
    if data_type == 'train':
        pip_train = HybridTrainPipe_CIFAR(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=local_rank,
            data_dir=image_dir,
            crop=32,
            world_size=world_size,
            local_rank=local_rank,
            cutout=cutout,
            manual_seed=manual_seed)
        pip_train.build()
        dali_iter_train = DALIDataloader(
            pipeline=pip_train,
            size=CIFAR_IMAGES_NUM_TRAIN // world_size,
            batch_size=batch_size,
            onehot_label=True,
            dataset='cifar10')
        return dali_iter_train

    elif data_type == 'val':
        pip_val = HybridValPipe_CIFAR(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=local_rank,
            data_dir=image_dir,
            crop=32,
            size=val_size,
            world_size=world_size,
            local_rank=local_rank,
            manual_seed=manual_seed)
        pip_val.build()
        dali_iter_val = DALIDataloader(
            pipeline=pip_val,
            size=CIFAR_IMAGES_NUM_TEST // world_size,
            batch_size=batch_size,
            onehot_label=True,
            dataset='cifar10')
        return dali_iter_val

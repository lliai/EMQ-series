import gc
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

import emq.models.hubconf as hubconf
from emq.api import EMQAPI
from emq.dataset.imagenet import build_imagenet_data
from emq.models.mixed.resnet_imagenet import ResNet_ImageNet
from emq.predictor.pruners import predictive
from emq.quant import QuantModel, QuantModule
from emq.structures.tree import TreeStructure
from emq.utils.rank_consistency import spearman
from exps.search.evo_search_emq_bit import execute_zc_workflow
from exps.search.evo_search_emq_zc import build_cnn

emqapi = EMQAPI('./data/PTQ-GT.pkl', verbose=False)

# load qnn original
# cnn = eval('hubconf.resnet18(pretrained=False)')
cnn = ResNet_ImageNet(depth=18)

cnn.eval()

train_loader, test_loader = build_imagenet_data(
    batch_size=8, workers=0, data_path='D:\gitee\imagenet-mini')


def random_bit_cfg(arch='resnet18', quant_type='PTQ'):
    """Generate different bit cfg based on architecture"""
    assert arch in ['resnet18', 'mobilenetv2', 'resnet50']
    if arch == 'resnet18':
        if quant_type == 'PTQ':
            return tuple(random.choice(range(2, 5)) for _ in range(18))
        elif quant_type == 'QAT':
            # QAT
            return tuple(random.choice(range(4, 9)) for _ in range(18))
    elif arch == 'mobilenetv2':
        if quant_type == 'PTQ':
            return tuple(random.choice(range(2, 5)) for _ in range(53))
        elif quant_type == 'QAT':
            # QAT
            return tuple(random.choice(range(4, 9)) for _ in range(53))
    elif arch == 'resnet50':
        if quant_type == 'PTQ':
            return tuple(random.choice(range(2, 5)) for _ in range(53))
        elif quant_type == 'QAT':
            # QAT
            return tuple(random.choice(range(4, 9)) for _ in range(53))


# img, label = next(iter(train_loader))


def execute_zc_workflow(train_loader, nn, bit_cfg, zc_name='qe_score'):
    dataload_info = ['random', 1, 1000]
    zc = predictive.find_measures(
        nn,
        train_loader,
        dataload_info=dataload_info,
        measure_names=[zc_name],
        loss_fn=F.cross_entropy,
        device=torch.device('cuda:0')
        if torch.cuda.is_available() else torch.device('cpu'),
        bit_cfg=bit_cfg)
    return zc


def fitness_zc_score(dataloader, arch, bit_cfg, zc_name, positive=True):
    """Solution is belong to popultion."""
    nn = build_cnn(arch)
    zc = execute_zc_workflow(dataloader, nn, bit_cfg, zc_name)

    if isinstance(zc, Tensor):
        zc = zc.item()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    del nn
    return zc if positive else zc * -1


m = ResNet_ImageNet(depth=18)
# with open('./data/PTQ-GT.pkl', 'r') as f:
with open('./scripts/bench/extracted.emq', 'r') as f:
    lines = f.readlines()
    # record for plot
    x_param = []
    y_acc = []
    z_zc = []

    for line in tqdm(lines):
        # for i in tqdm(range(50)):
        bit_cfg, acc = line.split(':')
        # bit_cfg = emqapi.fix_bit_cfg(i)
        # acc = emqapi.query_by_cfg(bit_cfg)

        bit_cfg = eval(bit_cfg)
        acc = float(acc)
        params, first_last_size = m.cfg2params_perlayer(
            cfg=m.cfg, length=len(bit_cfg), quant_type='PTQ')
        params = [i / (1024 * 1024) for i in params]
        first_last_size = first_last_size / (1024 * 1024)
        model_size = np.sum(
            np.array(bit_cfg) * np.array(params) / 8) + first_last_size

        x_param.append(model_size)
        y_acc.append(acc)
        z_zc.append(
            execute_zc_workflow(
                train_loader, cnn, bit_cfg=bit_cfg, zc_name='nwot'))
        # z_zc.append(struct(img, label, cnn, bit_cfg))

    print('spearman: ', spearman(y_acc, z_zc))

    plt.figure()
    # plt.scatter(x_param, y_acc, c=z_zc, cmap='rainbow')
    # plt.scatter(x_param, z_zc, c=y_acc, cmap='rainbow')
    plt.scatter(y_acc, z_zc, c=x_param, cmap='rainbow')
    plt.colorbar()
    plt.xlabel('ACC')
    plt.ylabel('ZC')
    plt.title('EMQ')
    plt.show()

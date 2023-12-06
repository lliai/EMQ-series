import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import emq.models.hubconf as hubconf
from emq.api import EMQAPI
from emq.dataset.imagenet import build_imagenet_data
from emq.models.mixed.resnet_imagenet import ResNet_ImageNet
from emq.quant import QuantModel, QuantModule
from emq.structures.tree import TreeStructure
from emq.utils.rank_consistency import spearman
from exps.search.evo_search_emq_bit import execute_zc_workflow
from exps.search.evo_search_emq_zc import build_cnn

# load qnn original
# cnn = eval('hubconf.resnet18(pretrained=False)')
cnn = ResNet_ImageNet(depth=18)

# cnn.cuda()
cnn.eval()

# # build quantization parameters
# wq_params = {'n_bits': 8, 'channel_wise': False, 'scale_method': 'mse'}
# aq_params = {
#     'n_bits': 8,
#     'channel_wise': False,
#     'scale_method': 'mse',
#     'leaf_param': False,
# }
# qnn = QuantModel(
#     model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
# # qnn.cuda()
# qnn.eval()
# qnn.set_first_last_layer_to_8bit()
# qnn.set_quant_state(weight_quant=True, act_quant=False)

train_loader, test_loader = build_imagenet_data(
    batch_size=8, workers=0, data_path='E://all_imagenet_data')
# dataiter = iter(train_loader)


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


# official one
OM45 = [4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 3, 3, 3, 3]  # 24.8430
OM50 = [4, 3, 3, 3, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 3, 4, 4, 4]  # 24.8430
OM55 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4]  # 24.9549

# our emq one
EM45 = [2, 4, 4, 3, 3, 3, 3, 4, 4, 4, 3, 3, 2, 3, 2, 3, 4, 4]  # 24.7275
EM55 = [4, 4, 3, 2, 3, 4, 4, 2, 2, 4, 2, 4, 4, 3, 4, 4, 4, 3]  # 24.7529
EM40 = [3, 4, 3, 3, 3, 2, 4, 3, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2]  # 24.58

# ALL_M = [OM45, OM50, OM55, EM45, EM55, EM40]

img, label = next(iter(train_loader))

# for bit_cfg in ALL_M:
#     zc = execute_zc_workflow(img, label, qnn, bit_cfg=bit_cfg)
#     print(zc)

tree_geno = {
    # 'op_geno': [[2, 14], [4, 14], 0], # 81
    'op_geno': [[14, 6], [4, 2], 1],
    'input_geno': ['weight', 'weight'],
}

struct = TreeStructure()
struct.genotype = tree_geno

m = ResNet_ImageNet(depth=18)
# with open('./data/PTQ-GT.pkl', 'r') as f:
with open('./scripts/bench/extracted.pkl', 'r') as f:
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
        # z_zc.append(
        #     execute_zc_workflow(
        #         img,
        #         label,
        #         qnn,
        #         bit_cfg=bit_cfg,
        #         zc_idxs=[9, 1, 19, 17],
        #         zc_names='t2'))
        z_zc.append(struct(img, label, cnn, bit_cfg))

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

# https://github.com/alibaba/lightweight-neural-architecture-search/blob/main/nas/scores/compute_madnas.py

# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import time
from abc import ABCMeta

import numpy as np
# from .bn_score import network_weight_gaussian_init
import torch

from emq.quant.quant_block import QuantBasicBlock, QuantBottleneck, QuantModule
from emq.quant.quant_model import QuantModel
from . import measure


class ComputeMadnasScore(metaclass=ABCMeta):

    def __init__(self, bit_cfg=None):
        self.init_std = 1
        self.init_std_act = 1
        self.batch_size = 8
        self.resolution = 224
        self.in_ch = 3
        self.ratio_coef = [0, 0, 1, 1, 6]  # for resnet18

        self.budget_layers = 17  # for resnet18
        self.align_budget_layers = False
        self.bit_cfg = bit_cfg

    def ratio_score(self, stages_num, block_std_list):
        assert stages_num in [5, 7]
        if stages_num == 5:
            self.ratio_coef = [0, 0, 1, 1, 6]
        elif stages_num == 7:
            self.ratio_coef = [0, 0, 1, 1, 2, 2, 6]
        else:
            raise ValueError(
                'the length of the stage_features_list (%d) must be equal to the length of ratio_coef (%d)'
                % (stages_num, len(self.ratio_coef)))

        nas_score_list = []
        for idx, ratio in enumerate(self.ratio_coef):
            if ratio == 0:
                nas_score_list.append(0.0)
                continue

            # compute std scaling
            nas_score_std = 0.0
            for idx1 in range(self.stage_block_num[idx]):
                nas_score_std += block_std_list[idx1]

            # larger channel and larger resolution, larger performance.
            nas_score_feat = np.log(self.stage_channels[idx])
            nas_score_stage = nas_score_std + nas_score_feat

            nas_score_list.append(nas_score_stage * ratio)

        return nas_score_list

    def assign_bit_cfg(self, model, bit_cfg):
        i = 0
        for name, module in model.named_modules():
            if 'fc' in name or 'model.conv1' in name:
                continue

            if isinstance(module, QuantBasicBlock):
                module.nbitsW = bit_cfg[i:i + 2]
                i += 2
            elif isinstance(module, QuantBottleneck):
                module.nbitsW = bit_cfg[i:i + 3]
                i += 3

    def __call__(self, model, bit_cfg):
        timer_start = time.time()
        self.assign_bit_cfg(model, bit_cfg)
        # [0, 1, 2, 3, 3]    [2, 4, 6, 7, 8]      [4, 8, 12, 14, 16]     [64, 128, 256, 512, 512]
        self.stage_idx, self.stage_block_num, self.stage_layer_num, self.stage_channels = model.get_stage_info(
        )

        block_std_list = model.madnas_forward_pre_GAP()
        nas_score_once = self.ratio_score(len(self.stage_idx), block_std_list)

        timer_end = time.time()
        nas_score_once = np.array(nas_score_once)
        avg_nas_score = np.sum(nas_score_once)

        if self.align_budget_layers:
            nas_score_once = nas_score_once / \
                self.stage_layer_num[-1] * self.budget_layers

        info = {
            'avg_nas_score': avg_nas_score,
            'std_nas_score': avg_nas_score,
            'nas_score_list': nas_score_once,
            'time': timer_end - timer_start
        }

        return info


def convert_qnn(cnn=None):
    wq_params = {'n_bits': 2, 'channel_wise': False, 'scale_method': 'mse'}
    aq_params = {
        'n_bits': 8,
        'channel_wise': False,
        'scale_method': 'mse',
        'leaf_param': False
    }
    qnn = QuantModel(
        model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    if torch.cuda.is_available():
        qnn.cuda()
    qnn.eval()
    qnn.set_first_last_layer_to_8bit()
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    return qnn


@measure('qe_score', bn=True)
def compute_qe_score(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
    bit_cfg=None,
):
    # convert cnn to qnn
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    qnn = convert_qnn(net)

    madnas = ComputeMadnasScore(bit_cfg)
    # net = network_weight_gaussian_init(net)
    info = madnas(qnn, bit_cfg)

    return info['avg_nas_score']

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from emq.models.mobilenetv2 import InvertedResidual
from emq.models.resnet import BasicBlock, Bottleneck
from emq.quant.quant_layer import (QuantModule, StraightThrough,
                                   UniformAffineQuantizer)
from emq.utils.bit_lut import STD_BITS_LUT


class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """

    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self,
                        weight_quant: bool = False,
                        act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantBasicBlock(BaseQuantBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """

    def __init__(self,
                 basic_block: BasicBlock,
                 weight_quant_params: dict = {},
                 act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.basic_block = basic_block
        self.outplanes = basic_block.outplanes
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params,
                                 act_quant_params)
        self.conv1.activation_function = basic_block.relu1
        self.conv2 = QuantModule(
            basic_block.conv2,
            weight_quant_params,
            act_quant_params,
            disable_act_quant=True)

        # modify the activation function to ReLU
        self.activation_function = basic_block.relu2
        self.nbitsA: List = [8, 8]
        self.nbitsW: List = [8, 8]

        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(
                basic_block.downsample[0],
                weight_quant_params,
                act_quant_params,
                disable_act_quant=True)
        # copying all attributes in original block
        self.stride = basic_block.stride

    def get_block_num(self):
        return 1

    def get_num_layers(self):
        return 2

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def get_log_zen_score(self):
        conv1_std = np.log(STD_BITS_LUT[1][self.nbitsA[0]] *
                           STD_BITS_LUT[1][self.nbitsW[0]])
        conv2_std = np.log(STD_BITS_LUT[1][self.nbitsA[1]] *
                           STD_BITS_LUT[1][self.nbitsW[1]])
        return [conv1_std + conv2_std]


class QuantBottleneck(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """

    def __init__(self,
                 bottleneck: Bottleneck,
                 weight_quant_params: dict = {},
                 act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.bottleneck = bottleneck
        self.outplanes = bottleneck.outplanes
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params,
                                 act_quant_params)
        self.conv1.activation_function = bottleneck.relu1
        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params,
                                 act_quant_params)
        self.conv2.activation_function = bottleneck.relu2
        self.conv3 = QuantModule(
            bottleneck.conv3,
            weight_quant_params,
            act_quant_params,
            disable_act_quant=True)
        self.nbitsA: List = [8, 8, 8]
        self.nbitsW: List = [8, 8, 8]

        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu3

        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(
                bottleneck.downsample[0],
                weight_quant_params,
                act_quant_params,
                disable_act_quant=True)
        # copying all attributes in original block
        self.stride = bottleneck.stride

    def get_log_zen_score(self):
        conv1_std = np.log(STD_BITS_LUT[1][self.nbitsA[0]] *
                           STD_BITS_LUT[1][self.nbitsW[0]])
        conv2_std = np.log(STD_BITS_LUT[1][self.nbitsA[1]] *
                           STD_BITS_LUT[1][self.nbitsW[1]])
        conv3_std = np.log(STD_BITS_LUT[1][self.nbitsA[2]] *
                           STD_BITS_LUT[1][self.nbitsW[2]])
        return [conv1_std + conv2_std + conv3_std]

    def get_block_num(self):
        return 1

    def get_num_layers(self):
        return 3

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantInvertedResidual(BaseQuantBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """

    def __init__(self,
                 inv_res: InvertedResidual,
                 weight_quant_params: dict = {},
                 act_quant_params: dict = {}):
        super().__init__(act_quant_params)

        self.use_res_connect = inv_res.use_res_connect
        self.expand_ratio = inv_res.expand_ratio
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params,
                            act_quant_params),
                QuantModule(
                    inv_res.conv[3],
                    weight_quant_params,
                    act_quant_params,
                    disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
        else:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params,
                            act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params,
                            act_quant_params),
                QuantModule(
                    inv_res.conv[6],
                    weight_quant_params,
                    act_quant_params,
                    disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].activation_function = nn.ReLU6()

        self.nbitsA: List = [8, 8, 8]
        self.nbitsW: List = [8, 8, 8]

    def forward(self, x):
        out = x + self.conv(x) if self.use_res_connect else self.conv(x)
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def get_log_zen_score(self):
        conv1_std = np.log(STD_BITS_LUT[1][self.nbitsA[0]] *
                           STD_BITS_LUT[1][self.nbitsW[0]])
        conv2_std = np.log(STD_BITS_LUT[1][self.nbitsA[1]] *
                           STD_BITS_LUT[1][self.nbitsW[1]])
        conv3_std = np.log(STD_BITS_LUT[1][self.nbitsA[2]] *
                           STD_BITS_LUT[1][self.nbitsW[2]])
        return [conv1_std + conv2_std + conv3_std]

    def get_block_num(self):
        return 1

    def get_num_layers(self):
        return 3


specials = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    InvertedResidual: QuantInvertedResidual,
}

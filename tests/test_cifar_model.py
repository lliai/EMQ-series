import unittest
from unittest import TestCase

import torch

import emq.models.joint.spaces.mutator  # noqa: F401
from emq.models.joint.cnnnet import CnnNet
from emq.models.joint.spaces.space_quant_k1dwk1 import SpaceQuantk1dwk1
from exps.cifar.resnet import resnet20

m = resnet20()

print(m)

# calculate the params of `m`
params = sum(p.numel() for p in m.parameters())
print(f'resnet20 - params: {params/1e6}M')

init_bit = 4
bits_list = [init_bit, init_bit, init_bit]
structure_info = [
    {
        'class': 'ConvKXBNRELU',
        'in': 3,
        'out': 16,  # out channel
        's': 2,  # stride
        'k': 3,  # kernel size
        'nbitsA': 8,
        'nbitsW': 8
    },
    {
        'class': 'SuperQuantResK1DWK1',
        'in': 16,
        'out': 64,
        's': 2,
        'k': 3,
        'L': 1,  # number of layers
        'btn': 48,
        'nbitsA': bits_list,
        'nbitsW': bits_list
    },
    {
        'class': 'SuperQuantResK1DWK1',
        'in': 64,
        'out': 128,
        's': 2,
        'k': 3,
        'L': 1,
        'btn': 320,
        'nbitsA': bits_list,
        'nbitsW': bits_list
    },
    {
        'class': 'SuperQuantResK1DWK1',
        'in': 128,
        'out': 256,
        's': 1,
        'k': 3,
        'L': 1,
        'btn': 512,
        'nbitsA': bits_list,
        'nbitsW': bits_list
    },
    {
        'class': 'SuperQuantResK1DWK1',
        'in': 256,
        'out': 256,
        's': 1,
        'k': 3,
        'L': 1,
        'btn': 512,
        'nbitsA': bits_list,
        'nbitsW': bits_list
    },
    {
        'class': 'ConvKXBNRELU',
        'in': 256,
        'out': 10,
        's': 1,
        'k': 1,
        'nbitsA': init_bit,
        'nbitsW': init_bit
    },
]

net = CnnNet(
    structure_info=structure_info, out_indices=(1, 2, 3), num_classes=10)
print(net)
params = sum(p.numel() for p in net.parameters())
# param = net.get_params_for_trt(32)
print(f'quant param: {params/1e6}M')

input = torch.randn(1, 3, 32, 32)
output, inner_layer_feature = net.forward_inner_layer_features(input)
print(output.shape)

import unittest
from unittest import TestCase

import torch

import emq.models.joint.spaces.mutator  # noqa: F401
from emq.models.joint.cnnnet import CnnNet
from emq.models.joint.spaces.space_quant_k1dwk1 import SpaceQuantk1dwk1


class TestTinyNAS(TestCase):

    @unittest.skip('skip')
    def test_model_build(self):
        init_bit = 4
        bits_list = [init_bit, init_bit, init_bit]
        structure_info = [
            {
                'class': 'ConvKXBNRELU',
                'in': 3,
                'out': 16,
                's': 2,
                'k': 3,
                'nbitsA': 8,
                'nbitsW': 8
            },
            {
                'class': 'SuperQuantResK1DWK1',
                'in': 16,
                'out': 24,
                's': 2,
                'k': 3,
                'L': 1,
                'btn': 48,
                'nbitsA': bits_list,
                'nbitsW': bits_list
            },
            {
                'class': 'SuperQuantResK1DWK1',
                'in': 24,
                'out': 48,
                's': 2,
                'k': 3,
                'L': 1,
                'btn': 96,
                'nbitsA': bits_list,
                'nbitsW': bits_list
            },
            {
                'class': 'SuperQuantResK1DWK1',
                'in': 48,
                'out': 64,
                's': 2,
                'k': 3,
                'L': 1,
                'btn': 128,
                'nbitsA': bits_list,
                'nbitsW': bits_list
            },
            {
                'class': 'SuperQuantResK1DWK1',
                'in': 64,
                'out': 96,
                's': 1,
                'k': 3,
                'L': 1,
                'btn': 192,
                'nbitsA': bits_list,
                'nbitsW': bits_list
            },
            {
                'class': 'SuperQuantResK1DWK1',
                'in': 96,
                'out': 192,
                's': 2,
                'k': 3,
                'L': 1,
                'btn': 384,
                'nbitsA': bits_list,
                'nbitsW': bits_list
            },
            {
                'class': 'ConvKXBNRELU',
                'in': 192,
                'out': 1280,
                's': 1,
                'k': 1,
                'nbitsA': init_bit,
                'nbitsW': init_bit
            },
        ]

        net = CnnNet(structure_info=structure_info, )

        input = torch.randn(3, 3, 224, 224)

        output = net(input)

        print(len(output))

        for i in output:
            print(i.shape)

    @unittest.skip('skip')
    def test_space(self):
        init_bit = 4
        bits_list = [init_bit, init_bit, init_bit]
        # used for mutation
        structure_info = [
            {
                'class': 'ConvKXBNRELU',
                'in': 3,
                'out': 16,
                's': 2,
                'k': 3,
                'nbitsA': 8,
                'nbitsW': 8
            },
            {
                'class': 'SuperQuantResK1DWK1',
                'in': 16,
                'out': 24,
                's': 2,
                'k': 3,
                'L': 1,
                'btn': 48,
                'nbitsA': bits_list,
                'nbitsW': bits_list
            },
            {
                'class': 'SuperQuantResK1DWK1',
                'in': 24,
                'out': 48,
                's': 2,
                'k': 3,
                'L': 1,
                'btn': 96,
                'nbitsA': bits_list,
                'nbitsW': bits_list
            },
            {
                'class': 'SuperQuantResK1DWK1',
                'in': 48,
                'out': 64,
                's': 2,
                'k': 3,
                'L': 1,
                'btn': 128,
                'nbitsA': bits_list,
                'nbitsW': bits_list
            },
            {
                'class': 'SuperQuantResK1DWK1',
                'in': 64,
                'out': 96,
                's': 1,
                'k': 3,
                'L': 1,
                'btn': 192,
                'nbitsA': bits_list,
                'nbitsW': bits_list
            },
            {
                'class': 'SuperQuantResK1DWK1',
                'in': 96,
                'out': 192,
                's': 2,
                'k': 3,
                'L': 1,
                'btn': 384,
                'nbitsA': bits_list,
                'nbitsW': bits_list
            },
            {
                'class': 'ConvKXBNRELU',
                'in': 192,
                'out': 1280,
                's': 1,
                'k': 1,
                'nbitsA': init_bit,
                'nbitsW': init_bit
            },
        ]
        space = SpaceQuantk1dwk1(image_size=224)

        print(structure_info)
        out = space.mutate(structure_info)
        print(out)

        print(f'equal: {out == structure_info}')

    def test_model_size(self):

        struct_info = [{
            'class': 'ConvKXBNRELU',
            'in': 3,
            'k': 3,
            'nbitsA': 8,
            'nbitsW': 8,
            'out': 8,
            's': 2
        }, {
            'L': 3,
            'btn': 64,
            'class': 'SuperResK1DWK1',
            'in': 8,
            'inner_class': 'ResK1DWK1',
            'k': 5,
            'nbitsA': [5, 5, 5, 5, 5, 5, 5, 5, 5],
            'nbitsW': [5, 5, 5, 5, 5, 5, 4, 4, 4],
            'out': 32,
            's': 2
        }, {
            'L': 3,
            'btn': 128,
            'class': 'SuperResK1DWK1',
            'in': 32,
            'inner_class': 'ResK1DWK1',
            'k': 5,
            'nbitsA': [5, 5, 5, 5, 5, 5, 5, 5, 5],
            'nbitsW': [5, 5, 5, 5, 5, 5, 4, 4, 4],
            'out': 64,
            's': 2
        }, {
            'L': 3,
            'btn': 224,
            'class': 'SuperResK1DWK1',
            'in': 64,
            'inner_class': 'ResK1DWK1',
            'k': 5,
            'nbitsA': [5, 5, 5, 5, 5, 5, 5, 5, 5],
            'nbitsW': [5, 5, 5, 5, 5, 5, 5, 5, 5],
            'out': 64,
            's': 2
        }, {
            'L': 3,
            'btn': 336,
            'class': 'SuperResK1DWK1',
            'in': 64,
            'inner_class': 'ResK1DWK1',
            'k': 5,
            'nbitsA': [5, 5, 5, 5, 5, 5, 5, 5, 5],
            'nbitsW': [5, 5, 5, 6, 6, 6, 5, 5, 5],
            'out': 96,
            's': 1
        }, {
            'L': 3,
            'btn': 672,
            'class': 'SuperResK1DWK1',
            'in': 96,
            'inner_class': 'ResK1DWK1',
            'k': 5,
            'nbitsA': [5, 5, 5, 5, 5, 5, 5, 5, 5],
            'nbitsW': [6, 6, 6, 4, 4, 4, 5, 5, 5],
            'out': 192,
            's': 2
        }, {
            'class': 'ConvKXBNRELU',
            'in': 192,
            'k': 1,
            'nbitsA': 5,
            'nbitsW': 5,
            'out': 1280,
            's': 1
        }]

        net = CnnNet(structure_info=struct_info)

        param = sum(p.numel() for p in net.parameters())

        print(f'quant param1: {param/1e6}M')

        param = net.get_model_size()

        print(f'quant param2: {param/1e6}M')

        flops = net.get_flops(224)

        print(f'quant flops: {flops/1e6}M')


if __name__ == '__main__':
    unittest.main()

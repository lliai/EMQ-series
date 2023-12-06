from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo  # noqa: F403,F401
from models.base_models import *  # noqa: F403,F401


def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes_1,
                 planes_2=0,
                 stride=1,
                 downsample=None,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        conv1 = conv3x3(inplanes, planes_1, stride)
        bn1 = norm_layer(planes_1)
        relu = nn.ReLU(inplace=True)
        if planes_2 == 0:
            conv2 = conv3x3(planes_1, inplanes)
            bn2 = norm_layer(inplanes)
        else:
            conv2 = conv3x3(planes_1, planes_2)
            bn2 = norm_layer(planes_2)
        self.relu = relu
        self.conv1 = nn.Sequential(
            OrderedDict([('conv', conv1), ('bn', bn1), ('relu', relu)]))
        self.conv2 = nn.Sequential(OrderedDict([('conv', conv2), ('bn', bn2)]))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    def __init__(self,
                 inplanes,
                 planes_1,
                 planes_2,
                 planes_3=0,
                 stride=1,
                 downsample=None,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        conv1 = conv1x1(inplanes, planes_1)
        bn1 = norm_layer(planes_1)
        conv2 = conv3x3(planes_1, planes_2, stride)
        bn2 = norm_layer(planes_2)
        if planes_3 == 0:
            conv3 = conv1x1(planes_2, inplanes)
            bn3 = norm_layer(inplanes)
        else:
            conv3 = conv1x1(planes_2, planes_3)
            bn3 = norm_layer(planes_3)
        relu = nn.ReLU(inplace=True)
        self.relu = relu
        self.conv1 = nn.Sequential(
            OrderedDict([('conv', conv1), ('bn', bn1), ('relu', relu)]))
        self.conv2 = nn.Sequential(
            OrderedDict([('conv', conv2), ('bn', bn2), ('relu', relu)]))
        self.conv3 = nn.Sequential(OrderedDict([('conv', conv3), ('bn', bn3)]))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet_ImageNet(MyNetwork):

    def __init__(self, cfg=None, depth=18, block=BasicBlock, num_classes=1000):
        super(ResNet_ImageNet, self).__init__()
        self.cfgs_base = {
            18: [64, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512],
            34: [
                64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256,
                256, 256, 256, 256, 512, 512, 512, 512
            ],
            50: [
                64, 64, 64, 256, 64, 64, 64, 64, 128, 128, 512, 128, 128, 128,
                128, 128, 128, 256, 256, 1024, 256, 256, 256, 256, 256, 256,
                256, 256, 256, 256, 512, 512, 2048, 512, 512, 512, 512
            ]
        }
        if depth == 18:
            block = BasicBlock
            blocks = [2, 2, 2, 2]
            _cfg = self.cfgs_base[18]
        elif depth == 34:
            block = BasicBlock
            blocks = [3, 4, 6, 3]
            _cfg = self.cfgs_base[34]
        elif depth == 50:
            block = Bottleneck
            blocks = [3, 4, 6, 3]
            _cfg = self.cfgs_base[50]
        if cfg is None:
            cfg = _cfg
        norm_layer = nn.BatchNorm2d
        self.block = block
        self.num_classes = num_classes
        self._norm_layer = norm_layer
        self.depth = depth
        self.cfg = cfg
        self.inplanes = cfg[0]
        self.blocks = blocks
        self.conv1 = nn.Sequential(
            OrderedDict([('conv',
                          nn.Conv2d(
                              3,
                              self.inplanes,
                              kernel_size=7,
                              stride=2,
                              padding=3,
                              bias=False)), ('bn', norm_layer(self.inplanes)),
                         ('relu', nn.ReLU(inplace=True))]))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if depth != 50:
            self.layer1 = self._make_layer(block, cfg[1:blocks[0] + 2],
                                           blocks[0])
            self.layer2 = self._make_layer(
                block,
                cfg[blocks[0] + 2:blocks[0] + 2 + blocks[1] + 1],
                blocks[1],
                stride=2,
            )
            self.layer3 = self._make_layer(
                block,
                cfg[blocks[0] + blocks[1] + 3:blocks[0] + blocks[1] +
                    blocks[2] + 4],
                blocks[2],
                stride=2,
            )
            self.layer4 = self._make_layer(
                block,
                cfg[blocks[0] + blocks[1] + blocks[2] + 4:],
                blocks[3],
                stride=2,
            )
            self.fc = nn.Linear(cfg[blocks[0] + blocks[1] + blocks[2] + 5],
                                num_classes)
        else:
            self.layer1 = self._make_layer(block, cfg[1:2 * blocks[0] + 2],
                                           blocks[0])
            self.layer2 = self._make_layer(
                block,
                cfg[2 * blocks[0] + 2:2 * blocks[0] + 2 + 2 * blocks[1] + 1],
                blocks[1],
                stride=2,
            )
            self.layer3 = self._make_layer(
                block,
                cfg[2 * blocks[0] + 2 * blocks[1] + 3:2 * blocks[0] +
                    2 * blocks[1] + 2 * blocks[2] + 4],
                blocks[2],
                stride=2,
            )
            self.layer4 = self._make_layer(
                block,
                cfg[2 * blocks[0] + 2 * blocks[1] + 2 * blocks[2] + 4:],
                blocks[3],
                stride=2,
            )
            self.fc = nn.Linear(
                cfg[2 * blocks[0] + 2 * blocks[1] + 2 * blocks[2] + 6],
                num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if self.depth == 50:
            first_planes = planes[0:3]
            # downsample at each 1'st layer, for pruning
            downsample = nn.Sequential(
                OrderedDict([('conv',
                              conv1x1(self.inplanes, first_planes[-1],
                                      stride)),
                             ('bn', norm_layer(first_planes[-1]))]))
            layers = []
            layers.append(
                block(self.inplanes, first_planes[0], first_planes[1],
                      first_planes[2], stride, downsample, norm_layer))
            self.inplanes = first_planes[-1]
            later_planes = planes[3:3 + 2 * (blocks - 1)]
            for i in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        later_planes[2 * (i - 1)],
                        later_planes[2 * (i - 1) + 1],
                        norm_layer=norm_layer))
            return nn.Sequential(*layers)
        else:
            first_planes = planes[0:2]
            # downsample at each 1'st layer, for pruning
            downsample = nn.Sequential(
                OrderedDict([('conv',
                              conv1x1(self.inplanes, first_planes[-1],
                                      stride)),
                             ('bn', norm_layer(first_planes[-1]))]))
            layers = []
            layers.append(
                block(self.inplanes, first_planes[0], first_planes[1], stride,
                      downsample, norm_layer))
            self.inplanes = first_planes[-1]
            later_planes = planes[2:2 + blocks - 1]
            for i in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        later_planes[i - 1],
                        norm_layer=norm_layer))
            return nn.Sequential(*layers)

    def cfg2params_perlayer(self, cfg, length, quant_type):
        blocks = self.blocks
        params = [0. for j in range(length)]
        count = 0
        first_last_size = 0.
        if quant_type == 'PTQ':
            params[count] = (3 * 7 * 7 * cfg[0] + 2 * cfg[0])
            count += 1
        inplanes = cfg[0]
        if self.depth != 50:
            sub_cfgs = [
                cfg[1:blocks[0] + 2],
                cfg[blocks[0] + 2:blocks[0] + 2 + blocks[1] + 1],
                cfg[blocks[0] + blocks[1] + 3:blocks[0] + blocks[1] +
                    blocks[2] + 4], cfg[blocks[0] + blocks[1] + blocks[2] + 4:]
            ]
        else:
            sub_cfgs = [
                cfg[1:2 * blocks[0] + 2],
                cfg[2 * blocks[0] + 2:2 * blocks[0] + 2 + 2 * blocks[1] + 1],
                cfg[2 * blocks[0] + 2 * blocks[1] + 3:2 * blocks[0] +
                    2 * blocks[1] + 2 * blocks[2] + 4],
                cfg[2 * blocks[0] + 2 * blocks[1] + 2 * blocks[2] + 4:]
            ]
        for i in range(4):
            planes = sub_cfgs[i]
            if self.depth != 50:
                first_planes = planes[0:2]
                later_planes = planes[2:2 + blocks[i] - 1]
            else:
                first_planes = planes[0:3]
                later_planes = planes[3:3 + 2 * (blocks[i] - 1)]
            if i != 0 and self.block == BasicBlock:
                params[count + 1] += (inplanes * 1 * 1 * first_planes[-1] +
                                      2 * first_planes[-1])  # downsample layer
            if self.block == Bottleneck:
                params[count + 2] += (inplanes * 1 * 1 * first_planes[-1] +
                                      2 * first_planes[-1])  # downsample layer
            if self.depth != 50:
                params[count] += (
                    inplanes * 3 * 3 * first_planes[0] + 2 * first_planes[0])
                params[count + 1] += (
                    first_planes[0] * 3 * 3 * first_planes[1] +
                    2 * first_planes[1])
                count += 2
            else:
                params[count] += (
                    inplanes * 1 * 1 * first_planes[0] + 2 * first_planes[0])
                params[count + 1] += (
                    first_planes[0] * 3 * 3 * first_planes[1] +
                    2 * first_planes[1])
                params[count + 2] += (
                    first_planes[1] * 1 * 1 * first_planes[2] +
                    2 * first_planes[2])
                count += 3
            for j in range(1, self.blocks[i]):
                inplanes = first_planes[-1]
                if self.depth != 50:
                    params[count] += (
                        inplanes * 3 * 3 * later_planes[j - 1] +
                        2 * later_planes[j - 1])
                    params[count + 1] += (
                        later_planes[j - 1] * 3 * 3 * inplanes + 2 * inplanes)
                    count += 2
                else:
                    params[count] += (
                        inplanes * 1 * 1 * later_planes[2 * (j - 1)] +
                        2 * later_planes[2 * (j - 1)])
                    params[count + 1] += (
                        later_planes[2 * (j - 1)] * 3 * 3 *
                        later_planes[2 * (j - 1) + 1] +
                        2 * later_planes[2 * (j - 1) + 1])
                    params[count + 2] += (
                        later_planes[2 * (j - 1) + 1] * 1 * 1 * inplanes +
                        2 * inplanes)
                    count += 3
        if quant_type == 'PTQ':
            if self.depth == 50:
                params[count] += (
                    cfg[2 * blocks[0] + 2 * blocks[1] + 2 * blocks[2] + 6] +
                    1) * self.num_classes
            else:
                params[count] += (cfg[blocks[0] + blocks[1] + blocks[2] + 5] +
                                  1) * self.num_classes

        if quant_type == 'QAT':
            first_last_size += (3 * 7 * 7 * cfg[0] + 2 * cfg[0])
            if self.depth == 50:
                first_last_size += (
                    cfg[2 * blocks[0] + 2 * blocks[1] + 2 * blocks[2] + 6] +
                    1) * self.num_classes
            else:
                first_last_size += (cfg[blocks[0] + blocks[1] + blocks[2] + 5]
                                    + 1) * self.num_classes

        return params, first_last_size

    def cfg2flops_layerwise(
            self, cfg, length,
            quant_type):  # to simplify, only count convolution flops
        assert quant_type in ['PTQ', 'QAT']
        assert cfg is not None

        blocks = self.blocks
        flops = [0 for j in range(length)]
        count = 0
        size = 224
        first_last_flops = 0.
        size /= 2  # first conv layer s=2
        if quant_type == 'PTQ':
            flops[count] += (3 * 7 * 7 * cfg[0] * size * size +
                             5 * cfg[0] * size * size
                             )  # first layer, conv+bn+relu
            count += 1
        if quant_type == 'QAT':
            first_last_flops += (3 * 7 * 7 * cfg[0] * size * size +
                                 5 * cfg[0] * size * size
                                 )  # first layer, conv+bn+relu
        inplanes = cfg[0]
        size /= 2  # pooling s=2
        # flops += (3 * 3 * cfg[0] * size * size) # maxpooling
        if self.depth != 50:
            sub_cfgs = [
                cfg[1:blocks[0] + 2],
                cfg[blocks[0] + 2:blocks[0] + 2 + blocks[1] + 1],
                cfg[blocks[0] + blocks[1] + 3:blocks[0] + blocks[1] +
                    blocks[2] + 4], cfg[blocks[0] + blocks[1] + blocks[2] + 4:]
            ]
        else:
            sub_cfgs = [
                cfg[1:2 * blocks[0] + 2],
                cfg[2 * blocks[0] + 2:2 * blocks[0] + 2 + 2 * blocks[1] + 1],
                cfg[2 * blocks[0] + 2 * blocks[1] + 3:2 * blocks[0] +
                    2 * blocks[1] + 2 * blocks[2] + 4],
                cfg[2 * blocks[0] + 2 * blocks[1] + 2 * blocks[2] + 4:]
            ]
        for i in range(4):  # each layer
            planes = sub_cfgs[i]
            if self.depth != 50:
                first_planes = planes[0:2]
                later_planes = planes[2:2 + blocks[i] - 1]
            else:
                first_planes = planes[0:3]
                later_planes = planes[3:3 + 2 * (blocks[i] - 1)]
            if i in [1, 2, 3]:
                size /= 2
            if self.block == BasicBlock and i != 0:
                flops[count + 1] += (
                    inplanes * 1 * 1 * first_planes[-1] +
                    5 * first_planes[-1]) * size * size  # downsample layer
            elif self.block == Bottleneck:
                flops[count + 2] += (
                    inplanes * 1 * 1 * first_planes[-1] +
                    5 * first_planes[-1]) * size * size  # downsample layer
            if self.depth != 50:
                flops[count] += (inplanes * 3 * 3 * first_planes[0] +
                                 5 * first_planes[0]) * size * size
                flops[count +
                      1] += (first_planes[0] * 3 * 3 * first_planes[1] +
                             5 * first_planes[1]) * size * size
                count += 2
            else:
                size *= 2
                flops[count] += (inplanes * 1 * 1 * first_planes[0] +
                                 5 * first_planes[0]) * size * size
                size /= 2
                flops[count +
                      1] += (first_planes[0] * 3 * 3 * first_planes[1] +
                             5 * first_planes[1]) * size * size
                flops[count +
                      2] += (first_planes[1] * 1 * 1 * first_planes[2] +
                             5 * first_planes[2]) * size * size
                count += 3
            for j in range(1, self.blocks[i]):
                inplanes = first_planes[-1]
                if self.depth != 50:
                    flops[count] += (inplanes * 3 * 3 * later_planes[j - 1] +
                                     5 * later_planes[j - 1]) * size * size
                    flops[count +
                          1] += (later_planes[j - 1] * 3 * 3 * inplanes +
                                 5 * inplanes) * size * size
                    count += 2
                else:
                    flops[count] += (inplanes * 1 * 1 * later_planes[2 *
                                                                     (j - 1)] +
                                     5 * later_planes[2 *
                                                      (j - 1)]) * size * size
                    flops[count +
                          1] += (later_planes[2 * (j - 1)] * 3 * 3 *
                                 later_planes[2 * (j - 1) + 1] +
                                 5 * later_planes[2 *
                                                  (j - 1) + 1]) * size * size
                    flops[count +
                          2] += (later_planes[2 *
                                              (j - 1) + 1] * 1 * 1 * inplanes +
                                 5 * inplanes) * size * size
                    count += 3
        if quant_type == 'PTQ':
            flops[count] += (2 * cfg[-1] + 1) * self.num_classes
        if quant_type == 'QAT':
            first_last_flops += (2 * cfg[-1] + 1) * self.num_classes
        return flops, first_last_flops

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def feature_extract(self, x, quant_type):  # layer-wise
        tensor = []

        # layer-wise-PTQ
        if quant_type == 'PTQ':
            x = self.conv1(x)
            tensor.append(x)
            x = self.maxpool(x)

            for _layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in _layer:
                    if type(block) == BasicBlock:
                        tensor.append(block.conv1(x))
                        x = block(x)
                        tensor.append(x)
                    else:
                        tensor.append(block.conv1(x))
                        tensor.append(block.conv2(block.conv1(x)))
                        x = block(x)
                        tensor.append(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            tensor.append(x)

        # layer-wise
        if quant_type == 'QAT':
            x = self.conv1(x)
            x = self.maxpool(x)

            for _layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in _layer:
                    if type(block) == BasicBlock:
                        tensor.append(block.conv1(x))
                        x = block(x)
                        tensor.append(x)
                    else:
                        tensor.append(block.conv1(x))
                        tensor.append(block.conv2(block.conv1(x)))
                        x = block(x)
                        tensor.append(x)

        return tensor

    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'depth': self.depth,
            'cfg': self.cfg,
            'cfg_base': self.cfgs_base[self.depth],
            'dataset': 'ImageNet',
        }

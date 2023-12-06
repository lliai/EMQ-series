# !/usr/bin/env python

import math

import torch.nn as nn


def conv_bn(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, 3, stride, 1, bias=False),
        nn.BatchNorm2d(outplanes), nn.ReLU6(inplace=True))


def conv_1x1_bn(inplanes, outplanes):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
        nn.BatchNorm2d(outplanes), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inplanes, outplanes, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inplanes * expand_ratio)
        self.use_res_connect = self.stride == 1 and inplanes == outplanes
        self.expand_ratio = expand_ratio
        self.outplanes = outplanes
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, outplanes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outplanes),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inplanes, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, outplanes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outplanes),
            )

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

    def get_block_num(self):
        return 1

    def get_num_layers():
        return 3


class MobileNetV2(nn.Module):

    def __init__(self,
                 n_class=1000,
                 input_size=224,
                 width_mult=1.,
                 dropout=0.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(
                            input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(
                        block(
                            input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # self.features.append(nn.AvgPool2d(input_size // 32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_stage_info(self):
        stage_idx = []
        stage_channels = []
        stage_block_num = []
        stage_layer_num = []

        block_num = 0

        for i, (t, c, n, s) in enumerate(self.interverted_residual_setting):
            stage_idx.append(i)
            block_num += n
            stage_block_num.append(block_num)
            stage_channels.append(c)

        stage_layer_num = [i * 3 for i in stage_block_num]
        return stage_idx, stage_block_num, stage_layer_num, stage_channels

    def madnas_forward_pre_GAP(self):
        # avoid circle import
        from emq.quant.quant_block import QuantInvertedResidual
        block_std_list = []
        for invertR in self.features:
            if isinstance(invertR, (InvertedResidual, QuantInvertedResidual)):
                output_std_list_plain = invertR.get_log_zen_score()
                block_std_list += output_std_list_plain
        return block_std_list


def mobilenetv2(**kwargs):
    """
    Constructs a MobileNetV2 model.
    Length of bit_cfg = 53
    """
    return MobileNetV2(**kwargs)

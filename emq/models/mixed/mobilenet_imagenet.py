from collections import OrderedDict

import torch.nn as nn

from .base_models import MyNetwork


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        OrderedDict([('conv', nn.Conv2d(inp, oup, 3, stride, 1, bias=False)),
                     ('bn', nn.BatchNorm2d(oup)),
                     ('relu', nn.ReLU6(inplace=True))]))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        OrderedDict([('conv', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
                     ('bn', nn.BatchNorm2d(oup)),
                     ('relu', nn.ReLU6(inplace=True))]))


def conv_dw(inp, oup, stride):
    conv1 = nn.Sequential(
        OrderedDict([
            ('conv', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp,
                               bias=False)), ('bn', nn.BatchNorm2d(inp)),
            ('relu', nn.ReLU(inplace=True))
        ]))
    conv2 = nn.Sequential(
        OrderedDict([('conv', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
                     ('bn', nn.BatchNorm2d(oup)),
                     ('relu', nn.ReLU(inplace=True))]))
    return nn.Sequential(conv1, conv2)


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            dw = nn.Sequential(
                OrderedDict([('conv',
                              nn.Conv2d(
                                  hidden_dim,
                                  hidden_dim,
                                  3,
                                  stride,
                                  1,
                                  groups=hidden_dim,
                                  bias=False)),
                             ('bn', nn.BatchNorm2d(hidden_dim)),
                             ('relu', nn.ReLU6(inplace=True))]))
            pw = nn.Sequential(
                OrderedDict([('conv',
                              nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                             ('bn', nn.BatchNorm2d(oup))]))
            self.conv = nn.Sequential(dw, pw)
        else:
            pw = nn.Sequential(
                OrderedDict([('conv',
                              nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)),
                             ('bn', nn.BatchNorm2d(hidden_dim)),
                             ('relu', nn.ReLU6(inplace=True))]))
            dw = nn.Sequential(
                OrderedDict([('conv',
                              nn.Conv2d(
                                  hidden_dim,
                                  hidden_dim,
                                  3,
                                  stride,
                                  1,
                                  groups=hidden_dim,
                                  bias=False)),
                             ('bn', nn.BatchNorm2d(hidden_dim)),
                             ('relu', nn.ReLU6(inplace=True))]))
            pwl = nn.Sequential(
                OrderedDict([('conv',
                              nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                             ('bn', nn.BatchNorm2d(oup))]))
            self.conv = nn.Sequential(pw, dw, pwl)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(MyNetwork):

    def __init__(self, cfg=None, num_classes=1000, dropout=0.2):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        if cfg is None:
            cfg = [32, 16, 24, 32, 64, 96, 160, 320]
        input_channel = cfg[0]
        interverted_residual_setting = [
            # t, c, n, s
            [1, cfg[1], 1, 1],
            [6, cfg[2], 2, 2],
            [6, cfg[3], 3, 2],
            [6, cfg[4], 4, 2],
            [6, cfg[5], 3, 1],
            [6, cfg[6], 3, 2],
            [6, cfg[7], 1, 1],
        ]

        # building first layer
        # input_channel = int(input_channel * width_mult)
        self.cfg = cfg
        self.cfgs_base = [32, 16, 24, 32, 64, 96, 160, 320]
        self.dropout = dropout
        self.last_channel = 1280
        self.num_classes = num_classes
        self.features = [conv_bn(3, input_channel, 2)]
        self.interverted_residual_setting = interverted_residual_setting
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = c
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
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(self.last_channel, num_classes), )

    def cfg2params_perlayer(self, cfg, length, quant_type='PTQ'):
        params = [0. for j in range(length)]
        first_last_size = 0.
        count = 0
        params[count] += (3 * 3 * 3 * cfg[0] + 2 * cfg[0])  # first layer
        input_channel = cfg[0]
        count += 1
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = c
            for i in range(n):
                hidden_dim = round(input_channel * t)
                if i == 0:
                    # self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    if t == 1:
                        params[count] += (3 * 3 * hidden_dim + 2 * hidden_dim)
                        params[count +
                               1] += (1 * 1 * hidden_dim * output_channel +
                                      2 * output_channel)
                        count += 2
                    else:
                        params[count] += (1 * 1 * input_channel * hidden_dim +
                                          2 * hidden_dim)
                        params[count + 1] += (3 * 3 * hidden_dim +
                                              2 * hidden_dim)
                        params[count +
                               2] += (1 * 1 * hidden_dim * output_channel +
                                      2 * output_channel)
                        count += 3
                else:
                    # self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    if t == 1:
                        params[count] += (3 * 3 * hidden_dim + 2 * hidden_dim)
                        params[count +
                               1] += (1 * 1 * hidden_dim * output_channel +
                                      2 * output_channel)
                        count += 2
                    else:
                        params[count] += (1 * 1 * input_channel * hidden_dim +
                                          2 * hidden_dim)
                        params[count + 1] += (3 * 3 * hidden_dim +
                                              2 * hidden_dim)
                        params[count +
                               2] += (1 * 1 * hidden_dim * output_channel +
                                      2 * output_channel)
                        count += 3
                input_channel = output_channel
        params[count] += (1 * 1 * input_channel * self.last_channel +
                          2 * self.last_channel)  # final 1x1 conv
        count += 1
        params[count] += (
            (self.last_channel + 1) * self.num_classes)  # fc layer
        return params, first_last_size

    def cfg2flops_layerwise(
            self,
            cfg,
            length,
            quant_type='PTQ'):  # to simplify, only count convolution flops
        interverted_residual_setting = [
            # t, c, n, s
            [1, cfg[1], 1, 1],
            [6, cfg[2], 2, 2],
            [6, cfg[3], 3, 2],
            [6, cfg[4], 4, 2],
            [6, cfg[5], 3, 1],
            [6, cfg[6], 3, 2],
            [6, cfg[7], 1, 1],
        ]
        size = 224
        flops = [0 for j in range(length)]
        count = 0
        first_last_flops = 0.
        size = size // 2
        flops[count] += (3 * 3 * 3 * cfg[0] +
                         0 * cfg[0]) * size * size  # first layer
        count += 1
        input_channel = cfg[0]
        for t, c, n, s in interverted_residual_setting:
            output_channel = c
            for i in range(n):
                hidden_dim = round(input_channel * t)
                if i == 0:
                    if s == 2:
                        size = size // 2
                    # self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    if t == 1:
                        flops[count] += (3 * 3 * hidden_dim +
                                         0 * hidden_dim) * size * size
                        flops[count +
                              1] += (1 * 1 * hidden_dim * output_channel +
                                     0 * output_channel) * size * size
                        count += 2
                    else:
                        size = size * s
                        flops[count] += (1 * 1 * input_channel * hidden_dim +
                                         0 * hidden_dim) * size * size
                        size = size // s
                        flops[count + 1] += (3 * 3 * hidden_dim +
                                             0 * hidden_dim) * size * size
                        flops[count +
                              2] += (1 * 1 * hidden_dim * output_channel +
                                     0 * output_channel) * size * size
                        count += 3
                else:
                    # self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    if t == 1:
                        flops[count] += (3 * 3 * hidden_dim +
                                         0 * hidden_dim) * size * size
                        flops[count +
                              1] += (1 * 1 * hidden_dim * output_channel +
                                     0 * output_channel) * size * size
                        count += 2
                    else:
                        flops[count] += (1 * 1 * input_channel * hidden_dim +
                                         0 * hidden_dim) * size * size
                        flops[count + 1] += (3 * 3 * hidden_dim +
                                             0 * hidden_dim) * size * size
                        flops[count +
                              2] += (1 * 1 * hidden_dim * output_channel +
                                     0 * output_channel) * size * size
                        count += 3
                input_channel = output_channel
        flops[count] += (1 * 1 * input_channel * self.last_channel +
                         0 * self.last_channel) * size * size  # final 1x1 conv
        count += 1
        flops[count] += (
            (2 * self.last_channel - 1) * self.num_classes)  # fc layer
        return flops, first_last_flops

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def feature_extract(self, x, quant_type='PTQ'):
        # layerwise
        tensor = []
        for _layer in self.features:
            if type(_layer) is not InvertedResidual:
                x = _layer(x)
                tensor.append(x)
            elif len(_layer.conv) == 2:
                tensor.append(_layer.conv[0](x))
                tensor.append(_layer.conv(x))
                x = _layer(x)
            else:
                tensor.append(_layer.conv[0](x))
                tensor.append(_layer.conv[1](_layer.conv[0](x)))
                tensor.append(_layer(x))
                x = _layer(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        tensor.append(x)

        return tensor

    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'cfg': self.cfg,
            'cfg_base': self.cfgs_base,
            'dataset': 'ImageNet',
        }

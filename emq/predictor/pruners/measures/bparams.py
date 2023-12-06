import numpy as np

from emq.models.mixed.mobilenet_imagenet import MobileNetV2
from emq.models.mixed.resnet_imagenet import ResNet_ImageNet
from . import measure


@measure('bparams', bn=True)
def get_bparams(net,
                inputs,
                targets,
                loss_fn,
                split_data=1,
                skip_grad=False,
                bit_cfg=None,
                arch='resnet18'):
    assert arch in ['resnet18', 'mobilenetv2', 'resnet50']
    if arch == 'resnet18':
        m = ResNet_ImageNet(depth=18)
        length = 18
    elif arch == 'mobilenetv2':
        m = MobileNetV2()
        length = 53
    elif arch == 'resnet50':
        m = ResNet_ImageNet(depth=50)
        length = 53

    assert len(
        bit_cfg) == length, f'Except Length is {length} but got {len(bit_cfg)}'

    params, first_last_size = m.cfg2params_perlayer(
        cfg=m.cfg, length=len(bit_cfg), quant_type='PTQ')
    params = [i / (1024 * 1024) for i in params]
    first_last_size = first_last_size / (1024 * 1024)

    return np.sum(np.array(bit_cfg) * np.array(params) / 8) + first_last_size


def compute_bparams_per_weight(net,
                               inputs,
                               targets,
                               loss_fn,
                               split_data=1,
                               skip_grad=False,
                               bit_cfg=None,
                               arch='resnet18'):
    assert arch in ['resnet18', 'mobilenetv2', 'resnet50']
    if arch == 'resnet18':
        m = ResNet_ImageNet(depth=18)
        length = 18
    elif arch == 'mobilenetv2':
        m = MobileNetV2()
        length = 53
    elif arch == 'resnet50':
        m = ResNet_ImageNet(depth=50)
        length = 53

    assert len(
        bit_cfg) == length, f'Except Length is {length} but got {len(bit_cfg)}'

    params, first_last_size = m.cfg2params_perlayer(
        cfg=m.cfg, length=len(bit_cfg), quant_type='PTQ')
    params = [i / (1024 * 1024) for i in params]
    first_last_size = first_last_size / (1024 * 1024)

    return np.array(bit_cfg) * np.array(params) / 8

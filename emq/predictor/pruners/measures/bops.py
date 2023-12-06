from emq.models.mixed.mobilenet_imagenet import MobileNetV2
from emq.models.mixed.resnet_imagenet import ResNet_ImageNet
from . import measure


@measure('bops', bn=True)
def get_bops(net,
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

    assert len(bit_cfg) == length

    FLOPs, first_last_flops = m.cfg2flops_layerwise(
        cfg=m.cfg, length=len(bit_cfg), quant_type='PTQ')

    score = (first_last_flops * 8 * 8 + sum(FLOPs[i] * bit_cfg[i] * 5
                                            for i in range(length))) / 1e9

    return score


def compute_bops_per_weight(net,
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

    assert len(bit_cfg) == length

    FLOPs, first_last_flops = m.cfg2flops_layerwise(
        cfg=m.cfg, length=len(bit_cfg), quant_type='PTQ')

    score_list = [FLOPs[i] * bit_cfg[i] * 5 / 1e9 for i in range(length)]

    return score_list

# https://github.com/alibaba/lightweight-neural-architecture-search/blob/main/nas/scores/compute_entropy.py MAE_DET

# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import numpy as np
import torch
from torch import nn

from emq.quant.quant_layer import QuantModule
from . import measure

# def network_weight_gaussian_init(net: nn.Module, std=1):
#     with torch.no_grad():
#         for m in net.modules():
#             if isinstance(m, QuantModule):
#                 if isinstance(m.org_module, nn.Conv2d):
#                     nn.init.normal_(m.org_module.weight, std=std)
#                     if hasattr(m.org_module,
#                                'bias') and m.org_module.bias is not None:
#                         nn.init.zeros_(m.org_module.bias)
#                 elif isinstance(m.org_module, (nn.BatchNorm2d, nn.GroupNorm)):
#                     nn.init.ones_(m.org_module.weight)
#                     nn.init.zeros_(m.org_module.bias)
#                     m.org_module.track_running_stats = True
#                     m.org_module.eps = 1e-5
#                     m.org_module.momentum = 1.0
#                     m.org_module.train()
#                 elif isinstance(m.org_module, nn.Linear):
#                     nn.init.normal_(m.org_module.weight, std=std)
#                     if hasattr(m.org_module,
#                                'bias') and m.org_module.bias is not None:
#                         nn.init.zeros_(m.org_module.bias)
#                 else:
#                     continue
#     return net


@measure('entropy', bn=True)
def compute_entropy_score(
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
    nas_score_list = []

    device = inputs.device
    dtype = torch.half if fp16 else torch.float32

    output_list = []

    def hook_fw_fn(module, input, output):
        output_list.append(output.detach())

    for name, module in net.named_modules():
        if 'conv1' in name:
            continue
        if isinstance(module, (nn.Conv2d, QuantModule)):
            module.register_forward_hook(hook_fw_fn)

    with torch.no_grad():
        for _ in range(repeat):
            # network_weight_gaussian_init(net)
            input = torch.randn(
                size=list(inputs.shape), device=device, dtype=dtype)

            # outputs, logits = net.forward_with_features(input)
            _ = net(input)

            for i, output in enumerate(output_list):
                nas_score = torch.log(output.std(
                ))  # + torch.log(torch.var(output / (output.std() + 1e-9)))
                nas_score_list.append(float(nas_score) * bit_cfg[i])

    avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score

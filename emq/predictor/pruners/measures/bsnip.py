# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import measure


def get_layer_metric_array(net, metric):
    metric_array = []

    for name, module in net.named_modules():
        if 'model.conv1' in name:
            continue
        if 'fc' in name:
            continue
        if isinstance(module, nn.Conv2d):  # and 'conv' in name:
            content = metric(module)
            if content is not None:
                metric_array.append(content)

    return metric_array


def snip_forward_conv2d(self, x):
    return F.conv2d(
        x,
        self.weight * self.weight_mask,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


# def snip_forward_conv2d(self, x):
#     if self.use_weight_quant:
#         weight = self.weight_quantizer(self.weight)
#         bias = self.bias
#     else:
#         weight = self.org_weight
#         bias = self.org_bias

#     out = self.fwd_func(x, weight * self.weight_mask, bias, **self.fwd_kwargs)

#     if self.se_module is not None:
#         out = self.se_module(out)

#     out = self.activation_function(out)
#     if self.disable_act_quant:
#         return out

#     # whether use activation quant
#     if self.use_act_quant:
#         out = self.act_quantizer(out)

#     return out


@measure('bsnip', bn=True, mode='param')
def compute_bsnip_per_weight(net,
                             inputs,
                             targets,
                             mode,
                             loss_fn,
                             split_data=1,
                             bit_cfg=None):
    for layer in net.modules():
        if isinstance(layer, (nn.Conv2d, nn.Conv2d)):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, (nn.Conv2d, nn.Conv2d)):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        # split whole batch into split_data parts.
        outputs = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # select the gradients that we want to use for search/prune
    def snip(layer):
        if layer.weight_mask.grad is not None:
            return torch.abs(layer.weight_mask.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, snip)

    syn_list = []
    for g, b in zip(grads_abs, bit_cfg):
        syn_list.append(g * b)

    return syn_list

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

import torch
import torch.nn as nn

from . import measure
from .bn_score import network_weight_gaussian_init


def get_layer_metric_array(net, metric):
    metric_array = []

    for name, module in net.named_modules():
        if 'fc' in name:
            continue
        if isinstance(module, nn.Conv2d):  # and 'conv' in name:
            content = metric(module)
            if content is not None:
                metric_array.append(content)

    return metric_array


@measure('bsynflow', bn=False, mode='param')
@measure('bsynflow_bn', bn=True, mode='param')
def compute_bsynflow_per_weight(net,
                                inputs,
                                targets,
                                mode,
                                split_data=1,
                                loss_fn=None,
                                bit_cfg=None):

    net = network_weight_gaussian_init(net)

    device = inputs.device

    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net.double())

    # Compute gradients with input of 1s
    net.zero_grad()
    net.double()
    input_dim = list(inputs[0, :].shape)
    # inputs = torch.ones([1] + input_dim).to(device)
    inputs = torch.ones([1] + input_dim).double().to(device)
    output = net.forward(inputs)
    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return None

    grads_abs = get_layer_metric_array(net, synflow)

    # apply signs of all params
    nonlinearize(net, signs)

    # ignore the last one
    grads_abs = grads_abs[:-1]

    syn_list = []
    for g, b in zip(grads_abs, bit_cfg):
        syn_list.append(g * b)

    return syn_list

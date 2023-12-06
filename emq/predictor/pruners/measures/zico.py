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

from ..p_utils import get_layer_metric_array
from . import measure


@measure('zico', bn=True, mode='param')
def compute_zico_per_weight(net, inputs, targets, mode, loss_fn, split_data=1):

    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        outputs = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # select the gradients that we want to use for search/prune
    def zico(layer):
        if layer.weight.grad is not None:
            up = torch.abs(torch.mean(layer.weight.grad))
            down = torch.sqrt(torch.var(layer.weight.grad)) + 1e-9
            return torch.sum(up / down)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, zico, mode)
    return torch.log(torch.sum(torch.tensor(grads_abs)))

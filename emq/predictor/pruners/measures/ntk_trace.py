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

import numpy as np
import torch

import emq.utils.autograd_hacks as autograd_hacks
from . import measure


@measure('ntk_trace', bn=True, mode='param')
def compute_ntk_trace(net, inputs, targets, mode, loss_fn, split_data=1):
    # Compute gradients (but don't apply them)
    net.zero_grad()
    N = inputs.shape[0]
    grads = []

    autograd_hacks.add_hooks(net)
    outputs = net.forward(inputs)
    sum(outputs[torch.arange(N), targets]).backward()
    autograd_hacks.compute_grad1(net, loss_type='sum')

    grads = [
        param.grad1.flatten(start_dim=1) for param in net.parameters()
        if hasattr(param, 'grad1')
    ]
    grads = torch.cat(grads, axis=1)

    ntk = torch.matmul(grads, grads.t())
    eigenvalues, _ = torch.symeig(ntk)  # ascending
    return np.sqrt(eigenvalues.cpu().numpy().sum() / N)

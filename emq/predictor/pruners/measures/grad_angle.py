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

from ..p_utils import get_flattened_metric
from . import measure


@measure('grad_angle', bn=True)
def get_grad_angle(net,
                   inputs,
                   targets,
                   loss_fn,
                   split_data=1,
                   skip_grad=False):
    N = inputs.shape[0]
    batch_grad = []
    for i in range(N):
        net.zero_grad()
        outputs = net.forward(inputs[[i]])
        loss = loss_fn(outputs, targets[[i]])
        loss.backward(retain_graph=True)
        flattened_grad = get_flattened_metric(
            net, lambda l: l.weight.grad.data.cpu().numpy()
            if l.weight.grad is not None else torch.zeros_like(l.weight).numpy(
            ))
        batch_grad.append(flattened_grad)
    batch_grad = np.stack(batch_grad)  # (b, d)
    # score = abs(batch_grad.sum(axis=0)).mean()

    norm_scale = np.linalg.norm(batch_grad, axis=1, keepdims=True)  # (b)
    norm_batch_grad = batch_grad / norm_scale
    mean_grad = norm_batch_grad.mean(axis=0)  # (d)
    dot_prod = norm_batch_grad @ mean_grad
    norm_prod = np.linalg.norm(mean_grad) * \
        np.linalg.norm(norm_batch_grad, axis=1)
    angle = dot_prod / norm_prod
    score = angle.mean()
    return score

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


# proposed in gradsign
@measure('grad_conflict', bn=True)
def get_grad_conflict(net,
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
            net, lambda l: l.weight.grad.data.cpu().numpy() if l.weight.grad is
            not None else torch.zeros_like(l.weight).cpu().numpy(), False)
        batch_grad.append(flattened_grad)
    batch_grad = np.stack(batch_grad)
    direction_code = np.sign(batch_grad)
    direction_code = abs(direction_code.sum(axis=0))
    score = np.nansum(direction_code)
    return score

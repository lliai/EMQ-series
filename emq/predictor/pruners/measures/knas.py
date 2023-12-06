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

import copy

import torch

from . import measure


# proposed in knas green neural architecture search
@measure('mgm', bn=True)
def get_mgm_score(net,
                  inputs,
                  targets,
                  loss_fn,
                  split_data=1,
                  skip_grad=False):
    N = inputs.shape[0]
    grads = {}
    g = 0
    para = 0
    for i in range(N):
        net.zero_grad()
        outputs = net.forward(inputs[[i]])
        loss = loss_fn(outputs, targets[[i]])
        loss.backward()

        index_name = 0
        for name, param in net.named_parameters():
            if param.grad is None:
                continue

            if index_name > 10:
                break

            if len(param.grad.view(-1).data[:100]) < 50:
                continue

            index_name += 1

            if name in grads:
                grads[name].append(copy.copy(param.grad.view(-1).data[:100]))
            else:
                grads[name] = [copy.copy(param.grad.view(-1).data[:100])]

    for name in grads:
        for i in range(len(grads[name])):
            grad1 = torch.tensor(grads[name][i][:25].clone().detach())
            grad2 = torch.tensor(grads[name][i][25:50].clone().detach())
            grad1 = grad1 - grad1.mean()
            grad2 = grad2 - grad2.mean()
            g += torch.dot(grad1, grad2) / 2500
            para += 1

    return g / para

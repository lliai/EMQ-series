import torch
import torch.nn as nn
from torch.autograd import Variable

import emq.models.hubconf as hubconf  # noqa: F401, F403
from emq.dataset.imagenet import build_imagenet_data
from emq.operators.zc_inputs import *  # noqa: F401, F403
from emq.quant import QuantModel

cnn = eval('hubconf.resnet18(pretrained=False)')
cnn.eval()

input = torch.randn(7, 3, 224, 224)

input = Variable(input, requires_grad=True)

act_list = compute_activation(
    cnn,
    inputs=input,
    targets=torch.empty(7, dtype=torch.long).random_(4),
    loss_fn=nn.CrossEntropyLoss())

print(len(act_list))
for act in act_list:
    print('act:', act.shape)

grad_list = compute_gradient(
    cnn,
    inputs=input,
    targets=torch.empty(7, dtype=torch.long).random_(4),
    loss_fn=nn.CrossEntropyLoss())

print(len(grad_list))
for grad in grad_list:
    print('grad:', grad.shape)

weight_list = compute_weight(
    cnn,
    inputs=input,
    targets=torch.empty(7, dtype=torch.long).random_(4),
    loss_fn=nn.CrossEntropyLoss())

print(len(weight_list))
for weight in weight_list:
    print('weight:', weight.shape)

# virtual_list = compute_virtual_grad(
#     cnn,
#     inputs=input,
#     targets=torch.empty(7, dtype=torch.long).random_(4),
#     loss_fn=nn.CrossEntropyLoss())

# print(len(virtual_list))
# for virtual in virtual_list:
#     print('virtual:', virtual.shape)

# hessian_list = compute_hessian_trace(
#     net=cnn,
#     inputs=input,
#     targets=torch.empty(7, dtype=torch.long).random_(4),
#     loss_fn=nn.CrossEntropyLoss())

# print(type(hessian_list[0]))
# print(len(hessian_list))

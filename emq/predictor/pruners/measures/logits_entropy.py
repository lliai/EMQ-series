# https://github.com/ICLRsubmission1596/Regularizing-Deep-Neural-Networks-with-Stochastic-Estimators-of-Hessian-Trace/blob/main/Cifar10/cifar_resnet_confidence.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import measure


class HLoss(nn.Module):

    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


@measure('logits_entropy', bn=True)
def compute_hessian_trace(net, inputs, targets, split_data=1, loss_fn=None):
    logits = net(inputs)
    # 10 is the number of classes.
    logits = F.layer_norm(logits, torch.Size((1000, )), eps=1e-7)
    criterion = HLoss()
    ht = criterion(logits)
    return ht.detach().item()

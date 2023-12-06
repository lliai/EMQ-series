# https://github.com/idstcv/ZenNAS/blob/main/ZeroShotProxy/compute_te_nas_score.py

import numpy as np
import torch

from . import measure


def recal_bn(network, xloader, recalbn, device):
    for m in network.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.fill_(0)
            m.running_var.data.fill_(0)
            m.num_batches_tracked.data.zero_()
            m.momentum = None
    network.train()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(xloader):
            if i >= recalbn:
                break
            inputs = inputs.cuda(device=device, non_blocking=True)
            _, _ = network(inputs)
    return network


def get_ntk_n(networks,
              recalbn=0,
              train_mode=False,
              num_batch=None,
              batch_size=None,
              image_size=None,
              device=None):
    if device is None:
        device = torch.device('cpu')

    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()

    grads = [[] for _ in range(len(networks))]

    for i in range(num_batch):
        inputs = torch.randn((batch_size, 3, image_size, image_size),
                             device=device)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            inputs_ = inputs.clone()

            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]
            for _idx in range(len(inputs_)):
                logit[_idx:_idx + 1].backward(
                    torch.ones_like(logit[_idx:_idx + 1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                # if gpu is not None:
                #     torch.cuda.empty_cache()

    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        conds.append(
            np.nan_to_num(
                (eigenvalues[-1] / eigenvalues[0]).item(), copy=True))
    return conds


@measure('ntk', bn=True)
def compute_ntk_score(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
):
    device = inputs.device
    resolution = inputs.shape[2]
    batch_size = inputs.shape[0]

    ntk_score = get_ntk_n([net],
                          recalbn=0,
                          train_mode=True,
                          num_batch=1,
                          batch_size=batch_size,
                          image_size=resolution,
                          device=device)[0]
    return -1 * ntk_score

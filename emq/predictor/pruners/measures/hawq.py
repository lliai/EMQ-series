# Largely borrowed from MQBench https://github.com/ModelTC/MQBench

from typing import Dict

import numpy as np
import torch

from emq.utils.hessian import (group_product, hessian, hessian_vector_product,
                               normalization)
from . import measure


class hessian_per_layer(hessian):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_order_grad_dict = {}

        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                self.first_order_grad_dict[name] = mod.weight.grad + 0.

    def layer_eigenvalues(self, maxIter=100, tol=1e-3) -> Dict:
        """
        compute the max eigenvalues in one model by layer.
        """
        device = self.device
        max_eigenvalues_dict = {}

        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                weight = mod.weight
                eigenvalue = None
                v = [torch.randn(weight.size()).to(device)]
                v = normalization(v)
                first_order_grad = self.first_order_grad_dict[name]

                for i in range(maxIter):
                    self.model.zero_grad()

                    Hv = hessian_vector_product(first_order_grad, weight, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                    v = normalization(Hv)

                    if eigenvalue is None:
                        eigenvalue = tmp_eigenvalue
                    else:
                        if abs(eigenvalue - tmp_eigenvalue) / (
                                abs(eigenvalue) + 1e-6) < tol:
                            break
                        else:
                            eigenvalue = tmp_eigenvalue
                max_eigenvalues_dict[name] = eigenvalue

        return max_eigenvalues_dict

    def layer_trace(self, maxIter=100, tol=1e-3) -> Dict:
        """
        Compute the trace of hessian in one model by layer.
        """
        device = self.device
        trace_dict = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                trace_vhv = []
                trace = 0.
                weight = mod.weight
                first_order_grad = self.first_order_grad_dict[name]
                for i in range(maxIter):
                    self.model.zero_grad()
                    v = torch.randint_like(weight, high=2, device=device)
                    # generate Rademacher random variables
                    v[v == 0] = -1
                    v = [v]

                    Hv = hessian_vector_product(first_order_grad, weight, v)
                    trace_vhv.append(group_product(Hv, v).cpu().item())
                    if abs(np.mean(trace_vhv) - trace) / (abs(trace) +
                                                          1e-6) < tol:
                        break
                    else:
                        trace = np.mean(trace_vhv)
                trace_dict[name] = trace
        return trace_dict


@measure('hawqv1', bn=True)
def compute_hawqv1(net,
                   inputs,
                   targets,
                   split_data=1,
                   loss_fn=None,
                   bit_cfg=None):

    cuda = bool(torch.cuda.is_available())
    hessian_comp = hessian_per_layer(
        net, loss_fn, data=(inputs, targets), cuda=cuda)

    res_dict = hessian_comp.layer_eigenvalues()

    all_res = [v for k, v in res_dict.items()]
    if len(bit_cfg) == 18:  # resnet18
        all_res = all_res[1:-1]
    else:  # mobilenetv2
        bit_cfg = bit_cfg[1:]

    if bit_cfg is not None:
        assert len(bit_cfg) == len(
            all_res), f'{len(bit_cfg)} != {len(all_res)}'
    else:
        bit_cfg = [1] * len(all_res)
    return [a * b for a, b in zip(all_res, bit_cfg)]


@measure('hawqv2', bn=True)
def compute_hawqv2(net,
                   inputs,
                   targets,
                   split_data=1,
                   loss_fn=None,
                   bit_cfg=None):

    cuda = bool(torch.cuda.is_available())
    hessian_comp = hessian_per_layer(
        net, loss_fn, data=(inputs, targets), cuda=cuda)

    res_dict = hessian_comp.layer_trace()

    all_res = [v for k, v in res_dict.items()]
    if len(bit_cfg) == 18:  # resnet18
        all_res = all_res[1:-1]
    else:  # mobilenetv2
        bit_cfg = bit_cfg[1:]

    if bit_cfg is not None:
        assert len(bit_cfg) == len(
            all_res), f'{len(bit_cfg)} != {len(all_res)}'
    else:
        bit_cfg = [1] * len(all_res)
    return [a * b for a, b in zip(all_res, bit_cfg)]

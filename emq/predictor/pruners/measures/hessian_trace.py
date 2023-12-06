import numpy as np
import torch

from . import measure


def hessian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, _ = torch.autograd.grad(
            flat_y,
            x,
            grad_y,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum(torch.sum(x * y) for (x, y) in zip(xs, ys))


@measure('hessian_trace', bn=True)
def compute_hessian_trace(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    prob = 0.01
    output = net(inputs)
    params = [output]
    for _, param in net.named_parameters():
        if param.requires_grad:
            # param= param.reshape(-1)
            p = np.random.binomial(1, prob)
            if p == 1:
                params.append(param)

    loss = loss_fn(output, targets)
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, create_graph=True)
    grad_list = []
    for i in grads:
        grad_list.append(i)

    # calculate hessian trace
    trace = 0.
    if len(grad_list) > 0:
        for i in range(1):
            with torch.no_grad():
                v = [
                    torch.randint_like(p, high=1, device=device)
                    for p in params
                ]
                for v_i in v:
                    v_i[v_i == 0] = np.random.binomial(1, prob * 2)
                for v_i in v:
                    v_i[v_i == 1] = 2 * np.random.binomial(1, 0.5) - 1
            Hv = torch.autograd.grad(
                grad_list,
                params,
                grad_outputs=v,
                only_inputs=True,
                retain_graph=True)
            hessian_tr = group_product(Hv, v).cpu().item()
            trace += hessian_tr
    return trace

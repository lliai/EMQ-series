# REGULARIZING DEEP NEURAL NETWORKS WITH STOCHASTIC ESTIMATORS OF HESSIAN TRACE

import torch
import torch.nn as nn

from . import measure


class JacobianReg(nn.Module):
    '''
    Loss criterion that computes the trace of the square of the Jacobian.

    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output
            space and projection is non-random and orthonormal, yielding
            the exact result.  For any reasonable batch size, the default
            (n=1) should be sufficient.
    '''

    def __init__(self, n=-1):
        assert n == -1 or n > 0
        self.n = n
        super(JacobianReg, self).__init__()

    def forward(self, x, y):
        '''
        x: data; y: output
        computes (1/2) tr |dy/dx|^2
        '''
        B, C = y.shape
        if self.n == -1:
            num_proj = C
        else:
            num_proj = self.n
        J2 = 0
        for ii in range(num_proj):
            if self.n == -1:
                # orthonormal vector, sequentially spanned
                v = torch.zeros(B, C)
                v[:, ii] = 1
            else:
                # random properly-normalized vector for each sample
                v = self._random_vector(C=C, B=B)
            if x.is_cuda:
                v = v.cuda()
            Jv = self._jacobian_vector_product(y, x, v, create_graph=True)
            J2 += C * torch.norm(Jv)**2 / (num_proj * B)
        R = (1 / 2) * J2
        return R

    def _random_vector(self, C, B):
        '''
        creates a random vector of dimension C with a norm of C^(1/2)
        (as needed for the projection formula to work)
        '''
        if C == 1:
            return torch.ones(B)
        v = torch.randn(B, C)
        arxilirary_zero = torch.zeros(B, C)
        vnorm = torch.norm(v, 2, 1, True)
        v = torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
        return v

    def _jacobian_vector_product(self, y, x, v, create_graph=False):
        '''
        Produce jacobian-vector product dy/dx dot v.

        Note that if you want to differentiate it,
        you need to make create_graph=True
        '''
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        grad_x, = torch.autograd.grad(
            flat_y, x, flat_v, retain_graph=True, create_graph=create_graph)
        return grad_x


@measure('jacobian_trace', bn=True)
def compute_jacobian_trace(net, inputs, targets, split_data=1, loss_fn=None):
    # Compute gradients (but don't apply them)
    inputs.requires_grad = True
    reg_full = JacobianReg(n=-1)
    output = net(inputs)
    jt = reg_full(inputs, output)
    del inputs
    del output
    return jt.detach()

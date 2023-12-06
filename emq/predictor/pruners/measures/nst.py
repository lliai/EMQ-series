# Copyright (C) 2010-2021 Alibaba Group Holding Limited.
# =============================================================================

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import measure


@measure('nst', bn=True)
def compute_nst_score(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
):
    nas_score_list = []

    def nst(fm):
        fm = fm.view(fm.shape[0], fm.shape[1], -1)
        fm = F.normalize(fm, dim=2)
        return fm.sum(-1).pow(2)

    def sp1(fm):
        fm = fm.view(fm.shape[0], -1)
        fm = torch.mm(fm, fm.t())
        return F.normalize(fm, p=2, dim=1)

    def at(fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), 2)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)
        return am

    def sp2(fm):
        fm = fm.view(fm.size(0), -1)
        fm = torch.mm(fm, fm.t())
        norm_G_s = F.normalize(fm, p=2, dim=1)
        return norm_G_s.pow(2).sum(dim=1)

    def pdist(fm, squared=False, eps=1e-12):
        feat_square = fm.pow(2).sum(dim=1)
        feat_prod = torch.mm(fm, fm.t())
        feat_dist = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) -
                     2 * feat_prod).clamp(min=eps)
        if not squared:
            feat_dist = feat_dist.sqrt()
        feat_dist = feat_dist.clone()
        feat_dist[range(len(fm)), range(len(fm))] = 0

        return feat_dist

    def rkd_angle(fm):
        # N x C --> N x N x C
        feat_t_vd = (fm.unsqueeze(0) - fm.unsqueeze(1))
        norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
        feat_t_angle = torch.bmm(norm_feat_t_vd,
                                 norm_feat_t_vd.transpose(1, 2)).view(-1)
        return feat_t_angle

    def cc(fm):
        P_order = 2
        gamma = 0.4
        fm = F.normalize(fm, p=2, dim=-1)
        sim_mat = torch.matmul(fm, fm.t())
        corr_mat = torch.zeros_like(sim_mat)
        for p in range(P_order + 1):
            corr_mat += math.exp(-2 * gamma) * (2 * gamma) ** p / \
                math.factorial(p) * torch.pow(sim_mat, p)
        return corr_mat

    def ickd(fm):
        bsz, ch = fm.shape[0], fm.shape[1]
        fm = fm.view(bsz, ch, -1)
        emd_s = torch.bmm(fm, fm.permute(0, 2, 1))
        emd_s = torch.nn.functional.normalize(emd_s, dim=2)

        G_diff = emd_s
        loss = (G_diff * G_diff).view(bsz, -1).sum() / (ch * bsz)
        return loss

    single_kd = ickd

    with torch.no_grad():
        output, logits = net.forward_with_features(inputs)
        nas_score_list = [
            torch.sum(single_kd(f)).detach().cpu().numpy()
            for f in output[1:-1]
        ]

        avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score

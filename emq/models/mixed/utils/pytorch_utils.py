import math
import sys

import numpy as np
import torch
import torch.nn as nn


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


def raw2cfg(model, raw_ratios, flops, p=False, div=8):
    left = 0
    right = 50
    scale = 0
    cfg = None
    current_flops = 0
    base_channels = model.config['cfg_base']
    cnt = 0
    while (True):
        cnt += 1
        scale = (left + right) / 2
        scaled_ratios = raw_ratios * scale
        for i in range(len(scaled_ratios)):
            scaled_ratios[i] = max(0.1, scaled_ratios[i])
            scaled_ratios[i] = min(1, scaled_ratios[i])
        cfg = (base_channels * scaled_ratios).astype(int).tolist()
        for i in range(len(cfg)):
            cfg[i] = _make_divisible(cfg[i], div)  # 8 divisible channels
        current_flops = model.cfg2flops(cfg)
        if cnt > 20:
            break
        if abs(current_flops - flops) / flops < 0.01:
            break
        if p:
            print(
                str(current_flops) + '---' + str(flops) + '---left: ' +
                str(left) + '---right: ' + str(right) + '---cfg: ' + str(cfg))
        if current_flops < flops:
            left = scale
        elif current_flops > flops:
            right = scale
        else:
            break
    return cfg


def weight2mask(weight, keep_c):  # simple L1 pruning
    weight_copy = weight.abs().clone()
    L1_norm = torch.sum(weight_copy, dim=(1, 2, 3))
    arg_max = torch.argsort(L1_norm, descending=True)
    arg_max_rev = arg_max[:keep_c].tolist()
    mask = np.zeros(weight.shape[0])
    mask[arg_max_rev] = 1
    return mask


def get_unpruned_weights(model, model_origin):
    masks = []
    for [m0, m1] in zip(model_origin.named_modules(), model.named_modules()):
        if isinstance(m0[1], nn.Conv2d):
            if m0[1].weight.data.shape != m1[1].weight.data.shape:
                flag = False
                if m0[1].weight.data.shape[1] != m1[1].weight.data.shape[1]:
                    assert len(masks) > 0, 'masks is empty!'
                    if m0[0].endswith('downsample.conv'):
                        if model.config['depth'] >= 50:
                            mask = masks[-4]
                        else:
                            mask = masks[-3]
                    else:
                        mask = masks[-1]
                    idx = np.squeeze(np.argwhere(mask))
                    if idx.size == 1:
                        idx = np.resize(idx, (1, ))
                    w = m0[1].weight.data[:, idx.tolist(), :, :].clone()
                    flag = True
                    if m0[1].weight.data.shape[0] == m1[1].weight.data.shape[
                            0]:
                        masks.append(None)
                if m0[1].weight.data.shape[0] != m1[1].weight.data.shape[0]:
                    if m0[0].endswith('downsample.conv'):
                        mask = masks[-1]
                    else:
                        if flag:
                            mask = weight2mask(w.clone(),
                                               m1[1].weight.data.shape[0])
                        else:
                            mask = weight2mask(m0[1].weight.data,
                                               m1[1].weight.data.shape[0])
                    idx = np.squeeze(np.argwhere(mask))
                    if idx.size == 1:
                        idx = np.resize(idx, (1, ))
                    if flag:
                        w = w[idx.tolist(), :, :, :].clone()
                    else:
                        w = m0[1].weight.data[idx.tolist(), :, :, :].clone()
                    m1[1].weight.data = w.clone()
                    masks.append(mask)
                continue
            else:
                m1[1].weight.data = m0[1].weight.data.clone()
                masks.append(None)
        elif isinstance(m0[1], nn.BatchNorm2d):
            assert isinstance(
                m1[1], nn.BatchNorm2d), 'There should not be bn layer here.'
            if m0[1].weight.data.shape != m1[1].weight.data.shape:
                mask = masks[-1]
                idx = np.squeeze(np.argwhere(mask))
                if idx.size == 1:
                    idx = np.resize(idx, (1, ))
                m1[1].weight.data = m0[1].weight.data[idx.tolist()].clone()
                m1[1].bias.data = m0[1].bias.data[idx.tolist()].clone()
                m1[1].running_mean = m0[1].running_mean[idx.tolist()].clone()
                m1[1].running_var = m0[1].running_var[idx.tolist()].clone()
                continue
            m1[1].weight.data = m0[1].weight.data.clone()
            m1[1].bias.data = m0[1].bias.data.clone()
            m1[1].running_mean = m0[1].running_mean.clone()
            m1[1].running_var = m0[1].running_var.clone()


# noinspection PyUnresolvedReferences
def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.LogSoftmax(dim=1)
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 -
                                 label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    return total_params


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        return tuple([detach_variable(x) for x in inputs])
    else:
        x = inputs.detach()
        x.requires_grad = inputs.requires_grad
        return x


def accuracy(output, target, topk=(1, )):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def DFS_bit(value, weight):
    # value = [1.2, 3.8, 4.3, 7.9]
    # weight = [100, 10, 50, 30]
    thresh = sum(a * b for a, b in zip(value, weight))
    best = thresh
    ans = None

    def dfs(index, way, cur_value, cur_queue):
        nonlocal best, thresh, ans
        if index > len(value) - 1:
            return
        if way == 'ceil':
            v = math.ceil(value[index])
        # elif way == "ceil+1":
        #     v = math.ceil(value[index])+1
        elif way == 'floor':
            v = math.floor(value[index])
        # elif way == "floor-1":
        #     v = math.floor(value[index])-1

        cur_value += v * weight[index]
        if cur_value > thresh:
            return

        cur_queue.append(v)
        if index == len(value) - 1:
            if abs(cur_value - thresh) < best:
                # print("find a solution:")
                # print(cur_queue, abs(cur_value - thresh))
                ans = cur_queue.copy()
                best = abs(cur_value - thresh)

        # dfs(index + 1, "ceil+1", cur_value, cur_queue)
        dfs(index + 1, 'ceil', cur_value, cur_queue)
        dfs(index + 1, 'floor', cur_value, cur_queue)
        # dfs(index + 1, "floor-1", cur_value, cur_queue)
        cur_queue.pop()

    # print(f"the thresh of our problem is {thresh}")
    # dfs(0, "ceil+1", 0, [])
    dfs(0, 'ceil', 0, [])
    dfs(0, 'floor', 0, [])
    # dfs(0, "floor-1", 0, [])
    # print("-"*100)
    # print("the result is:")
    # print(ans)
    # print(best)
    return ans, best

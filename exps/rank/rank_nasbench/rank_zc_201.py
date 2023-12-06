import argparse
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from nas_201_api import NASBench201API

from emq.dataset.cifar10 import get_cifar10_dataloaders
from emq.dataset.cifar100 import get_cifar100_dataloaders
from emq.dataset.imagenet16 import get_imagenet16_dataloaders
from emq.models.nasbench201.utils import dict2config, get_cell_based_tiny_net
from emq.predictor.pruners import predictive
from emq.utils.rank_consistency import kendalltau, pearson, spearman

nb201_api = NASBench201API(
    file_path_or_dict='data/NAS-Bench-201-v1_1-096897.pth', verbose=False)


def random_sample_and_get_gt():
    index_range = list(range(15625))
    choiced_index = random.choice(index_range)
    # modelinfo is a index

    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': nb201_api.arch(choiced_index),
        'num_classes': NUM_CLASSES
    }
    net_config = dict2config(arch_config, None)
    model = get_cell_based_tiny_net(net_config)
    xinfo = nb201_api.get_more_info(choiced_index, dataset=TARGET, hp='200')
    return choiced_index, model, xinfo['test-accuracy']


def gen_fixed_and_get_gt(choiced_index, zen=False):
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': nb201_api.arch(choiced_index),
        'num_classes': 10,
        'zen': zen,
    }
    net_config = dict2config(arch_config, None)
    model = get_cell_based_tiny_net(net_config, zen=zen)
    xinfo = nb201_api.get_more_info(choiced_index, dataset='cifar10', hp='200')
    return choiced_index, model, xinfo['test-accuracy']


def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is None:
                    continue
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue
    return net


def compute_zero_cost_proxies_rank_procedure(dataloader,
                                             zc_name,
                                             sampled_idx_list=None,
                                             sample_number=50,
                                             is_zen=False):
    gt_list = []
    zcs_list = []

    chosen_idx = []
    for i in range(sample_number):
        # get random subnet
        idx, net, acc = gen_fixed_and_get_gt(sampled_idx_list[i], zen=is_zen)
        if idx not in chosen_idx:
            chosen_idx.append(idx)
        else:
            idx, net, acc = gen_fixed_and_get_gt(
                sampled_idx_list[i], zen=is_zen)

        network_weight_gaussian_init(net)

        if zc_name == 'grasp':
            dataload_info = ['grasp', 3, NUM_CLASSES]
        else:
            dataload_info = ['random', 3, NUM_CLASSES]
        gt_list.append(acc)

        score = predictive.find_measures(
            net,
            dataloader,
            dataload_info=dataload_info,
            measure_names=[zc_name],
            loss_fn=F.cross_entropy,
            device=torch.device('cuda'))
        print(f"The {i}-th network 's zc = {score}")
        zcs_list.append(score)

    def convert(item):
        return item[0] if isinstance(item, list) else item

    kd = kendalltau(zcs_list, gt_list)
    sp = spearman(zcs_list, gt_list)
    ps = pearson(zcs_list, gt_list)

    kd, sp, ps = convert(kd), convert(sp), convert(ps)

    print(f'{zc_name} kd: {kd:.4f} sp: {sp:.4f} ps: {ps:.4f}')

    info = {'kd': kd, 'sp': sp, 'ps': ps}
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument(
        '--target', type=str, default='cifar10', help='print frequency')
    parser.add_argument(
        '--zc', type=str, default='nwot', help='zero cost metric name')
    opt = parser.parse_args()

    TARGET = opt.target

    if os.path.exists('./data/sampled_subnet_idx_50.txt'):
        with open('./data/sampled_subnet_idx_50.txt', 'r') as f:
            sampled_idx_str = f.readline()
            splited_idx_str = sampled_idx_str.split(',')
            sampled_idx_list = [int(item.strip()) for item in splited_idx_str]
    else:
        sampled_idx_list = None

    if TARGET == 'cifar100':
        NUM_CLASSES = 100
    elif TARGET == 'ImageNet16-120':
        NUM_CLASSES = 120
    elif TARGET == 'cifar10':
        NUM_CLASSES = 10

    print(f'TARGET: {TARGET} / NUM_CLASSES: {NUM_CLASSES}')
    if TARGET == 'cifar100':
        train_loader, _ = get_cifar100_dataloaders(
            batch_size=16, num_workers=0)
    elif TARGET == 'ImageNet16-120':
        train_loader, _ = get_imagenet16_dataloaders(
            batch_size=16, num_workers=0)
    elif TARGET == 'cifar10':
        train_loader, _ = get_cifar10_dataloaders(batch_size=16, num_workers=0)

    # COMPUTE NORMAL KD RANK CONSISTENCY
    times = 1

    # COMPUTE ZC PROXIES RANK CONSISTENCY
    zc_dict = {}
    # zc_name_list = [
    #     'epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot',
    #     'plain', 'snip', 'synflow', 'flops', 'params', 'zen', 'bn_score'
    # ]
    # zc_name_list = ['grad_conflict', 'grad_angle', 'size',
    # 'logsynflow', 'zico' ,'entropy', 'ntk', 'linear_region',
    # 'nst', 'jacobian_trace', 'hessian_trace', 'logits_entropy']
    zc_name_list = [opt.zc]

    for zc_name in zc_name_list:
        every_time_sp = []
        every_time_kd = []
        every_time_ps = []
        for _ in range(times):
            info = compute_zero_cost_proxies_rank_procedure(
                train_loader,
                zc_name,
                sampled_idx_list,
                is_zen=opt.zc == 'zen')
            every_time_sp.append(info['sp'])
            every_time_kd.append(info['kd'])
            every_time_ps.append(info['ps'])

        print(f'ZC: {zc_name}')
        print(f'kendall: {every_time_kd}')
        print(f'spearman: {every_time_sp}')
        print(f'pearson: {every_time_ps}')

        zc_dict[zc_name] = {
            'kd': every_time_kd,
            'sp': every_time_sp,
            'ps': every_time_ps,
        }

    with open(f'./output/zc_name-result_x{times}_{TARGET}.txt', 'w') as f:
        f.write(json.dumps(zc_dict))

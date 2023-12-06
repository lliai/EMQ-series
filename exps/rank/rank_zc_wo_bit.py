import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import emq.models.hubconf as hubconf
from emq.api import EMQAPI
from emq.dataset.imagenet import build_imagenet_data
from emq.predictor.pruners import predictive
from emq.quant import QuantModel
from emq.quant.quant_layer import QuantModule
from emq.utils.rank_consistency import kendalltau, pearson, spearman

emqapi = EMQAPI('./data/PTQ-GT.pkl', verbose=True)


def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]


def initialize_quant_model(net):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, QuantModule)):
                device = m.weight.device
                if len(m.weight.shape) < 4:
                    continue
                in_channels, out_channels, k1, k2 = m.weight.shape
                m.weight[:] = torch.randn(
                    m.weight.shape, device=device) / np.sqrt(
                        k1 * k2 * in_channels)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                device = m.weight.device
                in_channels, out_channels = m.weight.shape
                m.weight[:] = torch.randn(
                    m.weight.shape, device=device) / np.sqrt(in_channels)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue
    return net


def gen_args():
    parser = argparse.ArgumentParser(
        description='running parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument(
        '--seed',
        default=1005,
        type=int,
        help='random seed for results reproduction')
    parser.add_argument(
        '--arch',
        default='resnet18',
        type=str,
        help='dataset name',
        choices=['resnet18', 'mobilenetv2'])
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
        help='mini-batch size for data loader')
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        help='number of workers for data loader')
    parser.add_argument(
        '--data_path',
        default='/home/dongpeijie/share/dataset/imagenet-mini',
        type=str,
        help='path to ImageNet data')

    # quantization parameters
    parser.add_argument(
        '--n_bits_w',
        default=8,
        type=int,
        help='bitwidth for weight quantization')
    parser.add_argument(
        '--channel_wise',
        action='store_true',
        help='apply channel_wise quantization for weights')
    parser.add_argument(
        '--n_bits_a',
        default=8,
        type=int,
        help='bitwidth for activation quantization')
    parser.add_argument(
        '--act_quant',
        action='store_true',
        help='apply activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parser.add_argument('--test_before_calibration', action='store_true')
    parser.add_argument('--bit_cfg', type=str, default='None')

    # weight calibration parameters
    parser.add_argument(
        '--num_samples',
        default=1024,
        type=int,
        help='size of the calibration dataset')
    parser.add_argument(
        '--iters_w',
        default=20000,
        type=int,
        help='number of iteration for adaround')
    parser.add_argument(
        '--weight',
        default=0.01,
        type=float,
        help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument(
        '--sym',
        action='store_true',
        help='symmetric reconstruction, not recommended')
    parser.add_argument(
        '--b_start',
        default=20,
        type=int,
        help='temperature at the beginning of calibration')
    parser.add_argument(
        '--b_end',
        default=2,
        type=int,
        help='temperature at the end of calibration')
    parser.add_argument(
        '--warmup',
        default=0.2,
        type=float,
        help='in the warmup period no regularization is applied')
    parser.add_argument(
        '--step', default=20, type=int, help='record snn output per step')
    parser.add_argument(
        '--use_bias',
        action='store_true',
        help='fix weight bias and variance after quantization')
    parser.add_argument(
        '--vcorr', action='store_true', help='use variance correction')
    parser.add_argument(
        '--bcorr', action='store_true', help='use bias correction')

    # activation calibration parameters
    parser.add_argument(
        '--iters_a',
        default=5000,
        type=int,
        help='number of iteration for LSQ')
    parser.add_argument(
        '--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument(
        '--p', default=2.4, type=float, help='L_p norm minimization for LSQ')
    parser.add_argument(
        '--zc_name',
        default='bsynflow',
        type=str,
        help='name of zero centering')

    return parser.parse_args()


if __name__ == '__main__':
    args = gen_args()

    # seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    # build imagenet data loader
    train_loader, test_loader = build_imagenet_data(
        batch_size=args.batch_size,
        workers=args.workers,
        data_path=args.data_path)

    # load model
    cnn = eval(f'hubconf.{args.arch}(pretrained=False)')
    cnn.cuda()
    cnn.eval()

    num_samples = 50
    gt_list = []
    zc_list = []

    # zc_names = [
    #     'epe_nas', 'fisher', 'grad_norm', 'jacov', 'l2_norm', 'nwot', 'plain',
    #     'snip', 'synflow', 'flops', 'params', 'zen', 'bn_score',
    #     'grad_conflict', 'grad_angle', 'size', 'logsynflow', 'zico', 'entropy',
    #     'ntk', 'linear_region', 'nst', 'jacobian_trace', 'hessian_trace',
    #     'logits_entropy'
    # ]

    # grasp OOM

    if args.zc_name == 'grasp':
        dataload_info = ['grasp', 1, 1000]
    else:
        dataload_info = ['random', 1, 1000]

    t1 = time.time()
    print(f' * zc_name: {args.zc_name}')
    for i in range(num_samples):
        bit_cfg = emqapi.fix_bit_cfg(i)
        acc = emqapi.query_by_cfg(bit_cfg)
        # initialize_quant_model(qnn)

        zc = predictive.find_measures(
            cnn,
            train_loader,
            dataload_info=dataload_info,
            measure_names=[args.zc_name],
            loss_fn=F.cross_entropy,
            device=torch.device('cuda:0'))
        print(f'Current zc {args.zc_name} is {zc}')
        gt_list.append(acc)
        zc_list.append(zc)

        print(f'kendalltau: {kendalltau(zc_list, gt_list)}\n'
              f'spearman:{spearman(zc_list, gt_list)}\n'
              f'pearson: {pearson(zc_list, gt_list)}\n')

    print(' * Time cost: ', time.time() - t1)
    print(' * Each time cost: ', (time.time() - t1) / num_samples)

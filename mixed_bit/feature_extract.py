import argparse
import math
import os
import random
import time

import numpy as np
import ORM
import pulp
import torch
import torch.nn as nn
from models import MobileNetV2, ResNet_ImageNet, TrainRunConfig
from pulp import *  # noqa: F403,F401
from run_manager import RunManager
from scipy import optimize
from utils.pytorch_utils import DFS_bit

parser = argparse.ArgumentParser()
""" model config """
parser.add_argument(
    '--path', type=str, default='/home/stack/data_sdc/data/hub')
parser.add_argument(
    '--model',
    type=str,
    default='resnet18',
    choices=['resnet50', 'mobilenetv2', 'mobilenet', 'resnet18'])
parser.add_argument('--cfg', type=str, default='None')
parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument('--model_size', default=0, type=float)
parser.add_argument('--beta', default=1, type=float)
parser.add_argument(
    '--quant_type', type=str, default='QAT', choices=['QAT', 'PTQ'])
""" dataset config """
parser.add_argument(
    '--dataset', type=str, default='imagenet', choices=['cifar10', 'imagenet'])
parser.add_argument('--save_path', type=str, default='./save_path')
""" runtime config """
parser.add_argument('--gpu', help='gpu available', default='0')
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--n_worker', type=int, default=4)
parser.add_argument('--local_rank', default=0, type=int)

if __name__ == '__main__':
    args = parser.parse_args()

    # cpu_num = 1
    # os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    # os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    # os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    # os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    # os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    # torch.set_num_threads(cpu_num)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.cuda.set_device(0)

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    # distributed setting
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()

    # prepare run config
    run_config_path = '%s/run.config' % args.path

    run_config = TrainRunConfig(**args.__dict__)
    if args.local_rank == 0:
        print('Run config:')
        for k, v in args.__dict__.items():
            print('\t%s: %s' % (k, v))

    if args.model == 'resnet50':
        assert args.dataset == 'imagenet', 'resnet50 only supports imagenet dataset'
        net = ResNet_ImageNet(
            depth=50,
            num_classes=run_config.data_provider.n_classes,
            cfg=eval(args.cfg))
    elif args.model == 'mobilenetv2':
        assert args.dataset == 'imagenet', 'mobilenetv2 only supports imagenet dataset'
        net = MobileNetV2(
            num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))
    elif args.model == 'resnet18':
        assert args.dataset == 'imagenet', 'resnet18 only supports imagenet dataset'
        net = ResNet_ImageNet(
            depth=18,
            num_classes=run_config.data_provider.n_classes,
            cfg=eval(args.cfg))

    # build run manager
    run_manager = RunManager(args.path, net, run_config)

    # load checkpoints
    best_model_path = '%s/checkpoints/model_best.pth.tar' % args.path
    assert os.path.isfile(best_model_path), f'wrong path is {best_model_path}'
    if torch.cuda.is_available():
        checkpoint = torch.load(
            best_model_path, map_location=torch.device(f'cuda:{args.gpu}'))
    else:
        checkpoint = torch.load(best_model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    # strip the module
    ckpt = {}
    for k, v in checkpoint.items():
        ckpt['module.' + k] = v
    # run_manager.net.load_state_dict(ckpt)
    output_dict = {}

    # feature extract
    # start = time.time()
    data_loader = run_manager.run_config.train_loader
    data = next(iter(data_loader))
    data = data[0]
    n = data.size()[0]

    with torch.no_grad():
        feature = net.feature_extract(data, args.quant_type)

    for i in range(len(feature)):
        feature[i] = feature[i].view(n, -1)
        feature[i] = feature[i].data.cpu().numpy()

    orthogonal_matrix = np.zeros((len(feature), len(feature)))

    for i in range(len(feature)):
        for j in range(len(feature)):
            with torch.no_grad():
                orthogonal_matrix[i][j] = ORM.orm(
                    ORM.gram_linear(feature[i]), ORM.gram_linear(feature[j]))

    def sum_list(a, j):
        b = 0
        for i in range(len(a)):
            if i != j:
                b += a[i]
        return b

    theta = []
    gamma = []
    flops = []

    for i in range(len(feature)):
        gamma.append(sum_list(orthogonal_matrix[i], i))

    # e^-x
    for i in range(len(feature)):
        theta.append(1 * math.exp(-1 * args.beta * gamma[i]))
    theta = np.array(theta)
    theta = np.negative(theta)

    length = len(feature)
    # layerwise
    params, first_last_size = net.cfg2params_perlayer(net.cfg, length,
                                                      args.quant_type)
    FLOPs, first_last_flops = net.cfg2flops_layerwise(net.cfg, length,
                                                      args.quant_type)
    params = [i / (1024 * 1024) for i in params]
    first_last_size = first_last_size / (1024 * 1024)

    # Objective function
    def func(x, sign=1.0):
        """ Objective function """
        global theta, length
        sum_fuc = []
        for i in range(length):
            temp = 0.
            for j in range(i, length):
                temp += theta[j]
            sum_fuc.append(x[i] * (sign * temp / (length - i)))

        return sum(sum_fuc)

    # Derivative function of objective function
    def func_deriv(x, sign=1.0):
        """ Derivative of objective function """
        global theta, length
        diff = []
        for i in range(length):
            temp1 = 0.
            for j in range(i, length):
                temp1 += theta[j]
            diff.append(sign * temp1 / (length - i))

        return np.array(diff)

    # Constraint function
    def constrain_func(x):
        """ constrain function """
        global params, length
        a = []
        for i in range(length):
            a.append(x[i] * params[i])
        return np.array([args.model_size - first_last_size - sum(a)])

    bnds = []  # bit search space: (0.25,0.5) for PTQ and (0.5,1.0) for QAT
    if args.quant_type == 'PTQ':
        for i in range(length):
            bnds.append((0.25, 0.5))
    else:
        for i in range(length):
            bnds.append((0.5, 1.0))

    bnds = tuple(bnds)
    cons = ({'type': 'ineq', 'fun': constrain_func})

    result = optimize.minimize(
        func,
        x0=[1 for i in range(length)],
        jac=func_deriv,
        method='SLSQP',
        bounds=bnds,
        constraints=cons)

    if args.model == 'resnet18':
        prun_bitcfg, _ = DFS_bit(
            result.x[::-1] * 8,
            [params[length - i - 1] for i in range(length)])
        prun_bitcfg = [prun_bitcfg[length - i - 1] for i in range(length)]
    else:
        prun_bitcfg = np.around(result.x * 8)

    # end = time.time()
    # print("Use", end - start, "seconds. ")

    optimize_cfg = []
    if type(prun_bitcfg[0]) != int:
        for i in range(len(prun_bitcfg)):
            b = list(prun_bitcfg)[i].tolist()
            optimize_cfg.append(int(b))
    else:
        optimize_cfg = prun_bitcfg

    # print(result.x)
    print(optimize_cfg)
    print(
        'Quantization model is',
        np.sum(np.array(optimize_cfg) * np.array(params) / 8) +
        first_last_size, 'Mb')
    print('Original model is',
          np.sum(np.array(params)) * 4 + first_last_size * 4, 'Mb')
    print('Quantization model BOPs is',
          (first_last_flops * 8 * 8 +
           sum([FLOPs[i] * optimize_cfg[i] * 5 for i in range(length)])) / 1e9)

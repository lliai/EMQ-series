import argparse
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

import emq.models.hubconf as hubconf
from emq.api import EMQAPI
from emq.dataset.imagenet import build_imagenet_data
from emq.predictor.pruners import predictive
from emq.utils.rank_consistency import kendalltau, pearson, spearman_top_k

emqapi = EMQAPI('./data/PTQ-GT.pkl', verbose=False)

if __name__ == '__main__':

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
        default=128,
        type=int,
        help='mini-batch size for data loader')
    parser.add_argument(
        '--workers',
        default=2,
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
        '--samples', default=50, type=int, help='number of iteration for LSQ')
    parser.add_argument(
        '--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument(
        '--p', default=2.4, type=float, help='L_p norm minimization for LSQ')
    parser.add_argument(
        '--zc_name', default='hawqv1', type=str, help='name of zero centering')

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    # build imagenet data loader
    train_loader, test_loader = build_imagenet_data(
        batch_size=args.batch_size,
        workers=args.workers,
        data_path=args.data_path)

    # load model
    cnn = eval(f'hubconf.{args.arch}(pretrained=False)')
    if torch.cuda.is_available():
        cnn.cuda()
    cnn.eval()

    gt_list = []
    zc_list = []

    assert args.zc_name in {
        'hawqv1', 'hawqv2', 'orm', 'bparams', 'qe_score', 'bsnip', 'bsynflow'
    }

    dataload_info = ['random', 1, 1000]

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # time the following code
    s = time.time()

    print(f'zc_name: {args.zc_name}')
    for i in tqdm(range(args.samples)):
        bit_cfg = emqapi.fix_bit_cfg(emqapi.random_index())
        acc = emqapi.query_by_cfg(bit_cfg)
        zc = predictive.find_measures(
            cnn,
            train_loader,
            dataload_info=dataload_info,
            measure_names=[args.zc_name],
            loss_fn=F.cross_entropy,
            device=device,
            bit_cfg=bit_cfg)
        print(f'The {i}-th zc {args.zc_name} is: {zc}.')
        gt_list.append(acc)
        zc_list.append(zc)

    k, sp_list = spearman_top_k(gt_list, zc_list, [1, 0.5, 0.2])
    kd, ps = kendalltau(gt_list, zc_list), pearson(gt_list, zc_list)

    print(f' * Time cost: {time.time() - s} s')
    print(' * Each time cost: ', (time.time() - s) / args.samples)
    print(
        f' * For {args.zc_name}: sp@top100%: {sp_list[0]}, sp@top50%: {sp_list[1]}, sp@top20%: {sp_list[2]}'
    )
    print(f' * For {args.zc_name}: kd: {kd}, ps: {ps}, sp: {sp_list[0]}')

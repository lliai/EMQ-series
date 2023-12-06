import argparse
import pickle
import random
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from emq.predictor.pruners.measures.bparams import get_bparams


def random_bit_cfg(arch: str, quant_type='PTQ'):
    assert arch in ['resnet18', 'mobilenetv2', 'resnet50']
    if arch == 'resnet18':
        if quant_type == 'PTQ':
            return tuple(random.choice(range(2, 5)) for _ in range(18))
        elif quant_type == 'QAT':
            # QAT
            return tuple(random.choice(range(4, 9)) for _ in range(18))
    elif arch == 'mobilenetv2':
        if quant_type == 'PTQ':
            return tuple(random.choice(range(2, 5)) for _ in range(53))
        elif quant_type == 'QAT':
            # QAT
            return tuple(random.choice(range(4, 9)) for _ in range(53))
    elif arch == 'resnet50':
        if quant_type == 'PTQ':
            return tuple(random.choice(range(2, 5)) for _ in range(53))
        elif quant_type == 'QAT':
            # QAT
            return tuple(random.choice(range(4, 9)) for _ in range(53))


def root_mutate_root(root_cfg: list, arch: str, quant_type='PTQ'):
    assert arch in {'resnet18', 'mobilenetv2', 'resnet50'}
    new_cfg = deepcopy(root_cfg)
    if arch == 'mobilenetv2':
        if quant_type == 'PTQ':
            p = np.array([0.1, 0.3, 0.6])
            assert len(root_cfg) == 53

            sample_num = random.sample(range(3), k=1)[0] + 1
            sample_idx = random.sample(range(53), k=sample_num)

            for idx in sample_idx:
                new_cfg[idx] = np.random.choice(range(2, 5), p=p.ravel())
            return tuple(new_cfg)

        elif quant_type == 'QAT':
            # QAT
            assert len(root_cfg) == 53
            p = np.array([0.1, 0.1, 0.2, 0.2, 0.4])

            sample_num = random.sample(range(6), k=1)[0] + 1
            sample_idx = random.sample(range(53), k=sample_num)

            for idx in sample_idx:
                new_cfg[idx] = np.random.choice(range(4, 9), p=p.ravel())
            return tuple(new_cfg)

    elif arch == 'resnet18':
        if quant_type == 'PTQ':
            p = np.array([0.1, 0.3, 0.6])
            assert len(root_cfg) == 18
            sample_num = random.sample(range(3), k=1)[0] + 1
            sample_idx = random.sample(range(18), k=sample_num)

            for idx in sample_idx:
                new_cfg[idx] = np.random.choice(range(2, 5), p=p.ravel())
            return tuple(new_cfg)

        elif quant_type == 'QAT':
            # QAT
            assert len(root_cfg) == 18
            p = np.array([0.1, 0.1, 0.2, 0.2, 0.4])

            sample_num = random.sample(range(6), k=1)[0] + 1
            sample_idx = random.sample(range(18), k=sample_num)

            for idx in sample_idx:
                new_cfg[idx] = np.random.choice(range(4, 9), p=p.ravel())
            return tuple(new_cfg)

    elif arch == 'resnet50':
        if quant_type == 'PTQ':
            p = np.array([0.1, 0.3, 0.6])
            assert len(root_cfg) == 53

            sample_num = random.sample(range(3), k=1)[0] + 1
            sample_idx = random.sample(range(53), k=sample_num)

            for idx in sample_idx:
                new_cfg[idx] = np.random.choice(range(2, 5), p=p.ravel())
            return tuple(new_cfg)

        elif quant_type == 'QAT':
            # QAT
            p = np.array([0.1, 0.1, 0.2, 0.2, 0.4])
            assert len(root_cfg) == 53

            sample_num = random.sample(range(6), k=1)[0] + 1
            sample_idx = random.sample(range(53), k=sample_num)

            for idx in sample_idx:
                new_cfg[idx] = np.random.choice(range(4, 9), p=p.ravel())
            return tuple(new_cfg)


def generate_files_for_mobilenetv2(model_size_list, quant_type, arch):
    for model_size in model_size_list:
        with open(
                f'./output/candidate_{quant_type}_mobilenetv2_{str(model_size)}.txt',
                'w') as f:
            if model_size == 1.5:
                root_cfg = [
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3
                ]
            elif model_size == 1.3:
                root_cfg = [
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 2
                ]
            elif model_size == 1.1:
                root_cfg = [
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 2, 2, 2
                ]
            elif model_size == 0.9:
                root_cfg = [
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 3, 4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 3, 2, 4, 2, 2, 4, 2, 2,
                    4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2
                ]

            iterations = 50000
            for _ in tqdm(range(iterations)):
                new_cfg = root_mutate_root(
                    root_cfg, arch=arch, quant_type=quant_type)
                cur_bparams = get_bparams(
                    net=None,
                    inputs=None,
                    targets=None,
                    loss_fn=None,
                    bit_cfg=new_cfg,
                    arch=arch)
                if cur_bparams < model_size and abs(cur_bparams -
                                                    model_size) < 0.5:
                    # write the new cfg to file
                    f.write(str(new_cfg))
                    f.write(':')
                    f.write(str(cur_bparams))
                    f.write('\n')


def generate_files_for_resnet18(model_size_list, quant_type, arch):
    for model_size in model_size_list:
        with open(
                f'./output/candidate_{quant_type}_{arch}_{str(model_size)}.txt',
                'w') as f:
            if model_size == 4.5:  # for 4.5 res-18
                root_cfg = [
                    4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 3, 3, 3, 3
                ]
            elif model_size == 5:
                root_cfg = [
                    4, 3, 3, 3, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 3, 4, 4, 4
                ]
            elif model_size == 5.5:
                root_cfg = [
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4
                ]
            elif model_size == 3:
                root_cfg = [
                    4, 3, 4, 4, 4, 4, 4, 3, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2
                ]
            elif model_size == 3.5:
                root_cfg = [
                    4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 3, 2, 2, 2, 3
                ]
            elif model_size == 4:
                root_cfg = [
                    4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4, 3, 2, 3, 3, 3
                ]
            elif model_size == 6.7:
                root_cfg = [  # for QAT
                    8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 4, 4, 6, 8, 4, 4, 4, 4
                ]

            iterations = 50000
            for _ in tqdm(range(iterations)):
                new_cfg = root_mutate_root(
                    root_cfg, arch=arch, quant_type=quant_type)
                cur_bparams = get_bparams(
                    net=None,
                    inputs=None,
                    targets=None,
                    loss_fn=None,
                    bit_cfg=new_cfg,
                    arch=arch)
                if cur_bparams < model_size and abs(cur_bparams -
                                                    model_size) < 2:
                    # write the new cfg to file
                    f.write(str(new_cfg))
                    f.write(':')
                    f.write(str(cur_bparams))
                    f.write('\n')


def generate_files_for_resnet50(model_size_list, quant_type, arch):
    # 16, 18.7
    for model_size in model_size_list:
        with open(
                f'./output/candidate_{quant_type}_{arch}_{str(model_size)}.txt',
                'w') as f:
            if model_size == 16:  # for 4.5 res-18
                root_cfg = [
                    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                    8, 8, 8, 8, 8, 4, 4, 4, 8, 4, 8, 8, 4, 8, 8, 5, 8, 8, 8, 8,
                    8, 4, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8
                ]
            elif model_size == 18:
                root_cfg = [
                    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                    8, 8, 8, 8, 8, 8, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                    8, 8, 8, 8, 4, 4, 4, 5, 4, 8, 8, 4, 8
                ]
            elif model_size == 21:
                root_cfg = [
                    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                    8, 8, 8, 8, 4, 4, 4, 8, 6, 8, 8, 8, 8
                ]
            else:
                print(f'not supported model size {model_size} for resnet50')

            iterations = 10000
            for _ in tqdm(range(iterations)):
                new_cfg = root_mutate_root(
                    root_cfg, arch=arch, quant_type=quant_type)
                newbie_cfg = random_bit_cfg(arch=arch, quant_type=quant_type)

                cur_bparams = get_bparams(
                    net=None,
                    inputs=None,
                    targets=None,
                    loss_fn=None,
                    bit_cfg=new_cfg,
                    arch=arch)
                new_bparam = get_bparams(
                    net=None,
                    inputs=None,
                    targets=None,
                    loss_fn=None,
                    bit_cfg=newbie_cfg,
                    arch=arch)

                if cur_bparams < model_size and abs(cur_bparams -
                                                    model_size) < 2:
                    # write the new cfg to file
                    f.write(str(new_cfg))
                    f.write(':')
                    f.write(str(cur_bparams))
                    f.write('\n')
                    print('new_cfg', new_cfg, cur_bparams)

                if new_bparam < model_size and abs(new_bparam -
                                                   model_size) < 2:
                    # write the new cfg to file
                    f.write(str(newbie_cfg))
                    f.write(':')
                    f.write(str(new_bparam))
                    f.write('\n')
                    print('newbie_cfg', newbie_cfg, new_bparam)


def deweight_files(model_size_list, quant_type, arch):
    for model_size in model_size_list:
        file_path = f'./output/candidate_{quant_type}_{arch}_{str(model_size)}.txt'
        out_file_path = f'./output/candidate_{quant_type}_{arch}_{str(model_size)}-deweight.txt'
        deweight_candidate_file(file_path, out_file_path)


def deweight_candidate_file(file_path, out_file_path):
    # deweight by new_cfg and write to file.
    with open(file_path, 'r') as f:
        lines = f.readlines()
        bitcfg2param_dict = {}
        for line in lines:
            k, v = line.split(':')
            if k in bitcfg2param_dict:
                continue
            bitcfg2param_dict[k] = v.strip()

        with open(out_file_path, 'w') as f:
            for k, v in bitcfg2param_dict.items():
                f.write(k)
                f.write(':')
                f.write(v)
                f.write('\n')


def convert_files_into_pickle(model_size_list, quant_type, arch):
    save_path = f'./output/candidate_{quant_type}_{arch}.pkl'
    save_dict = {}
    for model_size in model_size_list:
        file_path = f'./output/candidate_{quant_type}_{arch}_{str(model_size)}-deweight.txt'
        # convert file to binary files
        save_dict[model_size] = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                k, v = line.split(':')
                if k in save_dict[model_size]:
                    continue
                save_dict[model_size][k] = v.strip()
    pickle.dump(save_dict, open(save_path, 'wb'))


if __name__ == '__main__':
    # argparse add '--arch' and '--quant_type' and '--model_size'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch',
        '-a',
        metavar='ARCH',
        default='resnet50',
        help='model architecture: ' + ' (default: resnet18)')
    parser.add_argument(
        '--quant_type',
        '-qt',
        metavar='QUANT_TYPE',
        default='PTQ',
        help='quantization type: ' + ' (default: PTQ)')
    parser.add_argument(
        '--model_size',
        '-ms',
        metavar='MODEL_SIZE',
        default=6.7,
        type=float,
        help='model size: ' + ' (default: 6.7)')
    args = parser.parse_args()

    # PTQ
    # res18_model_size_list = [3, 3.5, 4, 4.5, 5, 5.5]
    # res18_arch = 'resnet18'
    # res50_arch = 'resnet50'

    # mbv2_model_size_list = [1.5, 1.3, 1.1, 0.9]
    # mbv2_arch = 'mobilenetv2'
    # quant_type = 'PTQ'

    # generate files from root cfg
    # generate_files_for_resnet18(res18_model_size_list, quant_type, res18_arch)
    # generate_files_for_mobilenetv2(mbv2_model_size_list, quant_type, mbv2_arch)

    # QAT
    # res18_model_size_list = [6.7]  # for qat
    # res50_model_size_list = [16, 18, 21]

    # if isinstance(args.model_size, float):
    #     model_size = [args.model_size]

    # if args.arch == 'resnet18':
    #     generate_files_for_resnet18(
    #         model_size, quant_type='QAT', arch='resnet18')  # 6.7
    # elif args.arch == 'resnet50':
    #     generate_files_for_resnet50(
    #         model_size, quant_type='QAT', arch='resnet50')  # 16
    # elif args.arch == 'mobilenetv2':
    #     generate_files_for_mobilenetv2(
    #         model_size, quant_type='QAT', arch='mobilenetv2')  # 1.5

    # deweight generated files
    res50_model_size_list = [16, 18, 21]
    # deweight_files(res50_model_size_list, quant_type='QAT', arch=args.arch)

    # convert files into pickle
    convert_files_into_pickle(
        res50_model_size_list, quant_type='QAT', arch=args.arch)

    # candidate_PTQ_resnet18.pkl
    # {
    #     3.5: {
    #         '[1, 3, 4, 2, 3]': 3.49,
    #         '[2, 3, 4, 1, 3]': 3.48,
    #     }
    # }

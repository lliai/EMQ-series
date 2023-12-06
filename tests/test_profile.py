# FOR DEBUG
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

import emq.models.hubconf as hubconf
from emq.dataset.imagenet import build_imagenet_data
from emq.quant import QuantModel
from emq.structures import TreeStructure
from emq.utils.rank_consistency import spearman_top_k as spearman
# from emq.utils.rank_consistency import spearman
from exps.search.evo_search_emq_zc import all_same, emqapi, is_anomaly


def fitness_spearman(dataiter, qnn, structure, num_sample=50):
    """structure is belong to popultion."""
    if structure.sp_score != -1:
        return structure.sp_score

    gt_score = []
    zc_score = []

    img, label = next(dataiter)

    for i in range(num_sample):
        bit_cfg = emqapi.fix_bit_cfg(i)
        gt = emqapi.query_by_cfg(bit_cfg)
        zc = structure(img, label, qnn, bit_cfg)

        if is_anomaly(zc):
            return -1

        # early exit
        if len(zc_score) > 3 and all_same(zc_score):
            return -1

        zc_score.append(zc)
        gt_score.append(gt)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # gc.collect()

    # TODO add inf check
    if len(zc_score) <= 1 or np.isnan(spearman(gt_score, zc_score)):
        return -1

    # release memory
    del img, label
    torch.cuda.empty_cache()
    # gc.collect()

    sp = spearman(gt_score, zc_score)
    if structure.sp_score == -1:
        structure.sp_score = sp
    return sp


def create_qnn(name='resnet18'):
    cnn = eval(f'hubconf.{name}(pretrained=False)')
    if torch.cuda.is_available():
        cnn.cuda()
    cnn.eval()

    # build quantization parameters
    wq_params = {'n_bits': 2, 'channel_wise': False, 'scale_method': 'mse'}
    aq_params = {
        'n_bits': 8,
        'channel_wise': False,
        'scale_method': 'mse',
        'leaf_param': False
    }
    qnn = QuantModel(
        model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    if torch.cuda.is_available():
        qnn.cuda()
    qnn.eval()

    # print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()

    qnn.set_quant_state(weight_quant=True, act_quant=False)
    return qnn


def test_fitness_spearman(dataiter, name, structure):
    # device = torch.device('cuda:1')
    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("my_function"):

    for i in range(1000):
        if i % 10 == 0:
            structure = TreeStructure()
        qnn = create_qnn(name)
        last_mem = torch.cuda.memory_allocated(0) / 1024 / 1024
        zc = fitness_spearman(dataiter, qnn, structure)
        if zc == -1:
            structure = TreeStructure()

        # print('zc: ', zc)
        del qnn
        torch.cuda.empty_cache()
        # print('GPU Memory Allocated {} MB'.format(
        #     torch.cuda.memory_allocated(device=device)/1024./1024.))
        if i % 20 == 0:
            print('added memory usage: ',
                  torch.cuda.memory_allocated(0) / 1024 / 1024 - last_mem)


if __name__ == '__main__':

    train_loader, test_loader = build_imagenet_data(
        batch_size=64,
        workers=1,
        data_path='/home/stack/data_sdb/all_imagenet_data/',
        proxy=True)
    # to save memory
    dataiter = iter(train_loader)
    structure = TreeStructure()
    name = 'resnet18'
    test_fitness_spearman(dataiter, name, structure)

# https://github.com/alibaba/lightweight-neural-architecture-search/blob/main/nas/scores/compute_madnas.py

# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

from . import measure


@measure('bit_sum', bn=True)
def compute_bit_sum(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
    bit_cfg=None,
):
    assert bit_cfg is not None

    return sum(bit_cfg)

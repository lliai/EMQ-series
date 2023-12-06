# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from .blocks_basic import ConvKXBN, ConvKXBNRELU
from .super_quant_res_k1dwk1 import SuperQuantResK1DWK1
from .super_res_k1dwk1 import ResK1DWK1, SuperResK1DWK1

__all_blocks__ = {
    'ResK1DWK1': ResK1DWK1,
    'SuperResK1DWK1': SuperResK1DWK1,
    'SuperQuantResK1DWK1': SuperQuantResK1DWK1,
    'ConvKXBN': ConvKXBN,
    'ConvKXBNRELU': ConvKXBNRELU,
}

##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# The macro structure is defined in NAS-Bench-201
from .genotypes import Structure as CellStructure
from .genotypes import architectures as CellArchitectures
from .search_model_darts import TinyNetworkDarts
from .search_model_darts_nasnet import NASNetworkDARTS
from .search_model_enas import TinyNetworkENAS
from .search_model_gdas import TinyNetworkGDAS
# NASNet-based macro structure
from .search_model_gdas_nasnet import NASNetworkGDAS
from .search_model_random import TinyNetworkRANDOM
from .search_model_setn import TinyNetworkSETN

nas201_super_nets = {
    'DARTS-V1': TinyNetworkDarts,
    'DARTS-V2': TinyNetworkDarts,
    'GDAS': TinyNetworkGDAS,
    'SETN': TinyNetworkSETN,
    'ENAS': TinyNetworkENAS,
    'RANDOM': TinyNetworkRANDOM,
}

nasnet_super_nets = {'GDAS': NASNetworkGDAS, 'DARTS': NASNetworkDARTS}

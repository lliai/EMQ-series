import os
import pickle
import random
from typing import Dict, List, Tuple, Union


def convert_str2tuple(str):
    str = str.lstrip('[').rstrip(']')
    str = str.lstrip('(').rstrip(')')
    items = str.split(',')
    return tuple([int(item) for item in items])


class EMQAPI:
    """EMQBenchmark API for querying the bit_cfg."""

    def __init__(self, path: str, verbose=True):
        assert os.path.exists(path)
        self.verbose = verbose

        self.bit_dict: Dict[Tuple, Dict] = dict()
        self.cfg2idx: Dict[Tuple, int] = dict()

        # load with pickle
        with open(path, 'rb') as f:
            self.bit_dict = pickle.load(f)

        # extract idx2bit_acc
        for idx, bit_acc in self.bit_dict.items():
            cfg = bit_acc['bit_cfg']
            cfg = convert_str2tuple(cfg)
            self.cfg2idx[cfg] = idx
            self.bit_dict[idx]['bit_cfg'] = cfg

    def random_index(self) -> int:
        assert len(self.bit_dict) > 0
        return random.choice(range(len(self.bit_dict)))

    def random_bit_cfg(self) -> tuple:
        assert len(self.bit_dict) > 0
        rnd_idx = self.random_index()
        return self.bit_dict[rnd_idx]['bit_cfg']

    def fix_bit_cfg(self, idx) -> tuple:
        assert idx < len(self.bit_dict)
        return self.bit_dict[idx]['bit_cfg']

    def query_by_idx(self, idx: int) -> float:
        """Query acc by index."""
        assert idx < len(self.bit_dict), \
            f'Out of range, max range {len(self.bit_dict)}'
        if self.verbose:
            print(
                f'Querying the {idx}-th arch: {self.bit_dict[idx]["bit_cfg"]} with acc: {self.bit_dict[idx]["acc"]}'
            )
        return self.bit_dict[idx]['acc']

    def get_idx_by_cfg(self, bit_cfg: Tuple) -> int:
        """Get corresponding index by cfg."""
        if bit_cfg in self.cfg2idx:
            return self.cfg2idx[bit_cfg]
        else:
            raise KeyError(f'Not supported Key: {bit_cfg}')

    def preprocess_cfg(self, bit_cfg: Union[Tuple, Union[List, str]]) -> Tuple:
        """preprocess all cfgs to Tuple"""
        if isinstance(bit_cfg, list):
            return tuple(bit_cfg)
        elif isinstance(bit_cfg, tuple):
            return bit_cfg
        elif isinstance(bit_cfg, str):
            return convert_str2tuple(bit_cfg)
        else:
            raise TypeError('Not Supported Type')

    def query_by_cfg(self, bit_cfg: Union[Tuple, Union[List, str]]) -> float:
        bit_cfg = self.preprocess_cfg(bit_cfg)
        idx = self.get_idx_by_cfg(bit_cfg)
        return self.query_by_idx(idx)

    def __iter__(self):
        return iter(self.bit_dict.items())

    def __len__(self):
        return len(self.bit_dict)

    def __next__(self):
        return next(self.bit_dict.items())

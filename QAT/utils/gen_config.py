import random
from typing import List

import yaml

with open('./QAT/bit_template.yaml', 'r') as f:
    bit_template = yaml.load(f, Loader=yaml.FullLoader)


def convert_by_bit_cfg(bit_key: str, bit_cfg: List):
    # filter the _placeholder_ and convert by bit_cfg
    cnt = 0
    new_dict = {}
    for k, v in bit_template[bit_key].items():
        if v == '_placeholder_':
            new_dict[k] = bit_cfg[cnt]
            cnt += 1
        else:
            new_dict[k] = v
    return new_dict


if __name__ == '__main__':
    # bit_cfg = [4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 3, 3, 3, 3]
    # res = convert_by_bit_cfg('bit_config_resnet18', bit_cfg)

    bit_cfg = [random.choice(range(3, 5)) for _ in range(53)]
    res = convert_by_bit_cfg('bit_config_resnet50', bit_cfg)

    print(res)

path = './scripts/bench/extracted.pkl'

# read file from path and split it by ':'.
# add index to make it convenient.
with open(path, 'r') as f:
    lines = f.readlines()
    idx2bit_acc_dict = {}
    for idx, line in enumerate(lines):
        k, v = line.split(':')
        idx2bit_acc_dict[idx] = {'bit_cfg': k, 'acc': float(v.strip())}

import pickle

pickle.dump(idx2bit_acc_dict, open('./data/PTQ-GT.pkl', 'wb'))

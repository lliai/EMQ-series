import numpy as np


def read_entropy_file(filename):
    """
    Read an input file, which includes target data information
    - information: (index, entropy, label)
    """
    index = []
    entropy = []
    label = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            tokens = line.split()
            index.append(int(tokens[0]))
            entropy.append(float(tokens[1]))
            label.append(int(tokens[2]))

    entropy = np.array(entropy)
    return index, entropy, label


def get_proxy_data_random(entropy, sampling_portion, logging=None):
    """
    Random selection
    """
    num_proxy_data = int(np.floor(sampling_portion * len(entropy)))
    indices = np.random.choice(
        range(len(entropy)), size=num_proxy_data, replace=False)

    np.random.shuffle(indices)
    return indices


def get_proxy_data_log_entropy_histogram(entropy,
                                         sampling_portion,
                                         sampling_type=1,
                                         dataset='cifar10',
                                         logging=None):
    """
    Proposed probabilistic selection
    - uses data entropy in log scale
    - sampling_type is one of {1, 2, 3}; P1, P2, and P3 in paper
    """
    # make histogram and collect data index in each bin
    log_entropy = np.log10(entropy)
    min_log_entropy, max_log_entropy = np.min(log_entropy), np.max(log_entropy)
    if dataset == 'cifar10':
        bin_width = 0.5
    elif dataset == 'cifar100':
        bin_width = 0.25
    elif dataset == 'svhn':
        bin_width = 0.2
    elif dataset == 'imagenet':
        bin_width = 0.25

    low_bin = np.round(min_log_entropy)
    while min_log_entropy < low_bin:
        low_bin -= bin_width
    high_bin = np.round(max_log_entropy)
    while max_log_entropy > high_bin:
        high_bin += bin_width
    bins = np.arange(low_bin, high_bin + bin_width, bin_width)

    def get_bin_idx(ent):
        for i in range(len(bins) - 1):
            if (bins[i] < ent) and (ent < bins[i + 1]):
                return i
        return None

    index_histogram = []
    for i in range(len(bins) - 1):
        index_histogram.append([])

    for index, e in enumerate(log_entropy):
        bin_idx = get_bin_idx(e)
        if bin_idx is None:
            raise ValueError(
                '[Error] histogram bin settings is wrong ... histogram bins: [%f ~ %f], current: %f'
                % (low_bin, high_bin, e))
        index_histogram[bin_idx].append(index)

    # prepare the histogram selection probability
    histo = np.array([len(l) for l in index_histogram])
    if sampling_type == 1:
        # P1
        inv_histo = (max(histo) - histo + 1) * (histo != 0)
        inv_histo_prob = inv_histo / np.sum(inv_histo)
    elif sampling_type == 2:
        # P2
        inv_histo_prob = np.array(
            [1 / (len(bins) - 1) for _ in index_histogram])
    elif sampling_type == 3:
        # P3
        inv_histo_prob = np.array([(1 / len(l) if len(l) != 0 else 0)
                                   for l in index_histogram])
    else:
        raise ValueError('Error in sampling type for histogram-based sampling')

    if logging is not None:
        logging.info(inv_histo_prob)

    # get indices from all buckets
    if dataset == 'imagenet':
        num_proxy_data = int(
            int(np.floor(sampling_portion * len(entropy))) / 1000) * 1000
    else:
        num_proxy_data = int(np.floor(sampling_portion * len(entropy)))
    indices = []
    total_indices = []
    total_prob = []
    for index_bin, prob in zip(index_histogram, inv_histo_prob):
        if len(index_histogram) == 0:
            continue
        total_indices += index_bin
        temp = np.array([prob for _ in range(len(index_bin))])
        temp = temp / len(index_bin)
        total_prob += temp.tolist()
    total_prob = total_prob / np.sum(total_prob)
    indices = np.random.choice(
        total_indices, size=num_proxy_data, replace=False, p=total_prob)

    selected_index_num = [0] * len(histo)
    for i in indices:
        selected_index_num[get_bin_idx(log_entropy[i])] += 1
    if logging is not None:
        for i, a, b in zip(
                range(len(histo)), selected_index_num, histo.tolist()):
            if b == 0:
                logging.info('{} selected from 0 (0%%) in bin {}'.format(a, i))
            else:
                logging.info('{} selected from {} ({:.2f}%%) in bin {}'.format(
                    a, b, a / b * 100, i))

    np.random.shuffle(indices)
    return indices

import concurrent.futures

import numpy as np


class obj:

    def __init__(self, x):
        self.x = x


def compute_sum(nums, info='test'):
    total = 0
    for num in nums:
        total += num
    print(info)
    return obj(total)


def compute_sum_parallel(nums, num_threads):
    chunk_size = len(nums) // num_threads
    chunks = [nums[i:i + chunk_size] for i in range(0, len(nums), chunk_size)]

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads) as executor:
        # results = list(executor.map(compute_sum, chunks))
        results = [
            executor.submit(compute_sum, chunk, f'thread {i}')
            for i, chunk in enumerate(chunks)
        ]

    r = []
    for tmp in results:
        r.append(tmp.x)
    return sum(r)


if __name__ == '__main__':
    nums = np.arange(0, 100, 1).tolist()
    total = compute_sum(nums)
    print(f'Sum of {nums} is {total.x}')

    num_threads = 4
    total_parallel = compute_sum_parallel(nums, num_threads)
    print(f'Sum of {nums} using {num_threads} threads is {total_parallel}')

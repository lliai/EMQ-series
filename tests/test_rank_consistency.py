import numpy as np

from emq.utils.rank_consistency import spearman_top_k

a = np.random.randn(50)
b = np.random.randn(50)

patks = spearman_top_k(a, b)

print(patks)

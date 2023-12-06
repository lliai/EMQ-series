import os
import sys
import unittest
from unittest import TestCase

root_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(root_path), '..'))

from emq.api import EMQAPI


class TestAPI(TestCase):

    def setUp(self):
        self.api = EMQAPI('./data/PTQ-GT.pkl')

    def test_query_by_idx(self):
        for i in range(10):
            self.api.query_by_idx(i)

    def test_query_by_cfg(self):
        # tuple
        cfg1 = (4, 4, 2, 2, 4, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 4, 3)
        # list
        cfg2 = [4, 4, 2, 2, 4, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 4, 3]
        # str
        cfg3 = '[4, 4, 2, 2, 4, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 4, 3]'
        cfg4 = '(4, 4, 2, 2, 4, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 4, 3)'

        cfg_list = [cfg1, cfg2, cfg3, cfg4]
        for cfg in cfg_list:
            self.api.query_by_cfg(cfg)


if __name__ == '__main__':
    unittest.main()

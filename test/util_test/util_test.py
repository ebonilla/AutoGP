import unittest

import numpy as np
import tensorflow as tf

from autogp import util


class TestInitList(unittest.TestCase):
    def test_empty(self):
        self.assertEquals(util.init_list(0.0, [0]), [])

    def test_single_element(self):
        self.assertEquals(util.init_list(1.0, [1]), [1.0])

    def test_nested_single(self):
        self.assertEquals(util.init_list(1.0, [1, 1, 1, 1]), [[[[1.0]]]])

    def test_single_level(self):
        self.assertEquals(util.init_list(2.0, [4]), [2.0, 2.0, 2.0, 2.0])

    def test_multiple_levels(self):
        self.assertEquals(util.init_list(3.25, [3, 2, 1]),
                          [[[3.25], [3.25]],
                           [[3.25], [3.25]],
                           [[3.25], [3.25]]])


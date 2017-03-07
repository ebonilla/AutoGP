import unittest

import numpy as np
import tensorflow as tf

from autogp import util


SIG_FIGS = 5


class TestCholNormal(unittest.TestCase):
    def chol_normal_log_prob(self, val, mean, covar):
        chol_normal = util.CholNormal(np.array(mean, dtype=np.float32),
                                      np.array(covar, dtype=np.float32))
        return tf.Session().run(chol_normal.log_prob(np.array(val, dtype=np.float32)))

    def test_same_mean(self):
        log_prob = self.chol_normal_log_prob([1.0], [1.0], [[1.0]])
        self.assertAlmostEqual(log_prob, -0.5 * np.log(2 * np.pi), SIG_FIGS)

    def test_scalar_covar(self):
        log_prob = self.chol_normal_log_prob([1.0], [1.0], [[np.sqrt(2.0)]])
        self.assertAlmostEqual(log_prob, -0.5 * (np.log(2 * np.pi) + np.log(2.0)), SIG_FIGS)

    def test_small_scalar_covar(self):
        log_prob = self.chol_normal_log_prob([1.0], [1.0], [[1e-10]])
        self.assertAlmostEqual(log_prob, -0.5 * (np.log(2 * np.pi) + np.log(1e-20)), SIG_FIGS)

    def test_large_scalar_covar(self):
        log_prob = self.chol_normal_log_prob([1.0], [1.0], [[1e10]])
        self.assertAlmostEqual(log_prob, -0.5 * (np.log(2 * np.pi) + np.log(1e20)), SIG_FIGS)

    def test_multi_covar_same_mean(self):
        log_prob = self.chol_normal_log_prob([1.0, 2.0], [1.0, 2.0], [[1.0, 0.0], [2.0, 3.0]])
        self.assertAlmostEqual(log_prob, -0.5 * (2.0 * np.log(2 * np.pi) + np.log(9.0)), SIG_FIGS)


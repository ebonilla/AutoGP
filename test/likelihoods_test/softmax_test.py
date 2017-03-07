import unittest

import numpy as np
import tensorflow as tf

from autogp import util
from autogp import likelihoods


SIG_FIGS = 5


class TestSoftmax(unittest.TestCase):
    def log_prob(self, outputs, latent):
        softmax = likelihoods.Softmax()
        return tf.Session().run(softmax.log_cond_prob(np.array(outputs, dtype=np.float32),
                                                      np.array(latent, dtype=np.float32)))

    def predict(self, latent_means, latent_vars):
        softmax = likelihoods.Softmax()
        return tf.Session().run(softmax.predict(np.array(latent_means, dtype=np.float32),
                                                np.array(latent_vars, dtype=np.float32)))

    def test_single_prob(self):
        log_prob = self.log_prob([[1.0, 0.0]], [[[5.0, 2.0]]])
        self.assertAlmostEqual(np.exp(log_prob), np.exp(5.0) / (np.exp(5.0) + np.exp(2.0)),
                               SIG_FIGS)

    def test_extreme_probs(self):
        log_prob = self.log_prob([[1.0, 0.0],
                                  [0.0, 1.0]],
                                 [[[1e10, -1e10],
                                   [-1e10, 1e10]],
                                  [[-1e10, 1e10],
                                   [1e10, -1e10]]])
        true_probs = np.array([[1.0, 1.0],
                               [0.0, 0.0]])
        np.testing.assert_almost_equal(np.exp(log_prob), true_probs, SIG_FIGS)

    def test_multi_probs(self):
        log_prob = self.log_prob([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]],
                                 [[[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0],
                                   [7.0, 8.0, 9.0]],
                                  [[10.0, 11.0, 12.0],
                                   [13.0, 14.0, 15.0],
                                   [16.0, 17.0, 18.0]]])
        true_probs = np.array([[np.exp(1.0) / (np.exp(1.0) + np.exp(2.0) + np.exp(3.0)),
                                np.exp(5.0) / (np.exp(4.0) + np.exp(5.0) + np.exp(6.0)),
                                np.exp(9.0) / (np.exp(7.0) + np.exp(8.0) + np.exp(9.0))],
                               [np.exp(10.0) / (np.exp(10.0) + np.exp(11.0) + np.exp(12.0)),
                                np.exp(14.0) / (np.exp(13.0) + np.exp(14.0) + np.exp(15.0)),
                                np.exp(18.0) / (np.exp(16.0) + np.exp(17.0) + np.exp(18.0))]])
        np.testing.assert_almost_equal(np.exp(log_prob), true_probs, SIG_FIGS)


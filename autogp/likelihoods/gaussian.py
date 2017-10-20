from __future__ import absolute_import
import numpy as np
import tensorflow as tf

from . import likelihood


class Gaussian(likelihood.Likelihood):
    def __init__(self, std_dev=1.0):
        # Save the raw standard deviation. Note that this value can be negative.
        self.raw_std_dev = tf.Variable(std_dev)

    def log_cond_prob(self, outputs, latent):
        var = self.raw_std_dev ** 2
        return -0.5 * tf.log(2.0 * np.pi * var) - ((outputs - latent) ** 2) / (2.0 * var)

    def get_params(self):
        return [self.raw_std_dev]

    def predict(self, latent_means, latent_vars):
        return latent_means, latent_vars + self.raw_std_dev ** 2


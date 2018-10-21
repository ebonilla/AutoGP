from __future__ import absolute_import
import numpy as np
import tensorflow as tf

from . import likelihood


class RegressionNetwork(likelihood.Likelihood):
    def __init__(self, output_dim, std_dev, num_samples=5000):
        self.output_dim = output_dim
        self.num_samples = num_samples
        self.log_std_dev = tf.Variable(
            np.ones([self.output_dim]) * np.log(std_dev),
            dtype=tf.float32
        )

    def log_cond_prob(self, outputs, latent):
        weights = latent[:, :, :self.output_dim]
        inputs = latent[:, :, self.output_dim:]
        prod = weights * inputs
        # diff = outputs - prod
        covar = tf.exp(self.log_std_dev)
        quad_form = tf.reduce_sum(1.0 / covar * (outputs - prod) ** 2, 2)
        return -0.5 * (
            self.output_dim * tf.log(2.0 * np.pi) +
            tf.reduce_sum(covar) + quad_form)

    def get_params(self):
        return [self.log_std_dev]

    def predict(self, latent_means, latent_vars):
        # Generate samples to estimate the expected value
        # and variance of outputs.
        num_points = tf.shape(latent_means)[0]
        output_dims = tf.shape(latent_means)[1]
        latent = (
            latent_means + tf.sqrt(latent_vars) *
            tf.random_normal([self.num_samples, num_points, output_dims]))
        weights = latent[:, :, :output_dims - 1]
        inputs = latent[:, :, output_dims - 1:]
        prod = weights * inputs
        return tf.reduce_mean(prod, 0), tf.reduce_mean(prod, 0)

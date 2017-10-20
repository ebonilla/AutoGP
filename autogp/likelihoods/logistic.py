from __future__ import absolute_import
import tensorflow as tf
from . import likelihood


class Logistic(likelihood.Likelihood):
    def __init__(self, num_samples=2000):
        self.num_samples = num_samples

    def log_cond_prob(self, outputs, latent):
        return latent * (outputs - 1) - tf.log(1 + tf.exp(-latent))

    def get_params(self):
        return []

    def predict(self, latent_means, latent_vars):
        # Generate samples to estimate the expected value
        # and variance of outputs.
        num_points = tf.shape(latent_means)[0]
        latent = (
            latent_means + tf.sqrt(latent_vars) *
            tf.random_normal([self.num_samples, num_points, 1])
        )
        # Compute the softmax of all generated latent values
        # in a stable fashion.
        logistic = 1.0 / (1.0 + tf.exp(-latent))

        # Estimate the expected value of the softmax
        # and the variance through sampling.
        pred_means = tf.reduce_mean(logistic, 0)
        pred_vars = tf.reduce_sum(
            (logistic - pred_means) ** 2, 0
        ) / (self.num_samples - 1.0)

        return pred_means, pred_vars

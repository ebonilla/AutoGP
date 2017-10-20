from __future__ import absolute_import
import numpy as np
import tensorflow as tf

from . import util


class Normal(object):
    def __init__(self, mean, covar):
        self.mean = mean
        self.covar = covar


class CholNormal(Normal):
    def prob(self, val):
        return tf.exp(self.log_prob(val))

    def log_prob(self, val):
        dim = tf.to_float(tf.shape(self.mean)[0])
        diff = tf.expand_dims(val - self.mean, 1)
        quad_form = tf.reduce_sum(diff * tf.cholesky_solve(self.covar, diff))
        return -0.5 * (dim * tf.log(2.0 * np.pi) + util.log_cholesky_det(self.covar) +
                       quad_form)


class DiagNormal(Normal):
    def prob(self, val):
        return tf.exp(self.log_prob(val))

    def log_prob(self, val):
        dim = tf.to_float(tf.shape(self.mean)[0])
        quad_form = tf.reduce_sum(self.covar * (val - self.mean) ** 2)
        return -0.5 * (dim * tf.log(2.0 * np.pi) + tf.reduce_sum(tf.log(self.covar)) + quad_form)


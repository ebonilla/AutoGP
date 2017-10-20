from __future__ import absolute_import
import numpy as np
import tensorflow as tf

from autogp import util
from . import kernel


class RadialBasis(kernel.Kernel):
    MAX_DIST = 1e8

    def __init__(self, input_dim, lengthscale=1.0, std_dev=1.0,
                 white=0.01, input_scaling=False):
        if input_scaling:
            self.lengthscale = tf.Variable(lengthscale * tf.ones([input_dim]))
        else:
            self.lengthscale = tf.Variable([lengthscale], dtype=tf.float32)

        self.std_dev = tf.Variable([std_dev], dtype=tf.float32)
        self.input_dim = input_dim
        self.white = white

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0])
        else:
            white_noise = 0.0

        points1 = points1 / self.lengthscale
        points2 = points2 / self.lengthscale
        magnitude_square1 = tf.expand_dims(tf.reduce_sum(points1 ** 2, 1), 1)
        magnitude_square2 = tf.expand_dims(tf.reduce_sum(points2 ** 2, 1), 1)
        distances = (magnitude_square1 - 2 * tf.matmul(points1, tf.transpose(points2)) +
                     tf.transpose(magnitude_square2))
        distances = tf.clip_by_value(distances, 0.0, self.MAX_DIST);

        kern = ((self.std_dev ** 2) * tf.exp(-distances / 2.0))
        return kern + white_noise

    def diag_kernel(self, points):
        return ((self.std_dev ** 2) + self.white) * tf.ones([tf.shape(points)[0]])

    def get_params(self):
        return [self.lengthscale, self.std_dev]


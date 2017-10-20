from __future__ import absolute_import
import numpy as np
import tensorflow as tf

from autogp import util
from . import kernel


class ArcCosine(kernel.Kernel):
    def __init__(self, input_dim, degree=0, depth=1, lengthscale=1.0,
                 std_dev=1.0, white=1e-4, input_scaling=False):
        self.degree = degree
        self.depth = depth
        self.white = white
        self.std_dev = tf.Variable([std_dev], dtype=tf.float32)
        if input_scaling:
            self.lengthscale = tf.Variable(lengthscale * tf.ones([input_dim]))
        else:
            self.lengthscale = tf.Variable([lengthscale], dtype=tf.float32)

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0])
        else:
            white_noise = 0.0

        kern = self.recursive_kernel(points1 / self.lengthscale, points2 / self.lengthscale, self.depth)
        return (self.std_dev ** 2) * kern + white_noise

    def recursive_kernel(self, points1, points2, depth):
        if depth == 1:
            mag_sqr1 = tf.expand_dims(tf.reduce_sum(points1 ** 2, 1), 1)
            mag_sqr2 = tf.expand_dims(tf.reduce_sum(points2 ** 2, 1), 1)
            point_prod = tf.matmul(points1, tf.transpose(points2))
        else:
            mag_sqr1 = tf.expand_dims(
                self.diag_recursive_kernel(points1, depth - 1),
                1)
            mag_sqr2 = tf.expand_dims(
                self.diag_recursive_kernel(points2, depth - 1),
                1)
            point_prod = self.recursive_kernel(points1, points2, depth - 1)

        mag_prod = tf.sqrt(mag_sqr1) * tf.transpose(tf.sqrt(mag_sqr2))
        cos_angles = (2 * point_prod) / (
            tf.sqrt(1 + 2 * mag_sqr1) * tf.transpose(
                tf.sqrt(1 + 2 * mag_sqr2)
            )
        )

        return (
            ((mag_prod ** self.degree) / np.pi) *
            self.angular_func(cos_angles)
        )

    def diag_kernel(self, points):
        return (self.std_dev ** 2) * self.diag_recursive_kernel(
            points / self.lengthscale, self.depth
        ) + self.white

    # TODO(karl): Add a memoize decorator.
    # @util.memoize
    def diag_recursive_kernel(self, points, depth):
        # TODO(karl): Consider computing this in closed form.
        if depth == 1:
            mag_sqr = tf.reduce_sum(points ** 2, 1)
        else:
            mag_sqr = self.diag_recursive_kernel(points, depth - 1)

        return (
            (mag_sqr ** self.degree) * self.angular_func(
                2 * mag_sqr / (1 + 2 * mag_sqr)
            ) / np.pi)

    def angular_func(self, cos_angles):
        angles = tf.acos(cos_angles)
        sin_angles = tf.sin(angles)
        pi_diff = np.pi - angles
        if self.degree == 0:
            return pi_diff
        elif self.degree == 1:
            return sin_angles + pi_diff * cos_angles
        elif self.degree == 2:
            return 3 * sin_angles * cos_angles + pi_diff * (
                1 + 2 * cos_angles ** 2
            )
        else:
            assert False

    def get_params(self):
        return [self.std_dev, self.lengthscale]

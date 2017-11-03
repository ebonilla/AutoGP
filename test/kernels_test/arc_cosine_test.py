import unittest

import numpy as np
import tensorflow as tf

from autogp import kernels


SIG_FIGS = 5


class TestArcCosine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestArcCosine, cls).setUpClass()
        cls.session.run(tf.global_variables_initializer())

    def kernel(self, points1, points2=None, degree=0, depth=1):
        arc_cosine = kernels.ArcCosine(degree, depth, white=0.0)
        if points2 is not None:
            return tf.Session().run(arc_cosine.kernel(np.array(points1, dtype=np.float32),
                                                      np.array(points2, dtype=np.float32)))
        else:
            return tf.Session().run(arc_cosine.kernel(np.array(points1, dtype=np.float32)))

    def diag_kernel(self, points, degree=0, depth=1):
        arc_cosine = kernels.ArcCosine(degree, depth, white=0.0)
        return tf.Session().run(arc_cosine.diag_kernel(np.array(points, dtype=np.float32)))

    def test_simple_kern(self):
        kern = self.kernel([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
        np.testing.assert_almost_equal(kern, [[1.0, 0.5, 0.5],
                                              [0.5, 1.0, 0.5],
                                              [0.5, 0.5, 1.0]])

    def test_parallel_kern(self):
        kern = self.kernel([[3.0, 5.0, 2.0],
                            [-3.0, -5.0, -2.0],
                            [6.0, 10.0, 4.0]])
        np.testing.assert_almost_equal(kern, [[1.0, 0.0, 1.0],
                                              [0.0, 1.0, 0.0],
                                              [1.0, 0.0, 1.0]])

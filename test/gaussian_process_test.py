import unittest

import numpy as np
import scipy.misc
import scipy.stats
import tensorflow as tf

import autogp
from autogp import kernels
from autogp import likelihoods


SIG_FIGS = 5


class TestGaussianProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We expect the child class to instantiate `cls.model` for us.
        cls.session = tf.Session()

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    @classmethod
    def entropy(cls, weights, means, covars):
        entropy = cls.model._build_entropy(weights=np.array(weights, dtype=np.float32),
                                           means=np.array(means, dtype=np.float32),
                                           covars=np.array(covars, dtype=np.float32))
        return cls.session.run(entropy)

    @classmethod
    def cross_ent(cls, weights, means, covars, kernel_chol):
        cross_ent = cls.model._build_cross_ent(weights=np.array(weights, dtype=np.float32),
                                               means=np.array(means, dtype=np.float32),
                                               covars=np.array(covars, dtype=np.float32),
                                               kernel_chol=np.array(kernel_chol, dtype=np.float32))
        return cls.session.run(cross_ent)

    @classmethod
    def interim_vals(cls, kernel_chol, inducing_inputs, train_inputs):
        kern_prods, kern_sums = cls.model._build_interim_vals(
            kernel_chol=np.array(kernel_chol, dtype=np.float32),
            inducing_inputs=np.array(inducing_inputs, dtype=np.float32),
            train_inputs=np.array(train_inputs, dtype=np.float32))
        return cls.session.run([kern_prods, kern_sums])

    @classmethod
    def sample_info(cls, kern_prods, kern_sums, means, covars):
        mean, var = cls.model._build_sample_info(kern_prods=np.array(kern_prods, dtype=np.float32),
                                                 kern_sums=np.array(kern_sums, dtype=np.float32),
                                                 means=np.array(means, dtype=np.float32),
                                                 covars=np.array(covars, dtype=np.float32))
        return cls.session.run([mean, var])


class TestSimpleFull(TestGaussianProcess):
    @classmethod
    def setUpClass(cls):
        super(TestSimpleFull, cls).setUpClass()
        likelihood = likelihoods.Gaussian(1.0)
        kernel = [kernels.RadialBasis(input_dim=1, lengthscale=1.0, std_dev=1.0, white=0.0)]
        # In most of our unit test, we will replace this value with something else.
        inducing_inputs = np.array([[1.0]])
        cls.model = autogp.GaussianProcess(likelihood_func=likelihood,
                                           kernel_funcs=kernel,
                                           inducing_inputs=inducing_inputs,
                                           num_components=1,
                                           diag_post=False,
                                           num_samples=10)
        cls.session.run(tf.initialize_all_variables())

    def test_simple_entropy(self):
        entropy = TestSimpleFull.entropy(weights=[1.0],
                                         means=[[[1.0]]],
                                         covars=[[[[1.0]]]])
        self.assertAlmostEqual(entropy, 0.5 * (np.log(2 * np.pi) + np.log(2.0)), SIG_FIGS)

    def test_small_covar_entropy(self):
        entropy = TestSimpleFull.entropy(weights=[1.0],
                                         means=[[[1.0]]],
                                         covars=[[[[1e-10]]]])
        self.assertAlmostEqual(entropy, 0.5 * (np.log(2 * np.pi) + np.log(2 * 1e-20)), SIG_FIGS)

    def test_large_covar_entropy(self):
        entropy = TestSimpleFull.entropy(weights=[1.0],
                                         means=[[[1.0]]],
                                         covars=[[[[1e10]]]])
        self.assertAlmostEqual(entropy, 0.5 * (np.log(2 * np.pi) + np.log(2 * 1e20)), SIG_FIGS)

    def test_simple_cross_ent(self):
        cross_ent = TestSimpleFull.cross_ent(weights=[1.0],
                                             means=[[[1.0]]],
                                             covars=[[[[1.0]]]],
                                             kernel_chol=[[[1.0]]])
        self.assertAlmostEqual(cross_ent, -0.5 * (np.log(2 * np.pi) + np.log(1.0) + 2.0),
                               SIG_FIGS)

    def test_small_cross_ent(self):
        cross_ent = TestSimpleFull.cross_ent(weights=[1.0],
                                             means=[[[1e-10]]],
                                             covars=[[[[1e-10]]]],
                                             kernel_chol=[[[1e-10]]])
        self.assertAlmostEqual(cross_ent, -0.5 * (np.log(2 * np.pi) + np.log(1e-20) + 2.0),
                               SIG_FIGS)

    def test_large_cross_ent(self):
        cross_ent = TestSimpleFull.cross_ent(weights=[1.0],
                                             means=[[[1e10]]],
                                             covars=[[[[1e10]]]],
                                             kernel_chol=[[[1e10]]])
        self.assertAlmostEqual(cross_ent, -0.5 * (np.log(2 * np.pi) + np.log(1e20) + 2.0),
                               SIG_FIGS)

    def test_simple_interim_vals(self):
        kern_prods, kern_sums = TestSimpleFull.interim_vals(kernel_chol=[[[1.0]]],
                                                            inducing_inputs=[[[1.0]]],
                                                            train_inputs=[[1.0]])
        self.assertAlmostEqual(kern_prods, 1.0, SIG_FIGS)
        self.assertAlmostEqual(kern_sums, 0.0, SIG_FIGS)

    def test_small_interim_vals(self):
        kern_prods, kern_sums = TestSimpleFull.interim_vals(kernel_chol=[[[1e-8]]],
                                                            inducing_inputs=[[[1e-8]]],
                                                            train_inputs=[[1e-8]])
        self.assertAlmostEqual(kern_prods, 1e16, SIG_FIGS)
        self.assertAlmostEqual(kern_sums, 1 - 1e16, SIG_FIGS)
        
    def test_large_interim_vals(self):
        kern_prods, kern_sums = TestSimpleFull.interim_vals(kernel_chol=[[[1e8]]],
                                                            inducing_inputs=[[[1e8]]],
                                                            train_inputs=[[1e8]])
        self.assertAlmostEqual(kern_prods, 1e-8, SIG_FIGS)
        self.assertAlmostEqual(kern_sums, 1 - 1e-8, SIG_FIGS)

    def test_multiple_inputs_interim_vals(self):
        inducing_distances = np.array([[1.0,          np.exp(-0.5), np.exp(-2.0)],
                                       [np.exp(-0.5), 1.0,          np.exp(-0.5)],
                                       [np.exp(-2.0), np.exp(-0.5), 1.0         ]],
                                      dtype=np.float32)
        kern_chol = np.linalg.cholesky(inducing_distances)[np.newaxis, :, :]
        kern_prods, kern_sums = TestSimpleFull.interim_vals(kern_chol,
                                                            inducing_inputs=[[[1.0], [2.0], [3.0]]],
                                                            train_inputs=[[3.0], [4.0]])
        train_inducing_distances = np.array([[np.exp(-2.0), np.exp(-0.5), 1.0         ],
                                             [np.exp(-4.5), np.exp(-2.0), np.exp(-0.5)]],
                                            dtype=np.float32)

        real_kern_prods = np.dot(train_inducing_distances, np.linalg.inv(inducing_distances))
        real_kern_sums = np.ones([2]) - np.diag(np.dot(real_kern_prods, train_inducing_distances.T))

        np.testing.assert_almost_equal(kern_prods[0], real_kern_prods, SIG_FIGS)
        np.testing.assert_almost_equal(kern_sums[0], real_kern_sums, SIG_FIGS)

    def test_simple_sample_info(self):
        mean, var = TestSimpleFull.sample_info(kern_prods=[[[2.0]]],
                                               kern_sums=[[3.0]],
                                               means=[[4.0]],
                                               covars=[[[5.0]]])
        self.assertAlmostEqual(mean, 8.0, SIG_FIGS)
        self.assertAlmostEqual(var, 103.0, SIG_FIGS)

    def test_multi_sample_info(self):
        mean, var = TestSimpleFull.sample_info(kern_prods=[[[1.0, 2.0],
                                                            [3.0, 4.0]]],
                                               kern_sums=[[5.0, 6.0]],
                                               means=[[7.0, 8.0]],
                                               covars=[[[9.0 , 10.0],
                                                        [11.0, 12.0]]])
        np.testing.assert_almost_equal(mean, [[23.0], [53.0]], SIG_FIGS)
        np.testing.assert_almost_equal(var, [[2122.0], [11131.0]], SIG_FIGS)


class TestSimpleDiag(TestGaussianProcess):
    @classmethod
    def setUpClass(cls):
        super(TestSimpleDiag, cls).setUpClass()
        likelihood = likelihoods.Gaussian(1.0)
        kernel = [kernels.RadialBasis(input_dim=1, lengthscale=1.0, std_dev=1.0, white=0.0)]
        # In most of our unit test, we will replace this value with something else.
        inducing_inputs = np.array([[1.0]])
        cls.model = autogp.GaussianProcess(likelihood_func=likelihood,
                                           kernel_funcs=kernel,
                                           inducing_inputs=inducing_inputs,
                                           num_components=1,
                                           diag_post=True,
                                           num_samples=10)
        cls.session.run(tf.initialize_all_variables())

    def test_simple_entropy(self):
        entropy = TestSimpleDiag.entropy(weights=[1.0],
                                         means=[[[1.0]]],
                                         covars=[[[1.0]]])
        self.assertAlmostEqual(entropy, 0.5 * (np.log(2 * np.pi) + np.log(2.0)), SIG_FIGS)

    def test_small_covar_entropy(self):
        entropy = TestSimpleDiag.entropy(weights=[1.0],
                                         means=[[[1.0]]],
                                         covars=[[[1e-10]]])
        self.assertAlmostEqual(entropy, 0.5 * (np.log(2 * np.pi) + np.log(2 * 1e-10)), SIG_FIGS)

    def test_large_covar_entropy(self):
        entropy = TestSimpleDiag.entropy(weights=[1.0],
                                         means=[[[1.0]]],
                                         covars=[[[1e10]]])
        self.assertAlmostEqual(entropy, 0.5 * (np.log(2 * np.pi) + np.log(2 * 1e10)), SIG_FIGS)

    def test_simple_cross_ent(self):
        cross_ent = TestSimpleDiag.cross_ent(weights=[1.0],
                                             means=[[[1.0]]],
                                             covars=[[[1.0]]],
                                             kernel_chol=[[[1.0]]])
        self.assertAlmostEqual(cross_ent, -0.5 * (np.log(2 * np.pi) + 2.0), SIG_FIGS)

    def test_small_cross_ent(self):
        cross_ent = TestSimpleDiag.cross_ent(weights=[1.0],
                                             means=[[[1e-10]]],
                                             covars=[[[1e-10]]],
                                             kernel_chol=[[[1e-10]]])
        self.assertAlmostEqual(cross_ent, -0.5 * (np.log(2 * np.pi) + np.log(1e-20) + 1.0 + 1e10),
                               SIG_FIGS)

    def test_large_cross_ent(self):
        cross_ent = TestSimpleDiag.cross_ent(weights=[1.0],
                                             means=[[[1e10]]],
                                             covars=[[[1e10]]],
                                             kernel_chol=[[[1e10]]])
        self.assertAlmostEqual(cross_ent, -0.5 * (np.log(2 * np.pi) + np.log(1e20) + 1.0 + 1e-10),
                               SIG_FIGS)

    def test_simple_sample_info(self):
        mean, var = TestSimpleDiag.sample_info(kern_prods=[[[2.0]]],
                                               kern_sums=[[3.0]],
                                               means=[[4.0]],
                                               covars=[[5.0]])
        self.assertAlmostEqual(mean, 8.0, SIG_FIGS)
        self.assertAlmostEqual(var, 23.0, SIG_FIGS)

    def test_multi_sample_info(self):
        mean, var = TestSimpleDiag.sample_info(kern_prods=[[[1.0, 2.0],
                                                            [3.0, 4.0]]],
                                               kern_sums=[[5.0, 6.0]],
                                               means=[[7.0, 8.0]],
                                               covars=[[9.0 , 10.0]])
        np.testing.assert_almost_equal(mean, [[23.0], [53.0]], SIG_FIGS)
        np.testing.assert_almost_equal(var, [[54.0], [247.0]], SIG_FIGS)


class TestMultiFull(TestGaussianProcess):
    @classmethod
    def setUpClass(cls):
        super(TestMultiFull, cls).setUpClass()
        likelihood = likelihoods.Softmax()
        kernel = [kernels.RadialBasis(input_dim=2, lengthscale=1.0, std_dev=1.0, white=0.0)
                  for i in range(2)]
        inducing_locations = np.array([[1.0, 2.0, 3.0, 4.0]])
        cls.model = autogp.GaussianProcess(likelihood_func=likelihood,
                                           kernel_funcs=kernel,
                                           inducing_inputs=inducing_locations,
                                           num_components=2,
                                           diag_post=False,
                                           num_samples=1)
        cls.session.run(tf.initialize_all_variables())

    def test_entropy(self):
        entropy = TestMultiFull.entropy(weights=[0.7, 0.3],
                                        means=[[[01.0, 02.0],
                                                [03.0, 04.0]],
                                               [[05.0, 06.0],
                                                [07.0, 08.0]]],
                                        covars=[[[[0.1, 0.0],
                                                  [0.2, 0.3]],
                                                 [[0.4, 0.0],
                                                  [0.5, 0.6]]],
                                                [[[0.7, 0.0],
                                                  [0.8, 0.9]],
                                                 [[1.0, 0.0],
                                                  [1.1, 1.2]]]])
        n11_1 = scipy.stats.multivariate_normal.logpdf([1.0, 2.0], [1.0, 2.0], [[0.02, 0.04],
                                                                                [0.02, 0.26]])
        n11_2 = scipy.stats.multivariate_normal.logpdf([3.0, 4.0], [3.0, 4.0], [[0.32, 0.40],
                                                                                [0.40, 1.22]])
        n12_1 = scipy.stats.multivariate_normal.logpdf([1.0, 2.0], [5.0, 6.0], [[0.50, 0.58],
                                                                                [0.58, 1.58]])
        n12_2 = scipy.stats.multivariate_normal.logpdf([3.0, 4.0], [7.0, 8.0], [[1.16, 1.30],
                                                                                [1.30, 3.26]])
        n21_1 = scipy.stats.multivariate_normal.logpdf([5.0, 6.0], [1.0, 2.0], [[0.50, 0.58],
                                                                                [0.58, 1.58]])
        n21_2 = scipy.stats.multivariate_normal.logpdf([7.0, 8.0], [3.0, 4.0], [[1.16, 1.30],
                                                                                [1.30, 3.26]])
        n22_1 = scipy.stats.multivariate_normal.logpdf([5.0, 6.0], [5.0, 6.0], [[0.98, 1.12],
                                                                                [1.12, 2.90]])
        n22_2 = scipy.stats.multivariate_normal.logpdf([7.0, 8.0], [7.0, 8.0], [[2.00, 2.20],
                                                                                [2.20, 5.30]])
        true_ent = -(
            0.7 * scipy.misc.logsumexp([np.log(0.7) + n11_1 + n11_2, np.log(0.3) + n12_1 + n12_2]) +
            0.3 * scipy.misc.logsumexp([np.log(0.7) + n21_1 + n21_2, np.log(0.3) + n22_1 + n22_2]))
        self.assertAlmostEqual(entropy, true_ent, SIG_FIGS - 4)

    def test_cross_ent(self):
        cross_ent = TestMultiFull.cross_ent(weights=[0.3, 0.7],
                                            means=[[[01.0, 02.0],
                                                    [03.0, 04.0]],
                                                   [[05.0, 06.0],
                                                    [07.0, 08.0]]],
                                            covars=[[[[01.0, 00.0],
                                                      [02.0, 03.0]],
                                                     [[04.0, 00.0],
                                                      [05.0, 06.0]]],
                                                    [[[07.0, 00.0],
                                                      [08.0, 09.0]],
                                                     [[10.0, 00.0],
                                                      [11.0, 12.0]]]],
                                            kernel_chol=[[[13.0, 0.0],
                                                          [14.0, 15.0]],
                                                         [[16.0, 0.0],
                                                          [17.0, 18.0]]])
        n11 = scipy.stats.multivariate_normal.logpdf([0.0, 0.0], [1.0, 2.0], [[169.0, 182.0],
                                                                              [182.0, 421.0]])
        n12 = scipy.stats.multivariate_normal.logpdf([0.0, 0.0], [3.0, 4.0], [[256.0, 272.0],
                                                                              [272.0, 613.0]])
        n21 = scipy.stats.multivariate_normal.logpdf([0.0, 0.0], [5.0, 6.0], [[169.0, 182.0],
                                                                              [182.0, 421.0]])
        n22 = scipy.stats.multivariate_normal.logpdf([0.0, 0.0], [7.0, 8.0], [[256.0, 272.0],
                                                                              [272.0, 613.0]])
        ki_1 = scipy.linalg.inv([[169.0, 182.0],
                                 [182.0, 421.0]])
        ki_2 = scipy.linalg.inv([[256.0, 272.0],
                                 [272.0, 613.0]])
        p11 = np.dot(ki_1, [[1.0, 2.0],
                            [2.0, 13.0]])
        p12 = np.dot(ki_2, [[16.0, 20.0],
                            [20.0, 61.0]])
        p21 = np.dot(ki_1, [[49.0, 56.0],
                            [56.0, 145.0]])
        p22 = np.dot(ki_2, [[100.0, 110.0],
                            [110.0, 265.0]])
        t11 = np.trace(p11)
        t12 = np.trace(p12)
        t21 = np.trace(p21)
        t22 = np.trace(p22)
        self.assertAlmostEqual(cross_ent, (0.3 * (n11 - 0.5 * t11 + n12 - 0.5 * t12) +
                                           0.7 * (n21 - 0.5 * t21 + n22 - 0.5 * t22)),
                               SIG_FIGS)

    def test_interim_vals(self):
        kern_prods, kern_sums = TestMultiFull.interim_vals(kernel_chol=[[[1.0, 0.0],
                                                                         [2.0, 3.0]],
                                                                        [[4.0, 0.0],
                                                                         [5.0, 6.0]]],
                                                           inducing_inputs=[[[7.0, 8.0],
                                                                             [9.0, 10.0]],
                                                                            [[11.0, 12.0],
                                                                             [13.0, 14.0]]],
                                                           train_inputs=[[15.0, 16.0],
                                                                         [17.0, 18.0]])
        kxz_1 = np.array([[np.exp(-64.0), np.exp(-36.0)],
                          [np.exp(-100.0), np.exp(-64.0)]])
        kxz_2 = np.array([[np.exp(-16.0), np.exp(-4.0)],
                          [np.exp(-36.0), np.exp(-16.0)]])
        kxx = np.array([[1.0, np.exp(-4.0)],
                        [np.exp(-4.0), 1.0]])
        kzz_inv1 = scipy.linalg.inv(np.array([[1.0, 2.0],
                                              [2.0, 13.0]]))
        kzz_inv2 = scipy.linalg.inv(np.array([[16.0, 20.0],
                                              [20.0, 61.0]]))
        a_1 = np.dot(kxz_1, kzz_inv1)
        a_2 = np.dot(kxz_2, kzz_inv2)
        np.testing.assert_almost_equal(kern_prods[0], a_1, SIG_FIGS)
        np.testing.assert_almost_equal(kern_prods[1], a_2, SIG_FIGS)
        np.testing.assert_almost_equal(kern_sums[0], np.diag(kxx - np.dot(a_1, kxz_1.T)), SIG_FIGS)
        np.testing.assert_almost_equal(kern_sums[1], np.diag(kxx - np.dot(a_2, kxz_2.T)), SIG_FIGS)

    def test_sample_info(self):
        mean, var = TestMultiFull.sample_info(kern_prods=[[[1.0, 2.0],
                                                           [3.0, 4.0]],
                                                          [[5.0, 6.0],
                                                           [7.0, 8.0]]],
                                              kern_sums=[[9.0, 10.0],
                                                         [11.0, 12.0]],
                                              means=[[13.0, 14.0],
                                                     [15.0, 16.0]],
                                              covars=[[[17.0 , 0.0],
                                                       [19.0, 20.0]],
                                                      [[21.0, 0.0],
                                                       [22.0, 23.0]]])
        true_mean = np.array([[41.0, 171.0],
                              [95.0, 233.0]])
        true_var = np.array([[4634.0, 75224.0],
                             [22539.0, 138197.0]])
        np.testing.assert_almost_equal(mean, true_mean, SIG_FIGS)
        np.testing.assert_almost_equal(var, true_var, SIG_FIGS)


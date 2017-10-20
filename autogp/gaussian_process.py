from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from . import util


class GaussianProcess(object):
    """
    The class representing the AutoGP model.

    Parameters
    ----------
    likelihood_func : subclass of likelihoods.Likelihood
        An object representing the likelihood function p(y|f).
    kernel_funcs : list of subclasses of kernels.Kernel
        A list of one kernel per latent function.
    inducing_inputs : ndarray
        An array of initial inducing input locations
        Dimensions: num_inducing * input_dim.
    num_components : int
        The number of mixture of Gaussian components.
    diag_post : bool
        True if the mixture of Gaussians uses a diagonal covariance,
        False otherwise.
    num_samples : int
        The number of samples to approximate the expected log likelihood
        of the posterior.
    """
    def __init__(self,
                 likelihood_func,
                 kernel_funcs,
                 inducing_inputs,
                 num_components=1,
                 diag_post=False,
                 num_samples=100):
        # Get the actual functions if they were initialized as strings.
        self.likelihood = likelihood_func
        self.kernels = kernel_funcs

        # Save whether our posterior is diagonal or not.
        self.diag_post = diag_post

        # Repeat the inducing inputs for all latent processes
        # if we haven't been given individually
        # specified inputs per process.
        if inducing_inputs.ndim == 2:
            inducing_inputs = np.tile(
                inducing_inputs[np.newaxis, :, :],
                [len(self.kernels), 1, 1]
            )

        # Initialize all model dimension constants.
        self.num_components = num_components
        self.num_latent = len(self.kernels)
        self.num_samples = num_samples
        self.num_inducing = inducing_inputs.shape[1]
        self.input_dim = inducing_inputs.shape[2]

        # Define all parameters that get optimized directly in raw form.
        # Some parameters get
        # transformed internally to maintain certain pre-conditions.
        self.raw_weights = tf.Variable(tf.zeros([self.num_components]))
        self.raw_means = tf.Variable(tf.zeros([
            self.num_components,
            self.num_latent,
            self.num_inducing]))
        if self.diag_post:
            self.raw_covars = tf.Variable(tf.ones([
                self.num_components,
                self.num_latent,
                self.num_inducing]))
        else:
            init_vec = np.zeros([
                self.num_components, self.num_latent] +
                util.tri_vec_shape(self.num_inducing),
                dtype=np.float32)
            self.raw_covars = tf.Variable(init_vec)
        self.raw_inducing_inputs = tf.Variable(
            inducing_inputs, dtype=tf.float32)
        self.raw_likelihood_params = self.likelihood.get_params()
        self.raw_kernel_params = sum([
            k.get_params() for k in self.kernels], []
        )

        # Define placeholder variables for training and predicting.
        self.num_train = tf.placeholder(tf.float32, shape=[], name="num_train")
        self.train_inputs = tf.placeholder(
            tf.float32,
            shape=[None, self.input_dim],
            name="train_inputs")
        self.train_outputs = tf.placeholder(
            tf.float32, shape=[None, None],
            name="train_outputs")
        self.test_inputs = tf.placeholder(
            tf.float32,
            shape=[None, self.input_dim],
            name="test_inputs")

        # Now build our computational graph.
        self.nelbo, self.loo_loss, self.predictions = self._build_graph(
            self.raw_weights,
            self.raw_means,
            self.raw_covars,
            self.raw_inducing_inputs,
            self.train_inputs,
            self.train_outputs,
            self.num_train,
            self.test_inputs)

        #config = tf.ConfigProto(
        # log_device_placement=True, allow_soft_placement=True)
        # Do all the tensorflow bookkeeping.
        self.session = tf.Session()
        self.optimizer = None
        self.train_step = None

    def fit(self, data, optimizer, loo_steps=10,  var_steps=10, epochs=200,
            batch_size=None, display_step=1, test=None, loss=None):
        """
        Fit the Gaussian process model to the given data.

        Parameters
        ----------
        data : subclass of datasets.DataSet
            The train inputs and outputs.
        optimizer : TensorFlow optimizer
            The optimizer to use in the fitting process.
        loo_steps : int
            Number of steps  to update hyper-parameters using loo objective
        var_steps : int
            Number of steps to update  variational parameters using variational
            objective (elbo).
        epochs : int
            The number of epochs to optimize the model for.
        batch_size : int
            The number of datapoints to use per mini-batch when training.
            If batch_size is None,
            then we perform batch gradient descent.
        display_step : int
            The frequency at which the objective values are printed out.
        """
        num_train = data.num_examples
        if batch_size is None:
            batch_size = num_train

        if self.optimizer != optimizer:
            self.optimizer = optimizer
            self.loo_train_step = optimizer.minimize(
                self.loo_loss, var_list=[
                    self.raw_inducing_inputs
                ] +
                self.raw_kernel_params +
                self.raw_likelihood_params
            )
            self.train_step = optimizer.minimize(self.nelbo)
            self.session.run(tf.global_variables_initializer())

        start = data.next_batch(batch_size)

        old_epoch = 0
        while data.epochs_completed < epochs:
            num_epochs = data.epochs_completed + var_steps
            while data.epochs_completed < num_epochs:
                batch = data.next_batch(batch_size)
                self.session.run(
                    self.train_step,
                    feed_dict={
                        self.train_inputs: batch[0],
                        self.train_outputs: batch[1],
                        self.num_train: num_train})
                if (data.epochs_completed % display_step == 0
                        and data.epochs_completed != old_epoch):
                    self._print_state(data, test, loss, num_train)
                    old_epoch = data.epochs_completed

            num_epochs = data.epochs_completed + loo_steps
            while data.epochs_completed < num_epochs:
                batch = data.next_batch(batch_size)
                self.session.run(self.loo_train_step, feed_dict={
                    self.train_inputs: batch[0],
                    self.train_outputs: batch[1],
                    self.num_train: num_train})
                if (
                        data.epochs_completed % display_step == 0 and
                        data.epochs_completed != old_epoch):
                    self._print_state(data, test, loss, num_train)
                    old_epoch = data.epochs_completed

    def predict(self, test_inputs, batch_size=None):
        """
        Predict outputs given inputs.

        Parameters
        ----------
        test_inputs : ndarray
            Points on which we wish to make predictions.
            Dimensions: num_test * input_dim.
        batch_size : int
            The size of the batches we make predictions on.
            If batch_size is None, predict on the
            entire test set at once.

        Returns
        -------
        ndarray
            The predicted mean of the test inputs.
            Dimensions: num_test * output_dim.
        ndarray
            The predicted variance of the test inputs.
            Dimensions: num_test * output_dim.
        """
        if batch_size is None:
            num_batches = 1
        else:
            num_batches = util.ceil_divide(test_inputs.shape[0], batch_size)

        test_inputs = np.array_split(test_inputs, num_batches)
        pred_means = util.init_list(0.0, [num_batches])
        pred_vars = util.init_list(0.0, [num_batches])
        for i in range(num_batches):
            pred_means[i], pred_vars[i] = self.session.run(
                self.predictions, feed_dict={self.test_inputs: test_inputs[i]})

        return np.concatenate(pred_means, axis=0), np.concatenate(pred_vars, axis=0)

    def _print_state(self, data, test, loss, num_train):
        if num_train <= 100000:
            nelbo = self.session.run(self.nelbo, feed_dict={
                self.train_inputs: data.X,
                self.train_outputs: data.Y,
                self.num_train: num_train})
            loo = self.session.run(self.loo_loss, feed_dict={
                self.train_inputs: data.X,
                self.train_outputs: data.Y,
                self.num_train: num_train})
            print(
                "i=" + repr(data.epochs_completed) +
                " nelbo=" + repr(nelbo), end=" ")
            print("loo=" + repr(loo))

        if loss is not None:
            ypred = self.predict(test.X)[0]
            print(
                "i=" + repr(data.epochs_completed) +
                " curent " + loss.get_name() + "=" + "%.4f"
                % loss.eval(test.Y, ypred))

    def _build_graph(
            self,
            raw_weights,
            raw_means,
            raw_covars,
            raw_inducing_inputs,
            train_inputs,
            train_outputs,
            num_train,
            test_inputs):
        # First transform all raw variables into their internal form.
        # Use softmax(raw_weights) to keep all weights normalized.
        weights = tf.exp(raw_weights) / tf.reduce_sum(tf.exp(raw_weights))

        if self.diag_post:
            # Use exp(raw_covars) so as to guarantee the diagonal matrix
            # remains positive definite.
            covars = tf.exp(raw_covars)
        else:
            # Use vec_to_tri(raw_covars) so as to only optimize
            # over the lower triangular portion.
            # We note that we will always operate
            # over the cholesky space internally.
            covars_list = [None] * self.num_components
            for i in range(self.num_components):
                mat = util.vec_to_tri(raw_covars[i, :, :])
                diag_mat = tf.matrix_diag(tf.matrix_diag_part(mat))
                exp_diag_mat = tf.matrix_diag(tf.exp(tf.matrix_diag_part(mat)))
                covars_list[i] = mat - diag_mat + exp_diag_mat
            covars = tf.stack(covars_list, 0)
        # Both inducing inputs and the posterior means can vary freely so don't change them.
        means = raw_means
        inducing_inputs = raw_inducing_inputs

        # Build the matrices of covariances between inducing inputs.
        kernel_mat = [
            self.kernels[i].kernel(inducing_inputs[i, :, :])
            for i in range(self.num_latent)]
        kernel_chol = tf.stack([tf.cholesky(k) for k in kernel_mat], 0)

        # Now build the objective function.
        entropy = self._build_entropy(weights, means, covars)
        cross_ent = self._build_cross_ent(weights, means, covars, kernel_chol)
        ell = self._build_ell(weights, means, covars, inducing_inputs,
                              kernel_chol, train_inputs, train_outputs)
        batch_size = tf.to_float(tf.shape(train_inputs)[0])
        nelbo = -((batch_size / num_train) * (entropy + cross_ent) + ell)

        # Build the leave one out loss function.
        loo_loss = self._build_loo_loss(
            weights, means, covars, inducing_inputs,
            kernel_chol, train_inputs, train_outputs)

        # Finally, build the prediction function.
        predictions = self._build_predict(
            weights, means, covars, inducing_inputs,
            kernel_chol, test_inputs)

        return nelbo, loo_loss, predictions

    def _build_loo_loss(self, weights, means, covars, inducing_inputs,
                        kernel_chol, train_inputs, train_outputs):
        kern_prods, kern_sums = self._build_interim_vals(
            kernel_chol, inducing_inputs, train_inputs)
        loss = 0
        for i in range(self.num_components):
            covar_input = (
                covars[i, :, :]
                if self.diag_post
                else covars[i, :, :, :]
            )
            latent_samples = self._build_samples(
                kern_prods, kern_sums,
                means[i, :, :], covar_input)
            loss += weights[i] * tf.reduce_mean(
                1.0 / (tf.exp(
                    self.likelihood.log_cond_prob(
                        train_outputs, latent_samples
                    )
                ) + 1e-7),
                0
            )
        return tf.reduce_sum(tf.log(loss))

    def _build_predict(self, weights, means, covars, inducing_inputs,
                       kernel_chol, test_inputs):
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, inducing_inputs, test_inputs)
        pred_means = util.init_list(0.0, [self.num_components])
        pred_vars = util.init_list(0.0, [self.num_components])
        for i in range(self.num_components):
            covar_input = covars[i, :, :] if self.diag_post else covars[i, :, :, :]
            sample_means, sample_vars = self._build_sample_info(kern_prods, kern_sums,
                                                                means[i, :, :], covar_input)
            pred_means[i], pred_vars[i] = self.likelihood.predict(sample_means, sample_vars)

        pred_means = tf.stack(pred_means, 0)
        pred_vars = tf.stack(pred_vars, 0)

        # Compute the mean and variance of the gaussian mixture from their components.
        weights = tf.expand_dims(tf.expand_dims(weights, 1), 1)
        weighted_means = tf.reduce_sum(weights * pred_means, 0)
        weighted_vars = (tf.reduce_sum(weights * (pred_means ** 2 + pred_vars), 0) -
                         tf.reduce_sum(weights * pred_means, 0) ** 2)
        return weighted_means, weighted_vars

    def _build_entropy(self, weights, means, covars):
        # First build half a square matrix of normals. This avoids re-computing symmetric normals.
        log_normal_probs = util.init_list(0.0, [self.num_components, self.num_components])
        for i in range(self.num_components):
            for j in range(i, self.num_components):
                for k in range(self.num_latent):
                    if self.diag_post:
                        normal = util.DiagNormal(means[i, k, :], covars[i, k, :] +
                                                                 covars[j, k, :])
                    else:
                        if i == j:
                            # Compute chol(2S) = sqrt(2)*chol(S).
                            covars_sum = tf.sqrt(2.0) * covars[i, k, :, :]
                        else:
                            # TODO(karl): Can we just stay in cholesky space somehow?
                            covars_sum = tf.cholesky(util.mat_square(covars[i, k, :, :]) +
                                                     util.mat_square(covars[j, k, :, :]))
                        normal = util.CholNormal(means[i, k, :], covars_sum)
                    log_normal_probs[i][j] += normal.log_prob(means[j, k, :])

        # Now compute the entropy.
        entropy = 0.0
        for i in range(self.num_components):
            weighted_log_probs = util.init_list(0.0, [self.num_components])
            for j in range(self.num_components):
                if i <= j:
                    weighted_log_probs[j] = tf.log(weights[j]) + log_normal_probs[i][j]
                else:
                    weighted_log_probs[j] = tf.log(weights[j]) + log_normal_probs[j][i]

            entropy -= weights[i] * util.logsumexp(tf.stack(weighted_log_probs))

        return entropy

    def _build_cross_ent(self, weights, means, covars, kernel_chol):
        cross_ent = 0.0
        for i in range(self.num_components):
            sum_val = 0.0
            for j in range(self.num_latent):
                if self.diag_post:
                    # TODO(karl): this is a bit inefficient since we're not making use of the fact
                    # that covars is diagonal. A solution most likely involves a custom tf op.
                    trace = tf.trace(tf.cholesky_solve(kernel_chol[j, :, :],
                                                       tf.diag(covars[i, j, :])))
                else:
                    trace = tf.reduce_sum(util.diag_mul(
                        tf.cholesky_solve(kernel_chol[j, :, :], covars[i, j, :, :]),
                        tf.transpose(covars[i, j, :, :])))

                sum_val += (util.CholNormal(means[i, j, :], kernel_chol[j, :, :]).log_prob(0.0) -
                            0.5 * trace)

            cross_ent += weights[i] * sum_val

        return cross_ent

    def _build_ell(self, weights, means, covars, inducing_inputs,
                   kernel_chol, train_inputs, train_outputs):
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, inducing_inputs, train_inputs)
        ell = 0
        for i in range(self.num_components):
            covar_input = covars[i, :, :] if self.diag_post else covars[i, :, :, :]
            latent_samples = self._build_samples(kern_prods, kern_sums,
                                                 means[i, :, :], covar_input)
            ell += weights[i] * tf.reduce_sum(self.likelihood.log_cond_prob(train_outputs,
                                                                            latent_samples))

        return ell / self.num_samples

    def _build_interim_vals(self, kernel_chol, inducing_inputs, train_inputs):
        kern_prods = util.init_list(0.0, [self.num_latent])
        kern_sums = util.init_list(0.0, [self.num_latent])
        for i in range(self.num_latent):
            ind_train_kern = self.kernels[i].kernel(inducing_inputs[i, :, :], train_inputs)
            # Compute A = Kxz.Kzz^(-1) = (Kzz^(-1).Kzx)^T.
            kern_prods[i] = tf.transpose(tf.cholesky_solve(kernel_chol[i, :, :], ind_train_kern))
            # We only need the diagonal components.
            kern_sums[i] = (self.kernels[i].diag_kernel(train_inputs) -
                            util.diag_mul(kern_prods[i], ind_train_kern))

        kern_prods = tf.stack(kern_prods, 0)
        kern_sums = tf.stack(kern_sums, 0)
        return kern_prods, kern_sums

    def _build_samples(self, kern_prods, kern_sums, means, covars):
        sample_means, sample_vars = self._build_sample_info(kern_prods, kern_sums, means, covars)
        batch_size = tf.shape(sample_means)[0]
        return (sample_means + tf.sqrt(sample_vars) *
                tf.random_normal([self.num_samples, batch_size, self.num_latent]))

    def _build_sample_info(self, kern_prods, kern_sums, means, covars):
        sample_means = util.init_list(0.0, [self.num_latent])
        sample_vars = util.init_list(0.0, [self.num_latent])
        for i in range(self.num_latent):
            if self.diag_post:
                quad_form = util.diag_mul(kern_prods[i, :, :] * covars[i, :],
                                          tf.transpose(kern_prods[i, :, :]))
            else:
                full_covar = tf.matmul(covars[i, :, :], tf.transpose(covars[i, :, :]))
                quad_form = util.diag_mul(tf.matmul(kern_prods[i, :, :], full_covar),
                                          tf.transpose(kern_prods[i, :, :]))
            sample_means[i] = tf.matmul(kern_prods[i, :, :], tf.expand_dims(means[i, :], 1))
            sample_vars[i] = tf.expand_dims(kern_sums[i, :] + quad_form, 1)

        sample_means = tf.concat( sample_means,1)
        sample_vars = tf.concat(sample_vars,1)
        return sample_means, sample_vars

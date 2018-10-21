import os
import subprocess
import sklearn.cluster
import numpy as np
import autogp
from autogp import likelihoods
from autogp import kernels
import tensorflow as tf
from autogp import datasets
from autogp import losses
from autogp  import util
import pandas
import scipy.io as sio


DATA_DIR = "experiments/data/"
TRAIN_PATH = DATA_DIR + "sarcos_inv.mat"
TEST_PATH = DATA_DIR + "sarcos_inv_test"

def init_z(train_inputs, num_inducing):
    # Initialize inducing points using clustering.
    mini_batch = sklearn.cluster.MiniBatchKMeans(num_inducing)
    cluster_indices = mini_batch.fit_predict(train_inputs)
    inducing_locations = mini_batch.cluster_centers_
    return inducing_locations


def get_sarcos_data():
    print "Getting sarcos data ..."
    os.chdir('experiments/data')
    subprocess.call(["./get_sarcos_data.sh"])
    os.chdir("../../")
    print "done"


def sarcos_all_joints_data():
    """
    Loads and returns data of SARCOS dataset for all joints.

    Returns
    -------
    data : list
        A list of length = 1, where each element is a dictionary which contains ``train_outputs``,
        ``train_inputs``, ``test_outputs``, ``test_inputs``, and ``id``
    """

    train = sio.loadmat(TRAIN_PATH)['sarcos_inv']
    test = sio.loadmat(TEST_PATH)['sarcos_inv_test']
    return{
        'train_inputs': train[:, :21],
        'train_outputs': train[:, 21:],
        'test_inputs': test[:, :21],
        'test_outputs': test[:, 21:],
        'id': 0
    }


if __name__ == '__main__':
    FLAGS = util.util.get_flags()
    BATCH_SIZE = FLAGS.batch_size
    LEARNING_RATE = FLAGS.learning_rate
    DISPLAY_STEP = FLAGS.display_step
    EPOCHS = FLAGS.n_epochs
    NUM_SAMPLES =  FLAGS.mc_train
    NUM_INDUCING = FLAGS.n_inducing
    IS_ARD = FLAGS.is_ard

    if os.path.exists(TRAIN_PATH) is False:  # directory does not exist, download the data
        get_sarcos_data()

    d = sarcos_all_joints_data()
    data = datasets.DataSet(d['train_inputs'].astype(np.float32), d['train_outputs'].astype(np.float32))
    test = datasets.DataSet(d['test_inputs'].astype(np.float32), d['test_outputs'].astype(np.float32))

    # Setup initial values for the model.
    likelihood = likelihoods.RegressionNetwork(7, 0.1)
    kern = [kernels.RadialBasis(data.X.shape[1], lengthscale=8.0, input_scaling = IS_ARD) for i in range(8)]
    # kern = [kernels.ArcCosine(data.X.shape[1], 1, 3, 5.0, 1.0, input_scaling=True) for i in range(10)]

    Z = init_z(data.X, NUM_INDUCING)
    m = autogp.GaussianProcess(likelihood, kern, Z, num_samples=NUM_SAMPLES)

    # setting up loss to be reported during training
    error_rate = None #losses.StandardizedMeanSqError(d['train_outputs'].astype(np.float32), data.Dout)

    import time
    o = tf.train.RMSPropOptimizer(LEARNING_RATE)
    start = time.time()
    m.fit(data, o, loo_steps=0, var_steps=50, epochs = EPOCHS, batch_size = BATCH_SIZE, display_step=DISPLAY_STEP, test = test,
            loss = error_rate )
    print time.time() - start

    ypred = m.predict(test.X)[0]
    print("Final " + error_rate.get_name() + "=" + "%.4f" % error_rate.eval(test.Y, ypred))


import sklearn.cluster
import numpy as np
import autogp
from autogp import likelihoods
from autogp import kernels
import tensorflow as tf
from autogp import datasets
from autogp import losses
from autogp  import util
import os
import subprocess

DATA_DIR = 'experiments/data/cifar-10-batches-py/'

def init_z(train_inputs, num_inducing):
    # Initialize inducing points using clustering.
    mini_batch = sklearn.cluster.MiniBatchKMeans(num_inducing)
    cluster_indices = mini_batch.fit_predict(train_inputs)
    inducing_locations = mini_batch.cluster_centers_
    return inducing_locations


def get_cifar_data():
    print "Getting cifar10 data ..."
    os.chdir('experiments/data')
    subprocess.call(["./get_cifar10_data.sh"])
    os.chdir("../../")
    print "done"

def load_cifar():
    if os.path.isdir(DATA_DIR) is False: # directory does not exist, download the data
        get_cifar_data()

    import cPickle
    train_X = np.empty([0, 3072], dtype=np.float32)
    train_Y = np.empty([0, 10], dtype=np.float32)
    for i in range(1, 6):
        f = open(DATA_DIR + "data_batch_" + str(i))
        d = cPickle.load(f)
        f.close()
        train_X = np.concatenate([train_X, d["data"]])
        train_Y = np.concatenate([train_Y, np.eye(10)[d["labels"]]])
    f = open(DATA_DIR + "test_batch")
    d = cPickle.load(f)
    f.close()
    train_X = train_X / 255.0
    test_X = np.array(d["data"], dtype=np.float32) / 255.0
    test_Y = np.array(np.eye(10)[d["labels"]], dtype=np.float32)
    return train_X, train_Y, test_X, test_Y


if __name__ == '__main__':
    FLAGS = util.util.get_flags()
    BATCH_SIZE = FLAGS.batch_size
    LEARNING_RATE = FLAGS.learning_rate
    DISPLAY_STEP = FLAGS.display_step
    EPOCHS = FLAGS.n_epochs
    NUM_SAMPLES =  FLAGS.mc_train
    NUM_INDUCING = FLAGS.n_inducing
    IS_ARD = FLAGS.is_ard

    train_X, train_Y, test_X, test_Y = load_cifar()
    data = datasets.DataSet(train_X, train_Y)
    test = datasets.DataSet(test_X, test_Y)

    # Setup initial values for the model.
    likelihood = likelihoods.Softmax()
    kern = [kernels.RadialBasis(data.X.shape[1], lengthscale=10.0, input_scaling = IS_ARD) for i in range(10)]
    # kern = [kernels.ArcCosine(X.shape[1], 2, 3, 5.0, 1.0, input_scaling=True) for i in range(10)] #RadialBasis(X.shape[1], input_scaling=True) for i in range(10)]

    Z = init_z(data.X, NUM_INDUCING)
    m = autogp.GaussianProcess(likelihood, kern, Z, num_samples=NUM_SAMPLES)

    # setting up loss to be reported during training
    error_rate = losses.ZeroOneLoss(data.Dout)

    o = tf.train.RMSPropOptimizer(LEARNING_RATE)
    m.fit(data, o, loo_steps=50, var_steps=50, epochs=EPOCHS, batch_size=BATCH_SIZE, display_step=DISPLAY_STEP, test=test,
          loss=error_rate)
    ypred = m.predict(test.X)[0]
    print("Final " + error_rate.get_name() + "=" + "%.4f" % error_rate.eval(test.Y, ypred))



import sklearn.cluster
import numpy as np
import autogp
from autogp import likelihoods
from autogp import kernels
import tensorflow as tf
from autogp import datasets
from autogp import losses
from autogp  import util
import subprocess
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_labels
from tensorflow.python.framework import dtypes
import gzip
import os


DATA_DIR = "experiments/data/infimnist/"
TRAIN_INPUTS = DATA_DIR + "train-patterns.gz"
TRAIN_OUTPUTS = DATA_DIR + "train-labels.gz"
TEST_INPUTS = DATA_DIR + "test-patterns.gz"
TEST_OUTPUTS = DATA_DIR + "test-labels.gz"

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D unit8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = int(_read32(bytestream))
    rows = int(_read32(bytestream))
    cols = int(_read32(bytestream))
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def process_mnist(images, dtype = dtypes.float32, reshape=True):
    if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
    if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

    return images


def get_mnist8m_data():
    print "Getting mnist8m data ..."
    os.chdir('experiments/data')
    subprocess.call(["./get_mnist8m_data.sh"])
    os.chdir("../../")
    print "done"

def import_mnist():
    if os.path.isdir(DATA_DIR) is False: # directory does not exist, download the data
        get_mnist8m_data()

    with open(TRAIN_INPUTS) as f:
        train_images = extract_images(f)
        train_images = process_mnist(train_images)

    with open(TRAIN_OUTPUTS) as f:
        train_labels = extract_labels(f, one_hot=True)

    with open(TEST_INPUTS) as f:
        test_images = extract_images(f)
        test_images = process_mnist(test_images)

    with open(TEST_OUTPUTS) as f:
        test_labels = extract_labels(f, one_hot=True)

    return datasets.DataSet(train_images, train_labels), datasets.DataSet(test_images, test_labels)


def init_z(train_inputs, num_inducing):
    # Initialize inducing points using clustering.
    mini_batch = sklearn.cluster.MiniBatchKMeans(num_inducing)
    cluster_indices = mini_batch.fit_predict(train_inputs)
    inducing_locations = mini_batch.cluster_centers_
    return inducing_locations


if __name__ == '__main__':
    FLAGS = util.util.get_flags()
    BATCH_SIZE = FLAGS.batch_size
    LEARNING_RATE = FLAGS.learning_rate
    DISPLAY_STEP = FLAGS.display_step
    EPOCHS = FLAGS.n_epochs
    NUM_SAMPLES =  FLAGS.mc_train
    NUM_INDUCING = FLAGS.n_inducing
    IS_ARD = FLAGS.is_ard

    data, test = import_mnist()

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



import numpy as np

class DataSet():

    def __init__(self, X, Y, shuffle=True):
        self._num_examples = X.shape[0]
        perm = np.arange(self._num_examples)
        if (shuffle):
            np.random.shuffle(perm)
        self._X = X[perm,:]
        self._Y = Y[perm,:]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._Din = X.shape[1]
        self._Dout = Y.shape[1]

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if (self._index_in_epoch > self._num_examples) and (start != self._num_examples):
            self._index_in_epoch = self._num_examples
        if self._index_in_epoch > self._num_examples:   # Finished epoch
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)                  # Shuffle the data
            self._X = self._X[perm,:]
            self._Y = self._Y[perm,:]
            start = 0                               # Start next epoch
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._X[start:end,:], self._Y[start:end,:]

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def Din(self):
        return self._Din

    @property
    def Dout(self):
        return self._Dout

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y



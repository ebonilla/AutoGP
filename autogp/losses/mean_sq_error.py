from __future__ import absolute_import
import numpy as np
from . import loss


class RootMeanSqError(loss.Loss):
    def __init__(self, dout):
        loss.Loss.__init__(self,dout)

    def eval(self, ytrue, ypred):
        error_rate = np.sqrt(np.mean(np.square(ytrue - ypred)))
        return error_rate

    def get_name(self):
        return "RMSE"

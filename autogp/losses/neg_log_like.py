from __future__ import absolute_import
import numpy as np
from . import loss

class NegLogLikelihood(loss.Loss):
    def __init__(self, dout):
        loss.Loss.__init__(self,dout)

    def eval(self, ytrue, ypred):
    	self.like.log_cond_prob(ytue, ypred)
        return error_rate

    def get_name(self):
        return "Negative Log Likelihood"

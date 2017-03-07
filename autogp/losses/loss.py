class Loss(object):

    def __init__(self, dout):
        self.dout = dout

    def eval(self, _ytrue, _ypred):
        """
        Subclass should implement log p(Y | F)
        :param output:  (batch_size x Dout) matrix containing true outputs
        :param latent_val: (MC x batch_size x Q) matrix of latent function values, usually Q=F
        :return:
        """
        raise NotImplementedError("Subclass should implement this.")

    def get_name(self):
        raise NotImplementedError("Subclass should implement this.")

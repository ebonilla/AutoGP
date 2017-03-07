import abc


class Likelihood:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_cond_prob(self, outputs, latent):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def get_params(self):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def predict(self, latent_means, latent_vars):
        raise NotImplementedError("Subclass should implement this.")


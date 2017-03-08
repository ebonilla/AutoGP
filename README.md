# AutoGP: Automated Variational Inference for Gaussian Process Models
An implementation of the model described in [AutoGP: Exploring the Capabilities and Limitations of Gaussian Process Models](https://arxiv.org/abs/1610.05392).

The code was tested on Python 2.7 and [TensorFlow 0.12](https://www.tensorflow.org/get_started/os_setup).

# Installation
You can download and install AutoGP using:
```
git clone git@github.com:ebonilla/AutoGP.git
cd AutoGP
python setup.py
```
# Usage 
There is a very simple example script showing how to use AutoGP in the `example.py`. 
# Acknowledgements
The code to support triangular matrices operations under `autogp/util/tf_ops` was taken from the GPflow repository (Hensman, Matthews et al. GPflow, http://github.com/GPflow/GPflow, 2016).


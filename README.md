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
The script `example.py`shows a very simple example on how to use AutoGP with the default settings. The main components are:

* Create a Likelihood object 
```
likelihood = autogp.likelihoods.Gaussian()
```
* Create a Kernel object
```
kernel = [autogp.kernels.RadialBasis(1)]
```
* Initialize inducing inputs
```
inducing_inputs = xtrain
```
* Create a new GaussianProcess object
```
model = autogp.GaussianProcess(likelihood, kernel, inducing_inputs)
```
* Select optimizer and train the model
```
optimizer = tf.train.RMSPropOptimizer(0.005)
model.fit(data, optimizer, loo_steps=10, var_steps=20, epochs=30)
```
Where we have selected to train a model using 10 Leave-One-Out optimization stept; 20 variational steps; and a total of 30 global iterations.
* Make predictions on unseen data
```
ypred, _ = model.predict(xtest)
```

# Acknowledgements
The code to support triangular matrices operations under `autogp/util/tf_ops` was taken from the GPflow repository (Hensman, Matthews et al. GPflow, http://github.com/GPflow/GPflow, 2016).


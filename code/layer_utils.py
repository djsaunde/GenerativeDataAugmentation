from layers import *
from fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that performs an affine transformation, normalization, then a ReLU
  """
  
  cache = dict()
  # affine transformation
  activations, cache['activations'] = affine_forward(x, w, b)
  # normalized to zero mean, unit variance
  normed_activations, cache['normed_activations'] = batchnorm_forward(activations, gamma, beta, bn_param)
  # applying ReLU nonlinearity
  relu_output, cache['relu_output'] = relu_forward(normed_activations)
  # return output, cached values
  return relu_output, cache


def affine_norm_relu_backward(dout, cache):
  """
  Backward pass for the affine-normalization-relu convenience layer
  """

  # gradient of ReLU layer WRT output
  drelu_output = relu_backward(dout, cache['relu_output'])
  # gradient of batch normalization layer (gives gradients for scale and shift parameters, too)
  dnormed_activations, dgamma, dbeta = batchnorm_backward(drelu_output, cache['normed_activations'])
  # gradient of input, weights, and biases given from affine layer activations
  dx, dw, db = affine_backward(dnormed_activations, cache['activations'])
  # returning calculated gradients
  return dx, dw, db, dgamma, dbeta



def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_norm_relu_pool_forward(x, w, b, gamma, beta, bn_param, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, spatial batchnorm, ReLU, and a max-pool.
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  normed_a, norm_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(normed_a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, norm_cache, relu_cache, pool_cache)
  return out, cache

def conv_norm_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, norm_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dnormed_a, dgamma, dbeta = spatial_batchnorm_backward(da, norm_cache)
  dx, dw, db = conv_backward_fast(dnormed_a, conv_cache)
  return dx, dw, db, dgamma, dbeta

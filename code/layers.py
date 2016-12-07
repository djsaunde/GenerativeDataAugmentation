from __future__ import division

import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None

  # store original shape of input data
  orig_shape = x.shape

  # reshaping the data into a matrix of dimension (N, D), where D = d_1 * ... * k_k
  x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
  
  # to calculate the output, we compute the dot product of the input with the layer's weights and apply the translation b
  out = x.dot(w) + b

  # reshape x for use later in the cache
  x = x.reshape(orig_shape)

  # store cache variables
  cache = (x, w, b)

  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dx = np.dot(dout, w.T).reshape(x.shape)  
  x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
  dw = np.dot(x.T, dout)
  db = np.sum(dout, axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(x, 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout * np.asarray(x >= 0, dtype='int32') # the derivative with respect to upstream derivatives is 1 if x > 0, and 0 otherwise
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    cache = dict()
    # storing the input
    cache['x'] = x
    # calculating mean and storing it
    mean_x = np.mean(x, axis=0)
    cache['mean'] = mean_x
    # calculating variance and storing it
    var_x = np.var(x, axis=0)
    cache['var'] = var_x
    # calculating the normalized inputs as in reference [3] and storing relevant variables
    x_hat = (x - mean_x) / np.sqrt(var_x + eps)
    cache['x_mu'] = x - mean_x
    cache['sqrtvar'] = np.sqrt(var_x + eps)
    cache['invar'] = 1.0 / np.sqrt(var_x + eps) 
    cache['x_hat'], cache['eps'] = x_hat, eps
    # scaling and shifting normalized data using gamma, beta
    out = gamma * x_hat + beta
    cache['gamma'], cache['beta'] = gamma, beta
    # calculating running mean and variance using the momentum parameter as in the class notes
    running_mean = momentum * running_mean + (1 - momentum) * mean_x
    running_var = momentum * running_var + (1 - momentum) * var_x
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    batch_norm = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * batch_norm + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  # getting gradients of beta, gamma * x_hat WRT output
  dbeta = np.sum(dout, axis = 0)
  dgammax_hat = dout
  # getting gradients of gamma, x_hat WRT gamma * x_hat
  dgamma = np.sum(dgammax_hat * cache['x_hat'], axis = 0)
  dx_hat = dgammax_hat * cache['gamma']
  # getting gradients of the inverse of the square root of the variance, and x_mu = x - mean_x WRT the inverse of the variance
  dinvar = np.sum(dx_hat * cache['x_mu'], axis = 0)
  dx_mu1 = dx_hat * cache['invar']
  # getting gradient of the square root of the variance
  dsqrtvar = dinvar * ( - float(1) / np.square(cache['sqrtvar']) )
  # getting gradient of variance
  dvar = dsqrtvar * 0.5 * ( float(1) / np.sqrt(cache['var'] + cache['eps']))
  # getting gradient of the square of x_mu
  dsqr = dvar * (float(1) / dout.shape[0]) * np.ones(dout.shape)
  # getting gradient of x_mu WRT the square of x_mu
  dx_mu2 = dsqr * 2 * cache['x_mu']
  # getting gradient of input WRT x_mu
  dx1 = dx_mu1 + dx_mu2
  # getting gradient of mu WRT x_mu
  dmu = -np.sum(dx_mu1 + dx_mu2, axis = 0)
  # getting gradient of input WRT mu
  dx2 = dmu * (float(1) / dout.shape[0]) * np.ones(dout.shape)
  # getting total gradint of input
  dx = dx1 + dx2
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  x_mu, inv_var, x_hat, gamma = cache['x_mu'], cache['invar'], cache['x_hat'], cache['gamma']

  dx_hat = dout * cache['gamma']

  dx = (float(1) / dout.shape[0]) * cache['invar'] * (dout.shape[0] * dx_hat - np.sum(dx_hat, axis=0) - cache['x_hat'] * np.sum(dx_hat * cache['x_hat'], axis=0))

  dbeta = np.sum(dout, axis=0)
  
  dgamma = np.sum(cache['x_hat'] * dout, axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) > p) / (1 - p) # first dropout mask
    out = x * mask # drop
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = mask * dout # drop
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  # unpacking convolution parameters
  stride = conv_param['stride']
  pad = conv_param['pad']
  # unpacking shape of input for use in logic
  N, C, H, W = x.shape
  # unpacking shape of weights for use in logic
  F, _, HH, WW = w.shape
  # calculating output volume
  H_prime = 1 + (H + 2 * pad - HH) / stride
  W_prime = 1 + (W + 2 * pad - WW) / stride
  # instantiating output volume
  out = np.zeros((N, F, H_prime, W_prime))
  # padding input data
  padded_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
  # reshaping the weight matrix
  reshaped_w = np.reshape(w, (F, C * HH * WW))
  # looping through new output volume and doing the convolution operation
  for i in xrange(H_prime):
      for j in xrange(W_prime):
          # getting the spatial region we care about (as a function of stride and reshaped weight matrix size)
          filter_input = padded_x[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
          # reshape the filter's input to be a collection of column vectors with the same shaped as the reshaped weight matrix
          filter_input_reshaped = np.reshape(filter_input, (N, C * HH * WW))
          # computing the dot product of the reshaped filter input with the reshaped weights and adding the bias vector
          out[:, :, i, j] = np.dot(filter_input_reshaped, reshaped_w.T) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  # getting stored parameters in cache
  x, w, b, conv_param = cache
  pad = conv_param['pad']
  stride = conv_param['stride']
  # getting shape of input, weights, and upstream derivatives
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  N, F, H_prime, W_prime = dout.shape
  # padding the input as per the padding parameter
  padded_dx = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
  padded_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
  # reshaping the weight matrix and derivative of weight matrix as per the class notes
  reshaped_w = np.reshape(w, (F, C * HH * WW))
  reshaped_dw = np.zeros((F, C * HH * WW))
  # looping through each row in the upstream derivatives
  for i in xrange(H_prime):
      for j in xrange(W_prime):
          # getting the upstream derivative at this point multiplied by the weight matrix
          dinput =  dout[:, :, i, j].dot(reshaped_w)
          # reshaping this as per the class notes
          reshaped_dinput = dinput.reshape(N, C, HH, WW)
          # adding this to the reshaped input derivatives 
          padded_dx[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += reshaped_dinput
          # reshaping the input matrix 
          reshaped_input = padded_x[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW].reshape(N, C * HH * WW)
          # reshaping the weight matrix
          reshaped_dw += dout[:, :, i, j].T.dot(reshaped_input)
          # getting the derivative of the input without the padded cells
          dx = padded_dx[:, :, pad:-pad, pad:-pad]
          # getting the derivative of the weight matrix (reshaping)
          dw = reshaped_dw.reshape((F, C, HH, WW))
          # getting the derivative of the bias
          db = np.sum(dout, axis=(0,2,3))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  # unpacking pooling parameters
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  # storing dimensions of input matrix
  N, C, H, W = x.shape
  # calculating dimensions of output volume
  H_prime = (H - pool_height) / stride + 1
  W_prime = (W - pool_width) / stride + 1
  # instantiating the output volume
  out = np.zeros((N, C, H_prime, W_prime))
  # looping through each cell in the output volume and calculating the maxpool operation
  for i in xrange(H_prime):
      for j in xrange(W_prime):
          out[:, :, i, j] = x[:, :, i * stride:i * stride + pool_height, j*stride:j*stride + pool_width].max(axis=(2,3))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  # getting stored cache variables
  x, pool_param = cache
  # unpacking pooling parameters
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  # getting the shape of the upstream derivatives
  N, C, H_prime, W_prime = dout.shape
  # getting the shape of the input matrix
  _, _, H, W = x.shape
  # instantiating matrix of derivatives with respect to the input x
  dx = np.zeros((N, C, H, W))
  # looping through the entries of the maxpooled matrix to compute derivatives for the backward pass
  for i in xrange(H_prime):
      for j in xrange(W_prime):
          # reshape input matrix in order to get maximum indices
          reshaped_x = x[:, :, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width].reshape(N * C, pool_height * pool_width)
          # find indices of maximum entries in the input matrix x
          indices = reshaped_x.argmax(axis=1)
          # reshape the input derivatives matrix to match
          reshaped_dx = np.zeros((N * C, pool_height * pool_width))
          # calculate the derivative based on the maximum indices
          reshaped_dx[range(N * C), indices] = dout[:, :, i, j].reshape(N * C)
          # transfer this to the correctly-shaped matrix of input derivatives
          dx[:, :, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width] = reshaped_dx.reshape(N, C, pool_height, pool_width)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  # get dimensions of input matrix
  N, C, H, W = x.shape
  # rotating / reshaping input x as above for convolutional networks
  reshaped_x = x.transpose((0, 2, 3, 1)).reshape(N * H * W, C)
  # calculating the output (with the above shape) using the vanilla version of batchnorm
  out, cache = batchnorm_forward(reshaped_x, gamma, beta, bn_param)
  # reshaping / rotating the output back to the original shape of the input
  out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  # get dimensions of upstream derivatives
  N, C, H, W = dout.shape
  # rotating / reshaping upstream derivatives as above for convolutional networks
  reshaped_dout = dout.transpose((0, 2, 3, 1)).reshape(N * H * W, C)
  # calculating the derivatives (with the above shape) using the vanilla version of batchnorm
  dx, dgamma, dbeta = batchnorm_backward_alt(reshaped_dout, cache)
  # reshaping / rotating the input derivatives back to the original shape of the input
  dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
  

def inverse_softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  probs = 1 - probs
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
  
  
def correlation_loss(X, transform):
  """
  Computes the loss associated with the degree of correlation between an image
  and its transformation.
  
  Inputs:
  - X: Input data, in the form of a minibatch of training samples
  - transform: the output of the transformer network
  
  Returns:
  - the correlation loss as defined by trace(X * transform) / | X * transform |
  """
  
  correlation = np.sum(np.linalg.norm(X) * np.linalg.norm(np.linalg.norm(transform)) / np.dot(X, transform.T), axis=1)
  
  print correlation.shape
  
  loss = np.sum(correlation) / X.shape[0]
  dx = correlation.copy()[np.arange(X.shape[0])] / X.shape[0]
  
  return loss, dx
  


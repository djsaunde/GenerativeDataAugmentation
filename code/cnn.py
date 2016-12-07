import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


class ConvolutionalNetwork(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7, use_batchnorm=True,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    # getting input dimensions
    C, H, W = input_dim

    # calculate size of padding
    pad = (filter_size - 1) / 2

    # calculating shape of hidden layer after convolution
    H2 = (1 + H + 2 * pad - filter_size) / 2
    W2 = (1 + W + 2 * pad - filter_size) / 2

    # initializing weights, biases
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.normal(0, weight_scale, (H2 * W2 * num_filters, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)

    # initializing gammas, betas
    self.params['gamma1'] = np.ones((num_filters))
    self.params['beta1'] = np.zeros((num_filters))
    self.params['gamma2'] = np.ones((hidden_dim))
    self.params['beta2'] = np.zeros((hidden_dim))
    
    # setting batchnorm parameters
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(2)]
    
    # setting correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    
    ################
    # FORWARD PASS #
    ################
    
    # setting the network operation mode
    mode = 'test' if y is None else 'train'
    
    # getting network parameters
    W1, b1, gamma1, beta1 = self.params['W1'], self.params['b1'], self.params['gamma1'], self.params['beta1']
    W2, b2, gamma2, beta2 = self.params['W2'], self.params['b2'], self.params['gamma2'], self.params['beta2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    # are we using batchnorm? if so...
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    # instantiating batchnorm and scores parameters
    bn_params = self.bn_params
    scores = None
    
    # creating a cache data structure for use in the backward pass
    self.cache = {}
    
    # computing the conv -> relu -> max-pool sandwich layer
    h1, self.cache['h1_cache'] = conv_norm_relu_pool_forward(X, W1, b1, gamma1, beta1, bn_params[0], conv_param, pool_param)
    
    # computing the affine -> relu sandwich layer
    h2, self.cache['h2_cache'] = affine_norm_relu_forward(h1, W2, b2, gamma2, beta2, bn_params[1])
    
    # computing the affine final layer 
    scores, self.cache['scores_cache'] = affine_forward(h2, W3, b3)

    
    # if we're at test time, return the raw scores
    if y is None:
      return scores


    #################
    # BACKWARD PASS #
    #################

    # instantiating loss and gradient parameters
    loss, grads = 0, {}

    # calculating the loss from the softmax function and the derivative of the output with respect to the loss value
    loss, dout = softmax_loss(scores, y)
    
    # calculating derivatives of affine final layer (upstream derivative dout) with respect to h4, W5, and b5
    dh2, dW3, grads['b3'] = affine_backward(dout, self.cache['scores_cache'])
    
    # calculating derivatives of affine -> relu sandwich layer (upstream derivative dh2) with respect to h3, W4, and b4
    dh1, dW2, grads['b2'], grads['gamma2'], grads['beta2'] = affine_norm_relu_backward(dh2, self.cache['h2_cache'])
    
    # calculating derivatives of conv -> relu -> max-pool layer (upstream derivative dh1) with respect to x, W3, and b3
    dx, dW1, grads['b1'], grads['gamma1'], grads['beta1'] = conv_norm_relu_pool_backward(dh1, self.cache['h1_cache'])
    
    # calculating L2 regularization for each layer
    W1_reg_loss = 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1'])
    W2_reg_loss = 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2'])
    W3_reg_loss = 0.5 * self.reg * np.sum(self.params['W3'] * self.params['W3'])
    
    # adding L2 regularization to loss
    loss += W1_reg_loss + W2_reg_loss + W3_reg_loss
    
    # adding L2 regularization to weight gradients
    grads['W1'] = dW1 + dW1 * self.reg
    grads['W2'] = dW2 + dW2 * self.reg
    grads['W3'] = dW3 + dW3 * self.reg
    
    # adding L2 regularization to input gradient
    grads['dx'] = dx + dx * self.reg

    return loss, grads, scores
    

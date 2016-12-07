from __future__ import division
import numpy as np


"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
  """
  Performs vanilla stochastic gradient descent.

  config format:
  - learning_rate: Scalar learning rate.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)

  w -= config['learning_rate'] * dw
  return w, config


def sgd_momentum(w, dw, config=None):
  """
  Performs stochastic gradient descent with momentum.

  config format:
  - learning_rate: Scalar learning rate.
  - momentum: Scalar between 0 and 1 giving the momentum value.
    Setting momentum = 0 reduces to sgd.
  - velocity: A numpy array of the same shape as w and dw used to store a moving
    average of the gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)
  v = config.get('velocity', np.zeros_like(w))
  
  next_w = None
  #############################################################################
  # TODO: Implement the momentum update formula. Store the updated value in   #
  # the next_w variable. You should also use and update the velocity v.       #
  #############################################################################
  mu = config['momentum']
  learning_rate = config['learning_rate']
  # Momentum update
  v = mu * v - learning_rate * dw # integrate velocity
  next_w = w + v # integrate position
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  config['velocity'] = v

  return next_w, config



def rmsprop(x, dx, config=None):
  """
  Uses the RMSProp update rule, which uses a moving average of squared gradient
  values to set adaptive per-parameter learning rates.

  config format:
  - learning_rate: Scalar learning rate.
  - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
    gradient cache.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - cache: Moving average of second moments of gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('decay_rate', 0.99)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('cache', np.zeros_like(x))

  next_x = None
  #############################################################################
  # TODO: Implement the RMSprop update formula, storing the next value of x   #
  # in the next_x variable. Don't forget to update cache value stored in      #  
  # config['cache'].                                                          #
  #############################################################################
  config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dx**2
  next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']) + config['epsilon'])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return next_x, config


def adam(x, dx, config=None):
  """
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.

  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(x))
  config.setdefault('v', np.zeros_like(x))
  config.setdefault('t', 0)
  
  next_x = None

  # update time step parameter t
  config['t'] += 1

  # update biased first moment estimate
  config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx

  # update biased second moment estimate
  config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dx ** 2)

  # applying bias corrections based on time step
  m_corrected = config['m'] / (1.0 - (config['beta1'] ** config['t']))
  v_corrected = config['v'] / (1.0 - (config['beta2'] ** config['t']))
    
  # computing the updates to the weights based on the bias-corrected first and second moment estimates
  next_x = x - (config['learning_rate'] * m_corrected) / (np.sqrt(v_corrected) + config['epsilon'])
  
  return next_x, config


def adagrad(x, dx, config=None):
    """
    Uses the Adagrad update rule, which adapts the learning rate to the parameters, performing
    larger updates to infrequent and smaller updates to frequent parameters. Adagrad uses a different
    learning rate for each parameter at each time step, making it a local method in time and parameter
    space. 
    
    config_format:
        learning_rate: original learning rate for all parameters
        epsilon: smoothing parameter to ensure numerical stability
        historical_grad: keeping a record of previous gradient updates
        t: iteration number
    """
    if config is None: config = {}
        
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('t', 0)
    config.setdefault('historical_grad', np.zeros(dx.shape))
    
    next_x = None
    # incrementing the iteration counter
    config['t'] += 1
    # updating the historical gradient
    config['historical_grad'] += np.square(dx)
    # applying the Adagrad update rule
    next_x = x - (config['learning_rate'] * dx / np.sqrt(config['historical_grad'] + config['epsilon']))
    # return gradients and updated config parameters
    return next_x, config

def adadelta(x, dx, config=None):
    """
    Uses the Adadelta update rule, which, instead of accumulating all past squared gradients (as in Adagrad), this
    method only keeps track of a moving average of historical gradients, for some fixed window of length w. The sum
    of the gradients is recursively defined as a decaying average of all past squared gradients.
    
    config_format:
        learning_rate: original learning rate for all parameters
        epsilon: smoothing parameter to ensure numerical stability
        historical_grad: keeping a record of previous gradient updates
        t: iteration number
        w: window length
    """
    if config is None: config = {}
        
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('t', 0)
    config.setdefault('historical_grad', np.zeros(dx.shape))
    config.setdefault('w', )
    config.setdefault('gamma', 0.9)
    
    next_x = None
    # incrementing the iteration counter
    config['t'] += 1
    # updating the historical gradient
    config['historical_grad'] = config['gamma'] * np.square(dx) + (1 - config['gamma']) * config['historical_grad']
    # applying the Adadelta udpate rule
    next_x = x - (config['learning_rate'] * dx / np.sqrt(config['historical_grad'] + config['epsilon']))
    # return gradients and updated configuration parameters
    return next_x, config
  
  


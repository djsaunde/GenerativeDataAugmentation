import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *
from util import *


class Transformer(object):
    '''
    A multilayer perceptron whose purpose is to transform input images by applying
    a nonlinear transformation. We constrain the output by imposing a penalty for
    correlation between it and the original input image.
    '''
    
    def __init__(self, input_dim=32*32*3, hidden_dim=1000, use_batchnorm=True, 
                    weight_scale=1e-3, reg=0.0, dtype=np.float32):
        '''
        Initialize the transformer network as a multilayer perceptron from the input
        space back into it. The hidden dimension, or "code" dimension, is typically
        smaller than the input dimension, so we are using a trick similar to autoencoders,
        but instead of reconstructing the input, we try to apply a useful transformation
        to it.
        
        The structure of this network is: affine - relu - affine.
        
        Note that there is no softmax 
        
        input:
            input_dim: the number of input nodes to the network (CIFAR-10 size by default)
            hidden_dim: the number of nodes in the hidden layer of the netowork
            use_batchnorm: whether or not to use batch normalization
            weight_scale: the scale about which to initialize model parameters
            reg: hyperparameter penalizing the magnitude of model parameters
            dtype: data type of the input to the network
        '''
        
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # initializing weights, biases
        self.params['W1'] = np.random.normal(loc=0, scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(loc=0, scale=weight_scale, size=(hidden_dim, input_dim))
        self.params['b2'] = np.zeros(input_dim)

        # initializing gammas, betas
        self.params['gamma1'] = np.ones((hidden_dim))
        self.params['beta1'] = np.zeros((hidden_dim))
        
        # setting batchnorm parameters
        self.bn_params = []
        if self.use_batchnorm:
          self.bn_params = [{'mode': 'train'} for i in xrange(1)]
        
        # setting correct datatype
        for k, v in self.params.iteritems():
          self.params[k] = v.astype(dtype)
          
          
    def forward(self, X):
        '''
        Forward pass of the network. We are simply passing the input through an 
        affine - relu - affine structure.
        
        input:
            X: minibatch array of input data, of shape (N, d_1, ..., d_k)
            
        output:
            transformed minibatch input data
        '''
        
        # setting the network operation mode
        mode = 'test'
        
        # initializing cache dictionary
        self.cache = {}
        
        # adding input minibatch array to cache
        self.cache['X'] = X
        
        # are we using batchnorm? if so...
        if self.use_batchnorm:
          for bn_param in self.bn_params:
            bn_param[mode] = mode
        
        # compute first layer's hidden unit activations and apply nonlinearity (thanks to layer_utils module)
        hidden, self.cache['hidden'] = affine_norm_relu_forward(X, self.params['W1'], self.params['b1'], self.params['gamma1'], self.params['beta1'], self.bn_params[0])
        
        # compute second layer's hidden unit activations
        transform, self.cache['output'] = affine_forward(hidden, self.params['W2'], self.params['b2'])

        # we returned the transformed input
        return transform

    
            
    def backward(self, transform, loss, dtransform):
        '''
        Computing loss and gradient for a minibatch of data. This will simply be
        the loss and gradient passed back from the discriminator network, combined 
        with the correlation loss we define between the input and output of this network.
        
        input:
            transform: the output of the network from the forward pass
            loss: the loss output from the discriminator network (we want to maximize the log of this!)
            dtransform: the gradient of the tranform output from the forward pass
            y: Array of labels, of shape (N,). y[i] gives the label for X[i]
        
        output:
            If y is None, then run a test-time forward pass of the fully-connected
            network, and output the transformed image.

            If y is not None, then run a training-time forward and backward pass and
            return a tuple of:
                Scalar value giving the loss
                Dictionary with the same keys as self.params, mapping parameter
                names to gradients of the loss with respect to those parameters.
        '''
        
        # initializing variables for loss and gradients (we try to maximize the log 
        # loss of the discriminator model as in the Goodfellow paper)
        loss, grads = -np.log(loss), {}
        
        # adding correlation loss between the original input and the transformed input
        X = self.cache['X']
        X = X.reshape(transform.shape)
        
        # TODO: figure out how to incorporate the correlation loss...
        # loss += correlation_loss(X, transform)
        
        # adding regularization loss to the total loss
        # this is calculated by summing over all squared weights of the network
        loss += 0.5 * (self.reg * (np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2']))))
        
        # computing the gradient of the weights of each layer with respect to the upstream gradient of the class predictions
        dhidden, grads['W2'], grads['b2'] = affine_backward(dtransform, self.cache['output'])
        dinput, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = affine_norm_relu_backward(dhidden, self.cache['hidden'])
        
        # adding regularization to the weight gradients (using its derivative w/ respect to the weights)
        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']
        
        # returning the calculated loss and gradients
        return loss, grads
        
        
        
        

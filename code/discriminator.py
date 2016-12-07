from __future__ import division

import numpy as np

from layers import *
from layer_utils import *
from cnn import *

import pickle as pkl
import os


class Discriminator(ConvolutionalNetwork):
    '''
    This class encapsulates all the logic involved with the discriminator
    network. 
    
    Basically, the discrimator behaves like a normal classifier, except
    that it makes an additional prediction for whether or not the input comes
    from the training dataset distribution, or from the transformer network. 
    
    The discriminator tries to maximize the probability that it gets this distinction
    right, jointly with minimizing the classification error. We can, in fact, think
    of this together as minimizing a "more inclusize" classification error.
    
    In this implementation, we initialize the network with a ConvNet.
    '''
    
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=5,
               hidden_dim=500, num_classes=11, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
        '''
        To initialize the discriminator network. We simply call the superclass
        constructor with the passed-in arguments. Note that there are 11 classes,
        1 additional class for the images given by the transformer network.
        
        input:
            input_dim: number of input nodes to the network (size of CIFAR-10 images)
            num_filters: the number of feature maps to use in convolutional layers
            filter_size: size of a single feature map
            hidden_dim: the number of nodes in the hidden layer of the network (assuming
                a 3-layer network
            num_classes: number of ways classify an image
            weight_scale: the magnitude at which to initialize network parameters about
            reg: hyperparameter penalizing magnitude of network parameters
            dtype: the data type of the input images
        '''
            
        # call the superclass constructor with passed-in parameters
        super(Discriminator, self).__init__(input_dim=input_dim, num_filters=num_filters, 
                filter_size=filter_size,use_batchnorm=True, hidden_dim=hidden_dim, 
                num_classes=num_classes, weight_scale=weight_scale, reg=reg, dtype=dtype)
                

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the discriminator network.

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
        
        # call the superclass loss function
        return super(Discriminator, self).loss(X, y)


    def transformer_backward(self, transformer_loss, transformer_dout):
        '''
        Simply, this function will return the derivative of the input to the network
        with respect to the inverse loss on the class scores (want to maximize the probability
        that the discriminator mistakes transformed data for real data. 
        '''
        
        grads = {}
        
        # calculating derivatives of affine final layer (upstream derivative dout) with respect to h4, W5, and b5
        dh2, dW3, grads['b3'] = affine_backward(transformer_dout, self.cache['scores_cache'])
        
        # calculating derivatives of affine -> relu sandwich layer (upstream derivative dh2) with respect to h3, W4, and b4
        dh1, dW2, grads['b2'], grads['gamma2'], grads['beta2'] = affine_norm_relu_backward(dh2, self.cache['h2_cache'])
        
        # calculating derivatives of conv -> relu -> max-pool layer (upstream derivative dh1) with respect to x, W3, and b3
        dx, dW1, grads['b1'], grads['gamma1'], grads['beta1'] = conv_norm_relu_pool_backward(dh1, self.cache['h1_cache'])
        
        # calculating L2 regularization for each layer
        W1_reg_loss = 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1'])
        W2_reg_loss = 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2'])
        W3_reg_loss = 0.5 * self.reg * np.sum(self.params['W3'] * self.params['W3'])
        
        # adding L2 regularization to loss
        transformer_loss += W1_reg_loss + W2_reg_loss + W3_reg_loss
        
        # adding L2 regularization to input gradient
        dx = dx + dx * self.reg

        return dx, transformer_loss

                
        

from __future__ import division

import numpy as np

from layers import *
from layer_utils import *
from dnn import *


class Generator(Transformer):
    '''
    This class encapsulates all the logic involved with the generator
    network.
    
    The generator tries to maximize the probability that the discriminator classifies
    the input generated images as true training dataset images.
    
    In this implementation, we initialize the network with a transformer network.
    '''
    
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=5,
               hidden_dim=500, num_classes=11, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    
    super(Transformer, self).__init__()

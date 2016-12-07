'''
Some helper methods for use in the main script and perhaps elsewhere.
'''

import numpy as np

def sample_training_data(X_train, y_train, N):
    '''
    Returns N randomly sampled data points.
    '''
    
    # randomly sample N points in the range of the training data
    idxs = np.random.choice(len(X_train), N)
    # return the data points at those indexes
    return np.asarray([X_train[idx] for idx in idxs]), np.asarray([y_train[idx] for idx in idxs])
    

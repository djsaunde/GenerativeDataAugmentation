

class AdversarialPair(object):
    '''
    This class will contain logic for the transformer, discriminator network
    pair, and contain a composite loss function and parameters so that we may
    use the Solver API to which we pass this pair.
    '''
    
    def __init__(self, transformer=Transformer(), discriminator=Discriminator()):
        '''
        Initializes the pair of adversarial networks.
        
        input:
            transformer: a Transformer object
            discriminator: a Discriminator object
            
        output:
            initialized adversarial pair
        '''
        
        # store parameters from both networks
        self.params = transformer.params.update(discriminator.params)
        
        
       
    def loss(self, X, y=None):
        '''
        Computes loss and gradient on a minibatch of data. We are interested
        in the loss of the disciminator on the data vs. the transformed data,
        and the correlation loss of the transformer.
        
        
        '''
        
    
    def get_params(self):
        '''
        Getter method for the parameters of both the transformer and discriminator.
        '''
        
        return self.transformer.get_params() + self.discriminator.get_params()

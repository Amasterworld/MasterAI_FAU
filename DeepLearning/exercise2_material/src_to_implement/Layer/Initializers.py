import numpy as np
class Constant:
    
    def __init__(self, weight_value = 0.1) -> None:
        self.weight_value = weight_value
        

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.weight_value)

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(low=0, high=1, size=weights_shape)

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        limit = np.sqrt(2 / (fan_in + fan_out))
        
        return np.random.normal(loc=0, scale=limit, size=weights_shape)
    
class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        limit = np.sqrt(2 /fan_in)
        
        # note: to return zero mean Gaussian we need to use random,normal not random.uniform 
        return np.random.normal(loc= 0, scale=limit, size=weights_shape)        
        
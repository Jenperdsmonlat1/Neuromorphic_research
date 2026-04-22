import numpy as np


class LIFSequential:

    def __init__(self, layers):

        self.layers = layers
        self.optimizer = None
    
    def __call__(self, x):

        out = x
        for layer in self.layers:
            out = layer.forward(out)
        
        return out

    def compile(self, optimizer):

        self.optimizer = optimizer

    def fit(self, datas):
        pass

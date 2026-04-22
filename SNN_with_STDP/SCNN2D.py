import numpy as np
from scipy.signal import convolve2d


class CUBALIFConv2DLayer:

    def __init__(self, filters, kernel_size, stride, padding):

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, spikes):
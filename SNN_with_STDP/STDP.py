import numpy as np


class STDP:

    def __init__(self, a_plus=-10, a_minus=10, tau_plus=10e-3, tau_minus=10e-3):

        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
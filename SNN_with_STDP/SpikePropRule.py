import numpy as np


class SpikePropOptimizer:

    def __init__(self, learning_rate, firing_rate):

        self.learning_rate = learning_rate
        self.firing_rate = firing_rate

    @staticmethod
    def heavyside(self, t_post, t_pre):

        return 1.0 if t_post > t_pre else 0.0

    def apply_gradient(self, y_pred, y_true, t_pre, t_post):

        delta_weight = self.learning_rate * self.firing_rate * np.subtract(y_pred, y_true) * self.heavyside(t_post, t_pre)
        return delta_weight
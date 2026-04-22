import numpy as np


class RateEncoding:

    def __init__(self, datas, p_max=0.2):

        self.datas = datas
        self.p_max = p_max

    def rate_decode(self, outputs, steps, value):

        values = []
        for i in range(value):
            values.append(outputs[:, i].sum() * np.max(self.datas) / (steps * self.p_max))

        return np.array(values)

    def rate_encode(self, steps):

        p = (self.datas / np.max(self.datas)) * self.p_max
        spikes = np.random.binomial(n=1, p=p, size=(steps, len(self.datas)))
        return spikes
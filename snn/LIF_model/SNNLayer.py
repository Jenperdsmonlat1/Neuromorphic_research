import numpy as np



class SNNLayer:
    def __init__(self, n_in, n_out, dt, gamma, tau_m, V_th, V_reset=0, R=10):
        
        self.n_in = n_in
        self.n_out = n_out
        self.alpha = 1 - dt / tau_m
        self.beta = R * dt / tau_m
        self.V_th = V_th
        self.V_reset = V_reset
        self.I_syn = np.zeros(n_out)
        self.V = np.zeros(n_out)
        self.spikes = np.zeros(n_out)
        self.W = np.random.rand(n_out, n_in)
        self.gamma = gamma

    def __call__(self, S_in):
        return self.step(S_in)

    def reset(self):

        self.I_syn[:] = 0
        self.V[:] = 0
        self.spikes[:] = 0

    def step(self, S_in):

        self.spikes[:] = 0

        if S_in.shape[0] != self.n_in:
            return None
        else:
            self.I_syn = self.gamma * self.I_syn + np.dot(self.W, S_in)
            self.V = self.alpha * self.V + self.beta * self.I_syn
            for i in range(len(self.V)):
                if self.V[i] >= self.V_th:
                    self.spikes[i] = 1
                    self.V[i] = self.V_reset
        
            return self.spikes

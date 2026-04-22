import numpy as np


class LIF:
    def __init__(self, dt, E_L=-70e-3, C_m=250e-12, tau_m=10e-3, t_ref=2e-3, V_th=-55e-3, V_reset=-70e-3, tau_syn=2e-3, I_e=0, V_min=-float('inf')):

        self.E_L = E_L
        self.C_m = C_m
        self.tau_m = tau_m
        self.t_ref = t_ref
        self.V_th = V_th
        self.V_reset = V_reset
        self.tau_syn = tau_syn
        self.I_e = I_e
        self.V_min = V_min
        self.dt = dt
        self.n = 0
        self.V_trace = []
        self.spikes = []

        self.V_trace.append(self.V_reset)

    def i_syn(self, t, t_spike):

        return ((np.e * (t - t_spike)) / self.tau_syn) * np.exp(- (t - t_spike) / self.tau_syn) if (t - t_spike) >= 0 else 0

    def compute_I_syn(self, t, synapses):

        I = 0.0

        for weight, spikes in synapses:
            for t_k in spikes:
                I += (weight * self.i_syn(t, t_k)) * 10e-12

        return I

    def steps(self, t, synapse):

        I_syn = self.compute_I_syn(t, synapse)
        self.V_trace.append(self.V_trace[self.n] * (1 - self.dt / self.tau_m) + (self.dt / self.tau_m) * self.E_L + (I_syn / self.C_m) * self.dt)

        if self.V_trace[self.n + 1] >= self.V_th:

            self.spikes.append(t)
            self.V_trace[-1] = self.V_reset

        self.n = self.n + 1

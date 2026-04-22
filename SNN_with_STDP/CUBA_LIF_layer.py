import numpy as np


class CubaLifLayer:

    def __init__(self, n_neuron_layer, n_neuron_previous_layer, dt=1e-3, tau_syn=10e-3, tau_mem=10e-3, v_thres=1.0, v_reset=0.0):

        self.N_neuron_layer = n_neuron_layer
        self.N_neuron_previous_layer = n_neuron_previous_layer
        self.dt = dt
        self.tau_syn = tau_syn
        self.tau_mem = tau_mem
        self.v_thres = v_thres
        self.v_reset = v_reset
        self.I = np.zeros((self.N_neuron_layer, 1))
        self.alpha = np.exp(-self.dt / self.tau_mem)
        self.beta = np.exp(-self.dt / self.tau_syn)
        self.spikes = np.zeros((self.N_neuron_layer, 1))
        self.potentials = np.zeros((self.N_neuron_layer, 1))
        self.weight = np.clip(np.random.rand(self.N_neuron_layer, self.N_neuron_previous_layer), 0, 1)

    def _integrate(self, spikes):

        spikes_in = np.array(spikes).reshape(-1, 1)
        self.I = np.add((self.alpha * self.I), (self.weight @ spikes_in))
    
    def _potential(self):

        self.potentials = (np.add((self.beta * self.potentials), self. I)) * np.subtract(1.0, self.spikes)

    def forward(self, spikes):

        self._integrate(spikes)
        self._potential()

        self.spikes = (self.potentials >= self.v_thres).astype(float)
        return self.spikes

    def reset(self):

        self.spikes = np.zeros((self.N_neuron_layer, 1))
        self.potentials = np.zeros((self.N_neuron_layer, 1))
        self.I = np.zeros((self.N_neuron_layer, 1))
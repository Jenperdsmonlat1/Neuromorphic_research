import numpy as np


class IfLayer:

    def __init__(self, n_neuron_layer, n_neuron_previous_layer):

        self.N_neuron_layer = n_neuron_layer
        self.N_neuron_previous_layer = n_neuron_previous_layer
        self.dt = 1e-3
        self.v_thres = 1.5
        self.v_reset = 0.0
        self.I = np.zeros((self.N_neuron_layer, 1))
        self.spikes = np.zeros((self.N_neuron_layer, 1))
        self.potentials = np.zeros((self.N_neuron_layer, 1))
        self.weight = np.random.rand(self.N_neuron_layer, self.N_neuron_previous_layer) * 0.3

    def _integrate(self, spikes):

        spikes_in = np.array(spikes).reshape(-1, 1)
        self.I = self.weight @ spikes_in

    def _potential(self):
        self.potentials = (np.add(self.potentials, self.I)) * np.subtract(1.0, self.spikes)

    def forward(self, spikes):

        self._integrate(spikes)
        self._potential()

        self.spikes = (self.potentials >= self.v_thres).astype(float)
        return self.spikes
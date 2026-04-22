import numpy as np
from ANN_to_SNN_FPGA.LIF_model import LIF



class LIF_layer:

    def __init__(self, N_in, N_out, dt, E_L=-70e-3, C_m=250e-12, tau_m=10e-3, t_ref=2e-3, V_th=-55e-3, V_reset=-70e-3, tau_syn=2e-3, I_e=0, V_min=-float('inf')):
        
        self.N_in = N_in
        self.N_out = N_out
        self.dt = dt

        self.neurons = [LIF(dt=dt, E_L=E_L, C_m=C_m, tau_m=tau_m, t_ref=t_ref, V_th=V_th, V_reset=V_reset, tau_syn=tau_syn, I_e=I_e, V_min=V_min) for _ in range(N_out)]
        self.W = np.random.randn(N_out, N_in) * 10000

        print(self.W)
    
    def forward(self, t, pre_spike):

        post_spikes = [[] for i in range(self.N_out)]

        for i, neuron in enumerate(self.neurons):
            synapse_i = [(self.W[i, j], pre_spike[j]) for j in range(self.N_in)]
            neuron.steps(t, synapse=synapse_i)
            post_spikes[i] = [s for s in neuron.spikes if t - s < neuron.tau_syn] #neuron.spikes[-1] if neuron.spikes else []

        return post_spikes
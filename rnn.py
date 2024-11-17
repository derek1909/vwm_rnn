import torch
import torch.nn as nn
from config import *

class RNNMemoryModel(nn.Module):
    def __init__(self, max_item_num, num_neurons, tau=1.0, dt=0.1, noise_level=0.0):
        super(RNNMemoryModel, self).__init__()
        self.num_neurons = num_neurons
        self.tau = tau
        self.dt = dt
        self.noise_level = noise_level
        self.W = nn.Parameter(torch.randn(num_neurons, num_neurons) / num_neurons**0.5) 
        self.B = nn.Parameter(torch.randn(num_neurons, max_item_num*2)*10)
        self.F = nn.Parameter(torch.randn(max_item_num*2, num_neurons) / num_neurons**0.5)

    def activation_function(self, x):
        return 400 * (1 + torch.tanh(0.4 * x - 3)) / self.tau

    def forward(self, r, u):
        # RNN dynamics: τ * dr/dt + r = Φ(W * r + B * u)
        r = r.T # (trial,neuron) -> (neuron,trial) 
        u = u.T # (trial,input) -> (input,trial) 
        r_dot = (-r + self.activation_function(self.W @ r + self.B @ u)) / self.tau
        r = r + self.dt * r_dot + self.noise_level * torch.randn_like(r)
        return r.T

def decode(F, r):
    # r = (num_trials, num_neurons)
    # F = (num_neurons, num_neurons)
    u_hat = (F @ r.T).T.view(r.size(0), int(F.shape[0]/2), 2)
    normalized_u_hat = u_hat / torch.norm(u_hat, dim=2, keepdim=True)
    return normalized_u_hat.view(r.size(0), -1)




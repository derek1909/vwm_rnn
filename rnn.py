import torch
import torch.nn as nn
from config import device

class RNNMemoryModel(nn.Module):
    def __init__(self, max_item_num, num_neurons, tau=1.0, dt=0.1, spike_noise_factor=0.0, device='cpu', positive_input=True):
        super(RNNMemoryModel, self).__init__()
        self.num_neurons = num_neurons
        self.tau = tau
        self.dt = dt
        self.spike_noise_factor = spike_noise_factor
        self.batch_first = True  # Required attribute to use FixedPointFinder
        self.device = device
        self.positive_input = positive_input
        
        if self.positive_input:
            self.B = nn.Parameter(torch.abs(torch.randn(num_neurons, max_item_num*3, device=device))*2)
        else:
            self.B = nn.Parameter(torch.randn(num_neurons, max_item_num*2, device=device)*10)
        self.W = nn.Parameter(torch.randn(num_neurons, num_neurons, device=device) / num_neurons**0.5)
        self.F = nn.Parameter(torch.randn(max_item_num*2, num_neurons, device=device) / num_neurons**0.5)


    def to(self, *args, **kwargs):
        # Call the parent class's `to` method to move the model
        super().to(*args, **kwargs)
        # Update the device attribute based on the first argument
        device = torch._C._nn._parse_to(*args, **kwargs)[0]  # Internal parsing of `to` arguments
        if device:
            self.device = device
        return self
    
    def activation_function(self, x):
        """
        Saturation: 400*2/tau = 16Hz
        """
        return 400 * (1 + torch.tanh(0.4 * x - 3)) / self.tau

    def forward(self, u, r0=None):
        """
        Args:
            u: (batch_size, seq_len, input_size) = (trial, steps, 2*max_item_num)
            r0: (num_layers * num_directions, batch_size, hidden_size) = (1, trial, neuron)
                Initial firing rate for the RNN.

        Returns:
            r_output: (batch_size, seq_len, hidden_size) = (trial, steps, neuron)
                All hidden states over time.
            r: (num_layers * num_directions, batch_size, hidden_size) = (1, trial, neuron)
                Final hidden state.
        """
        batch_size, seq_len, _ = u.size()  # Extract dimensions from input
        r0 = r0 if r0 is not None else torch.zeros(1, batch_size, self.num_neurons, device=self.device)
        
        # Initialize the firing rate for all time steps
        r_output = torch.zeros(batch_size, seq_len, self.num_neurons, device=self.device)
        
        # Current firing rate
        r = r0.squeeze(0)  # Shape: (batch_size, neuron)

        # Iterate over time steps
        for t in range(seq_len):
            assert torch.all(r >= 0), "Negative values detected in r!"
            assert torch.all(torch.isfinite(r)), "NaN or Inf detected in r!"

            u_t = u[:, t, :]  # Current input at time step t (batch_size, input)
        
            # # Add Poisson-like noise to the firing rate r
            # observed_r = torch.clamp(
            #     r + self.spike_noise_factor * torch.sqrt(r*1e3/self.tau) * torch.randn_like(r, device=self.device),
            #     min=0)

            # Add Poisson noise to the firing rate
            observed_r = torch.poisson(r * self.dt) / (self.dt) # not 1e3 is to make poisson more smooth

            # RNN dynamics: τ * dr/dt + r = Φ(W * r + B * u)
            r_dot = (-r + self.activation_function(self.W @ observed_r.T + self.B @ u_t.T).T) / self.tau
            
            # Update firing rate with Euler integration
            r = r + self.dt * r_dot

            # Store the firing rate for this time step
            r_output[:, t, :] = r

        # Return all hidden states and the final hidden state
        return r_output, r.unsqueeze(0)

def decode(F, r):
    """
    Decodes the input firing rates (r) into a normalized output representation.

    Args:
        F : Decoding weight matrix of shape (input_size, hidden_size) = (2*max_item_num, num_neurons)
        r : Firing rate matrix of shape (num_trials, num_neurons)

    Returns:
        torch.Tensor: (num_trials, input_size)
    """
    # (num_trials, input_size/2, 2)
    u_hat = (F @ r.T).T.view(r.size(0), -1, 2)
    
    # Normalize the 2D representation of u_hat across the last dimension.
    normalized_u_hat = u_hat / torch.norm(u_hat, dim=2, keepdim=True)
    
    # Reshape normalized_u_hat into a flat representation.
    # (num_trials, input_size)
    return normalized_u_hat.view(r.size(0), -1)
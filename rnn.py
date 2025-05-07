import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
# from jaxtyping import jaxtyped, Float, Int  # Import jaxtyped helpers and type aliases
# from typeguard import typechecked  # For runtime type checking

class RNNMemoryModel(nn.Module):
    def __init__(self, max_item_num, num_neurons, dt=0.1, tau_min=50, tau_max=50, spike_noise_factor=0.0, saturation_firing_rate=60.0,
                 device='cpu', positive_input=True, dales_law=True):
        super(RNNMemoryModel, self).__init__()
        self.num_neurons = num_neurons
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.spike_noise_factor = spike_noise_factor
        self.batch_first = True  # Required attribute to use FixedPointFinder
        self.device = device
        self.positive_input = positive_input
        self.saturation_firing_rate = saturation_firing_rate
        self.dales_law = dales_law

        # To make sure all models have same initialization.
        # torch.manual_seed(39)

        # ---- Dale's law assignment ----
        if self.dales_law:
            excitatory_ratio = 0.5
            num_excitatory = int(num_neurons * excitatory_ratio)
            # Create a vector with +1 for excitatory and -1 for inhibitory neurons.
            dales_sign = torch.cat([torch.ones(num_excitatory), -torch.ones(num_neurons - num_excitatory)]).to(device)
        else:
            # When Dale's law is disabled, use a vector of ones.
            dales_sign = torch.ones(num_neurons, device=device)

        self.register_buffer('dales_sign', dales_sign)

        # ---- Sample tau ----
        # Log-space sampling for tau: (tau_min ~ tau_max) ms
        tau = self.tau_min * torch.exp(torch.rand(num_neurons, device=device) * math.log(self.tau_max / self.tau_min))
        if self.dales_law:
            # Only sort tau when Dale's law is enabled.
            num_excitatory = int(num_neurons * 0.5)
            tau[:num_excitatory], _ = torch.sort(tau[:num_excitatory])
            tau[num_excitatory:], _ = torch.sort(tau[num_excitatory:])
        else:
            tau, _ = torch.sort(tau)
        self.register_buffer('tau', tau)

        # ---- Define input matrix ----
        if self.positive_input:
            std = 0.418
            self.B = nn.Parameter(torch.abs(torch.randn(num_neurons, max_item_num * 3, device=device)) * std)
        else:
            std = 0.418 # This number is not verified for naive input.
            self.B = nn.Parameter(torch.randn(num_neurons, max_item_num * 2, device=device) * std)
        
        # ---- Define non-negative weight matrix ----
        if self.dales_law:
            std = 1 / (num_neurons * 0.682)**0.5
            self.W = nn.Parameter(torch.abs(torch.randn(num_neurons, num_neurons, device=device) * std))
        else:
            self.W = nn.Parameter(torch.randn(num_neurons, num_neurons, device=device) / num_neurons**0.5)

        # ---- Define readout matrix ----
        std = (2 / (num_neurons + max_item_num * 2))**0.5
        self.F = nn.Parameter(torch.randn(max_item_num * 2, num_neurons, device=device) * std)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device:
            self.device = device
        return self

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Activation function.
        Args:
            x: (num_neurons, batch_size)
        Returns:
            Tensor with same shape as x.
        """
        return self.saturation_firing_rate/2 * (1 + torch.tanh(0.14 * x - 4.2))
    
    def observed_r(self, r: torch.Tensor) -> torch.Tensor:
        """
        Applies poisson-like noise to the firing rate and ensures non-negative values.

        Args:
            r: (batch_size, neuron) - firing rate.

        Returns:
            Tensor with same shape as r, after adding noise and applying ReLU.
        """
        poisson_like_noise = self.spike_noise_factor * torch.sqrt(r * 1e3 / self.dt + 1e-10) * torch.randn_like(r, device=self.device)
        return F.relu(r + poisson_like_noise)
    
    def forward(self, u: torch.Tensor, r0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            u: (batch_size, seq_len, input_size) = (trial, steps, 3*max_item_num)
            r0: (num_layers * num_directions, batch_size, hidden_size) = (1, trial, neuron)
                Initial firing rate for the RNN.

        Returns:
            r_output: (batch_size, seq_len, hidden_size) = (trial, steps, neuron)
                All hidden states over time.
            r: (num_layers * num_directions, batch_size, hidden_size) = (1, trial, neuron)
                Final hidden state.
        """
        batch_size, seq_len, _ = u.size()  # Extract dimensions from input
        if r0 is None:
            r0 = torch.zeros(1, batch_size, self.num_neurons, device=self.device)

        # Explicitly assert that r0 is not None to satisfy TorchScript type inference
        assert r0 is not None, "r0 should never be None at this point"

        # Initialize the firing rate for all time steps
        r_output = torch.zeros(batch_size, seq_len, self.num_neurons, device=self.device)
        
        # Current firing rate
        r = r0.squeeze(0)  # Shape: (batch_size, neuron)

        # Compute effective W with or without Dale's law enforcement.
        effective_W = self.W * self.dales_sign.view(1, -1)
        
        for t in range(seq_len):
            assert torch.all(r >= 0), "Negative values detected in r!"
            assert torch.all(torch.isfinite(r)), "NaN or Inf detected in r!"

            u_t = u[:, t, :]  # Current input at time step t: (batch_size, input_size)

            # RNN dynamics: τ * dr/dt + r = Φ(W * r + B * u)
            r_dot = (-r + self.activation_function(effective_W @ self.observed_r(r).T + self.B @ u_t.T).T) / self.tau
            
            # Update firing rate with Euler integration
            r = r + self.dt * r_dot
            
            # Store the firing rate for this time step
            r_output[:, t, :] = r

        # Return all hidden states and the final hidden state
        return r_output, r.unsqueeze(0)
    
    @torch.jit.export
    def decode(self, r: torch.Tensor) -> torch.Tensor:
        """
        Decodes the input firing rates (r) into a normalized output representation using observed_r.

        Args:
            r : Firing rate matrix of shape (num_trials, num_neurons)
        
        Returns:
            torch.Tensor: Decoded output of shape (num_trials, input_size) where input_size = (max_item_num * 2)
        """
        # (num_trials, input_size/2, 2)
        u_hat = (self.F @ self.observed_r(r).T).T.view(r.size(0), -1, 2)
        
        # Normalize the 2D representation of u_hat across the last dimension.
        normalized_u_hat = u_hat / torch.norm(u_hat, dim=2, keepdim=True)
        
        # Reshape normalized_u_hat into a flat representation.
        # (num_trials, input_size)
        return normalized_u_hat.view(r.size(0), -1)
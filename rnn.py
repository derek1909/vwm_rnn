"""
Core RNN model implementation for visual working memory.

This module implements the main biologically plausible recurrent neural network
for visual working memory tasks. The model includes Dale's law, heterogeneous
time constants, multiple noise models, and realistic neural dynamics to study
how neural circuits maintain orientation information during memory delays.

Author: Derek Jinyu Dong
Date: 2024-2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

class RNNMemoryModel(nn.Module):
    """
    Biologically plausible recurrent neural network for visual working memory tasks.
    
    This model implements a continuous-time RNN with Dale's law, realistic time constants,
    and various noise sources to simulate neural dynamics in visual working memory tasks.
    The network maintains orientation information during delay periods and supports
    variable set sizes (1-10 items).
    
    Key biological features:
    - Dale's law: Separate excitatory/inhibitory populations
    - Heterogeneous time constants sampled from log-uniform distribution
    - Multiple noise models (gamma, gaussian, constant SNR)
    - Firing rate saturation
    - Biologically realistic parameter ranges
    
    Spike Noise Models:
    The model supports four different noise types to simulate biological variability:
    1. 'gamma': Gamma approximation to Poisson spike statistics
    2. 'gauss': Gaussian with rate-dependent variance  
    3. 'puregauss': Constant Gaussian noise (least realistic)
    4. 'csnr': Constant signal-to-noise ratio across rates. Not used for final report.
    
    Attributes:
        num_neurons (int): Number of neurons
        dt (float): Integration time step
        tau (torch.Tensor): Time constants for each neuron
        dales_sign (torch.Tensor): Sign vector for Dale's law (+1 for E, -1 for I)
        B (nn.Parameter): Input weight matrix [num_neurons x input_dim]
        W (nn.Parameter): Recurrent weight matrix [num_neurons x num_neurons]
        F (nn.Parameter): Readout weight matrix [output_dim x num_neurons]
    """
    
    def __init__(self, max_item_num: int, num_neurons: int, dt: float = 0.1, 
                 tau_min: float = 50, tau_max: float = 50, 
                 spike_noise_type: str = "gamma", spike_noise_factor: float = 0.0, 
                 saturation_firing_rate: float = 60.0, device: str = 'cpu', 
                 positive_input: bool = True, dales_law: bool = True):
        super(RNNMemoryModel, self).__init__()
        # Initialize core parameters
        self.num_neurons = num_neurons
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.spike_noise_factor = spike_noise_factor
        self.spike_noise_type = spike_noise_type
        self.batch_first = True  # Required attribute to use FixedPointFinder
        self.device = device
        self.positive_input = positive_input
        self.saturation_firing_rate = saturation_firing_rate
        self.dales_law = dales_law

        # Select noise function based on type
        if spike_noise_type.lower() == "gamma":
            self._noise_fn = self._gamma_noise
        elif spike_noise_type.lower() == "gauss":
            self._noise_fn = self._gaussian_noise
        elif spike_noise_type.lower() == "puregauss":
            self._noise_fn = self._pure_gaussian_noise
        elif spike_noise_type.lower() == "csnr":
            self._noise_fn = self._const_snr_noise
        else:
            raise ValueError(
                f"Unsupported spike_noise_type '{self.spike_noise_type}'. "
                "Expected 'gamma', 'gauss', 'puregauss', or 'csnr'."
            )

        # Initialize Dale's law constraints
        if self.dales_law:
            excitatory_ratio = 0.5
            num_excitatory = int(num_neurons * excitatory_ratio)
            # Create sign vector: +1 for excitatory, -1 for inhibitory neurons
            dales_sign = torch.cat([
                torch.ones(num_excitatory), 
                -torch.ones(num_neurons - num_excitatory)
            ]).to(device)
        else:
            # No Dale's law constraint - all neurons can be either E or I
            dales_sign = torch.ones(num_neurons, device=device)

        self.register_buffer('dales_sign', dales_sign)

        # Initialize heterogeneous time constants
        # Log-uniform sampling between tau_min and tau_max
        tau = self.tau_min * torch.exp(
            torch.rand(num_neurons, device=device) * 
            math.log(self.tau_max / self.tau_min)
        )
        
        if self.dales_law:
            # Sort time constants separately for E and I populations
            num_excitatory = int(num_neurons * 0.5)
            tau[:num_excitatory], _ = torch.sort(tau[:num_excitatory])
            tau[num_excitatory:], _ = torch.sort(tau[num_excitatory:])
        else:
            tau, _ = torch.sort(tau)
        
        self.register_buffer('tau', tau)

        # Initialize input weight matrix B
        # Maps from stimulus features to neural activations
        if self.positive_input:
            # Positive-only input weights (more biologically realistic)
            std = 0.418
            input_dim = max_item_num * 3  # 3 dimensional positive input (see report)
            self.B = nn.Parameter(
                torch.abs(torch.randn(num_neurons, input_dim, device=device)) * std
            )
        else:
            # Unrestricted input weights
            std = 0.418 # This number is not derived for naive input.
            input_dim = max_item_num * 2  # cos, sin for each item
            self.B = nn.Parameter(
                torch.randn(num_neurons, input_dim, device=device) * std
            )
        
        # Initialize recurrent weight matrix W
        if self.dales_law:
            # Non-negative weights only (will be multiplied by Dale's sign)
            std = 1 / (num_neurons * 0.682)**0.5
            self.W = nn.Parameter(
                torch.abs(torch.randn(num_neurons, num_neurons, device=device)) * std
            )
        else:
            # Unrestricted recurrent weights
            self.W = nn.Parameter(
                torch.randn(num_neurons, num_neurons, device=device) / num_neurons**0.5
            )

        # Initialize readout weight matrix F
        # Maps from neural activities to decoded orientations
        output_dim = max_item_num * 2  # cos, sin for each decoded item
        std = (2 / (num_neurons + output_dim))**0.5
        self.F = nn.Parameter(
            torch.randn(output_dim, num_neurons, device=device) * std
        )

    def to(self, *args, **kwargs):
        """Override to properly handle device transfers."""
        super().to(*args, **kwargs)
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device:
            self.device = device
        return self

    def activation_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Neural activation function with saturation.
        
        Implements a biologically realistic activation function that saturates
        at high firing rates, preventing unrealistic neural activity.
        
        Args:
            x (torch.Tensor): Neural inputs [num_neurons, batch_size]
            
        Returns:
            torch.Tensor: Activated neural firing rates [num_neurons, batch_size]
        """
        if self.saturation_firing_rate > 0:
            # Saturating tanh-based activation
            return self.saturation_firing_rate/2 * (1 + torch.tanh(0.14 * x - 4.2))
        else:
            # Power law activation without saturation
            return 0.18 * torch.clamp(x - 10, min=1e-10) ** 1.7

    def _pure_gaussian_noise(self, r: torch.Tensor) -> torch.Tensor:
        """
        Apply pure (constant standard deviation) Gaussian noise to firing rates.
        
        Additive Gaussian noise with constant variance independent of firing rate.
        This is the simplest noise model but least biologically realistic since
        real neural noise typically scales with activity level.
        
        Noise formula: r_noisy = max(0, r + std * N(0,1))
        where std = spike_noise_factor * 15
        
        Args:
            r (torch.Tensor): Neural firing rates
            
        Returns:
            torch.Tensor: Noisy firing rates (non-negative via ReLU)
        """
        gauss_noise = self.spike_noise_factor * 15 * torch.randn_like(r, device=self.device)        
        return F.relu(r + gauss_noise)

    def _gaussian_noise(self, r: torch.Tensor) -> torch.Tensor:
        """
        Apply Poisson-like Gaussian noise to firing rates. See report for more details.
                
        Args:
            r (torch.Tensor): Neural firing rates
            
        Returns:
            torch.Tensor: Noisy firing rates (non-negative via ReLU)
        """
        poisson_like_noise = (
            self.spike_noise_factor * 
            torch.sqrt(r * 1e3 / self.dt + 1e-10) * 
            torch.randn_like(r, device=self.device)
        )
        return F.relu(r + poisson_like_noise)

    def _gamma_noise(self, r: torch.Tensor) -> torch.Tensor:
        """
        Apply Gamma-distributed noise to firing rates. See report for more details.
        
        Gamma parameters: shape = r * λ, rate = λ
        where λ = (dt/1000) / spike_noise_factor²
        
        Args:
            r (torch.Tensor): Neural firing rates [batch_size, num_neurons]
            
        Returns:
            torch.Tensor: Noisy firing rates sampled from Gamma distribution
        """
        # Gamma distribution parameters
        lam = (self.dt/1e3) / self.spike_noise_factor**2
        shape = torch.clamp(r * lam, min=1e-10)         #  -> (batch_size, num_neurons)
        rate = torch.full_like(shape, lam)              #  -> (batch_size, num_neurons)
        gamma = torch.distributions.Gamma(shape, rate)     
        corrupted_r = gamma.rsample()                      #  -> (batch_size, num_neurons)
        return corrupted_r

    def _const_snr_noise(self, r: torch.Tensor) -> torch.Tensor:
        """
        Apply constant signal-to-noise ratio noise using Gamma distribution. Not yet used.
        
        Gamma parameters: shape = κ, rate = κ/r
        where κ = (r₀ * dt/1000) / spike_noise_factor²
        
        Args:
            r (torch.Tensor): Neural firing rates [batch_size, num_neurons]
            
        Returns:
            torch.Tensor: Noisy firing rates with constant SNR
        """
        # Set reference firing rate for calibration
        r0 = 5.0
        kappa = (r0 * self.dt / 1e3) / (self.spike_noise_factor ** 2)

        shape = torch.full_like(r, kappa)
        rate = kappa / torch.clamp(r, min=1e-10)
        
        gamma = torch.distributions.Gamma(shape, rate)
        corrupted_r = gamma.rsample()                # differentiable sample
        return corrupted_r

    def observed_r(self, r: torch.Tensor) -> torch.Tensor:
        """
        Apply spike noise to firing rates.
        
        Applies the selected noise model to simulate biological variability
        in neural firing rates.
        
        Args:
            r (torch.Tensor): Clean (internal) firing rates [batch_size, num_neurons]
            
        Returns:
            torch.Tensor: Noisy firing rates with same shape as input
        """
        if self.spike_noise_factor > 0.0:
            return self._noise_fn(r)
        else:
            return r
    
    def forward(self, u: torch.Tensor, r0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN. Single time step simulation.
        
        Implements continuous-time dynamics with Euler integration:
        τ dr/dt = -r + Phi(W^T @ r + B^T @ u)
        
        Args:
            u (torch.Tensor): Input sequence [batch_size, seq_len, input_size]
                            where input_size = 3*max_item_num (cos, sin, presence per item)
            r0 (torch.Tensor, optional): Initial hidden state [1, batch_size, num_neurons]
                                       If None, initialized to zeros
                                       
        Returns:
            tuple: (r_output, r_final)
                - r_output: All hidden states [batch_size, seq_len, num_neurons]
                - r_final: Final hidden state [1, batch_size, num_neurons]
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
    def readout(self, r: torch.Tensor) -> torch.Tensor:
        """
        Decode firing rates into output representation.
                
        Args:
            r (torch.Tensor): Firing rates [batch_size, num_neurons]
        
        Returns:
            torch.Tensor: Decoded output [batch_size, max_item_num * 2]
        """
        # (batch_size, max_item_num * 2)
        return (self.F @ self.observed_r(r).T).T
    
    @torch.jit.export
    def decode(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Convert cosine/sine representations to angular orientations.
        
        Transforms 2D vector representations back to angles using atan2.
        
        Args:
            u_hat (torch.Tensor): Unnormalised Cosine/sine components [steps, trials, max_items * 2]
                                 Alternating cos, sin values for each item
        
        Returns:
            torch.Tensor: Decoded angles [trials, max_items] in radians (-π to π)
        """
        # Reshape to separate cos/sin components: (steps, trials, max_items, 2)
        u_hat_reshaped = u_hat.reshape(u_hat.shape[0], u_hat.shape[1], -1, 2)

        # Average over time steps to get final decoded values
        u_hat_reshaped = u_hat_reshaped.mean(dim=0)  # (trials, max_items, 2)

        # Extract cosine and sine components
        cos_thetas = u_hat_reshaped[..., 0]  # (trials, max_items)
        sin_thetas = u_hat_reshaped[..., 1]  # (trials, max_items)
        
        # Convert to angles using atan2
        decoded_thetas = torch.atan2(sin_thetas, cos_thetas)  # (trials, max_items)

        return decoded_thetas

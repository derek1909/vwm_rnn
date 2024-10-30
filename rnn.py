import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


device = 'mps'


# Activation function Φ(x) = 0.4 * (1 + tanh(0.4 * x - 3))


# Define the RNN model class
class RNNMemoryModel(nn.Module):
    def __init__(self, max_item_num, hidden_size, tau=1.0, dt=0.1, noise_level=0.0):
        super(RNNMemoryModel, self).__init__()
        self.hidden_size = hidden_size
        self.tau = tau  # unit: ms
        self.dt = dt   # unit: ms
        self.noise_level = noise_level

        # Learnable parameters
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.B = nn.Parameter(torch.randn(hidden_size, max_item_num*2))
        self.F = nn.Parameter(torch.randn(max_item_num*2, hidden_size))

    def activation_function(self, x):
        return 400 * (1 + torch.tanh(0.4 * x - 3)) / self.tau
    
    def forward(self, r, u):
        # RNN dynamics: τ * dr/dt + r = Φ(W * r + B * u)

        r_dot = (-r + self.activation_function(self.W @ r + self.B @ u)) / self.tau
        r = r + self.dt * r_dot + self.noise_level * torch.randn_like(r)

        return r
    
    def decode(self, r):
        # Memory decoding
        return self.F @ r

    # Function to generate input vector u(t) as described in the document
def generate_input(strength, theta, noise_level=0.1, stimuli_present=True):
    """
    Generate the input vector u(t) based on the stimuli presence and noise level.
    """
    theta = theta + noise_level * torch.randn_like(theta)

    item_num = strength.shape[0]
    u_0 = torch.zeros(2 * item_num)
    for i in range(item_num):
        u_0[2 * i] = strength[i] * torch.cos(theta[i])
        u_0[2 * i + 1] = strength[i] * torch.sin(theta[i])
    
    # Add noise if specified
    u_t = u_0 * (1 if stimuli_present else 0) 
    return u_t

# Function to calculate memory error and loss
def memory_loss(F, r, u_0, lambda_reg=0.1):
    u_hat = F @ r
    error = u_0 - u_hat
    error_loss = torch.norm(error, p=2) ** 2
    activation_penalty = lambda_reg * torch.norm(r, p=1)
    return error_loss+activation_penalty, activation_penalty


# Function to calculate memory loss as integral over time
def memory_loss_integral(F, r_list, u_0, lambda_reg=0.1, dt=0.1):
    """
    Calculate the loss as an integral over time using a given list of r values.
    """
    total_error_loss = 0.0
    total_activation_penalty = 0.0
    
    # Iterate over all time steps
    for r_t in r_list:
        u_hat = F @ r_t  # Decode memory at each time step
        error = u_0 - u_hat
        total_error_loss += torch.norm(error, p=2) ** 2
        total_activation_penalty += torch.norm(r_t, p=1)

    # Approximate the integral over time by multiplying with the time step size dt
    total_activation_penalty = dt * lambda_reg * total_activation_penalty
    total_error_loss = dt * total_error_loss
    total_loss = total_error_loss + total_activation_penalty

    return total_loss, total_activation_penalty

# Function to extract orientation from decoded memory
def extract_orientation(decoded_memory):
    """
    Given a decoded memory state containing [cos(theta), sin(theta), ...],
    extract the orientation angle theta in radians.
    """
    # Reshape to pairs (cos(theta), sin(theta)) if needed
    decoded_memory = decoded_memory.view(-1, 2)
    # Calculate the orientation angle using arctan2
    orientation = torch.atan2(decoded_memory[:, 1], decoded_memory[:, 0])
    return orientation


# Model parameters
max_item_num = 4
hidden_size = 20
tau = 10  # ms
dt = 0.1    # ms

num_epochs = 400
encode_noise = 0.0
priocess_noise = 0.0
decode_noise = 0.0
T = 200  # Number of time steps

# Training parameters
eta = 0.05 # learning_rate
lambda_reg=0.1



# Create the RNN model
model = RNNMemoryModel(max_item_num=max_item_num, hidden_size=hidden_size, tau=tau, dt=dt, noise_level=priocess_noise)
optimizer = optim.Adam(model.parameters(), lr=eta)


# Stimuli
strength = torch.tensor([10.0, 0.0, 0.0, 0.0])  # Strengths of items
theta = torch.tensor([0.8, 0.0, 0.0, 0.0])  # Orientations of items

# Initialize hidden state r with zeros
r = torch.zeros(hidden_size)
decoded_orientations_before = []

# Simulate the RNN over a given number of time steps and decode the memory at each step
for t in range(T):
    u_t = generate_input(strength, theta, noise_level=encode_noise, stimuli_present=(t < T//10))
    r = model(r, u_t)
    decoded_memory = model.decode(r)
    orientation = extract_orientation(decoded_memory)  # Get orientation from decoded memory
    decoded_orientations_before.append(orientation[0].item())  # Store only the first item’s orientation

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Initialize hidden state r with zeros
    r = torch.zeros(hidden_size)
    # List to store r at each time step
    r_list = []
    
    # Simulate the RNN over a given number of time steps
    for t in range(T):
        u_t = generate_input(strength, theta, noise_level=encode_noise, stimuli_present=(t < T//10))
        r = model(r, u_t)
        r_list.append(r.clone())  # Store r at this time step

    # initial input, ground truth. No noise
    u_0 = generate_input(strength, theta, noise_level=0.0, stimuli_present=True)

    # Calculate the integral-based memory error loss
    total_loss, activ_penal = memory_loss_integral(model.F, r_list, u_0, lambda_reg=lambda_reg, dt=dt)
    total_loss.backward()
    optimizer.step()
    
    # Print loss for tracking
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Error loss = {total_loss.item()-activ_penal.item()}, Active panelty = {activ_penal.item()}")


# Initialize hidden state r with zeros
r = torch.zeros(hidden_size)
decoded_orientations_after = []


# Simulate the RNN over a given number of time steps and decode the memory at each step
for t in range(T):
    u_t = generate_input(strength, theta, noise_level=encode_noise, stimuli_present=(t < T//10))
    r = model(r, u_t)
    # print(f'average firing rate: {torch.mean(r)}')
    decoded_memory = model.decode(r)
    orientation = extract_orientation(decoded_memory)  # Get orientation from decoded memory
    decoded_orientations_after.append(orientation[0].item())  # Store only the first item’s orientation

# Plot the decoded memory orientation vs. time
plt.figure(figsize=(10, 6))
plt.plot(range(T), decoded_orientations_before, marker='o', linestyle='-', color='b')
plt.title('before - Decoded Memory Orientation vs. Time')
plt.xlabel('Time Steps')
plt.ylabel('Orientation (radians)')
plt.grid(True)

# Plot the decoded memory orientation vs. time
plt.figure(figsize=(10, 6))
plt.plot(range(T), decoded_orientations_after, marker='o', linestyle='-', color='b')
plt.title('after - Decoded Memory Orientation vs. Time')
plt.xlabel('Time Steps')
plt.ylabel('Orientation (radians)')
plt.grid(True)
plt.show()
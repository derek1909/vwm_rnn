import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


device = 'mps'


# Activation function Φ(x) = 0.4 * (1 + tanh(0.4 * x - 3))
def activation_function(x):
    return 0.4 * (1 + torch.tanh(0.4 * x - 3))

# Define the RNN model class
class RNNMemoryModel(nn.Module):
    def __init__(self, max_item_num, hidden_size, tau=1.0, dt=0.1):
        super(RNNMemoryModel, self).__init__()
        self.hidden_size = hidden_size
        self.tau = tau  # unit: ms
        self.dt = dt   # unit: ms

        # Learnable parameters
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.B = nn.Parameter(torch.randn(hidden_size, max_item_num*2))
        self.F = nn.Parameter(torch.randn(max_item_num*2, hidden_size))
        
    def forward(self, r, u):
        # RNN dynamics: τ * dr/dt + r = Φ(W * r + B * u)

        r_dot = (-r + activation_function(self.W @ r + self.B @ u)) / self.tau
        r = r + self.dt * r_dot

        return r
    
    def decode(self, r):
        # Memory decoding
        return self.F @ r

    # Function to generate input vector u(t) as described in the document
def generate_input(strength, theta, noise_level=0.1, stimuli_present=True):
    """
    Generate the input vector u(t) based on the stimuli presence and noise level.
    """

    item_num = strength.shape[0]
    u_0 = torch.zeros(2 * item_num)
    for i in range(item_num):
        u_0[2 * i] = strength[i] * torch.cos(theta[i])
        u_0[2 * i + 1] = strength[i] * torch.sin(theta[i])
    
    # Add noise if specified
    noise = noise_level * torch.randn_like(u_0)
    u_t = u_0 * (1 if stimuli_present else 0) + noise
    return u_t

# Function to calculate memory error and loss
def memory_loss(F, r, u_0, lambda_reg=0.1):
    u_hat = F @ r
    error = u_0 - u_hat
    error_loss = torch.norm(error, p=2) ** 2
    activation_penalty = lambda_reg * torch.norm(r, p=1)
    return error_loss + activation_penalty

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


# Initialize parameters and model
max_item_num = 4
hidden_size = 10

tau = 1.0  # Time constant
dt = 0.1
eta = 0.01 # learning_rate
num_epochs = 1000

# Create the RNN model
model = RNNMemoryModel(max_item_num, hidden_size, tau, dt)
optimizer = optim.Adam(model.parameters(), lr=eta)


# Simulation parameters
T = 200  # Number of time steps
strength = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Strengths of items
theta = torch.tensor([0.0, 0.0, 0.0, 0.0])  # Orientations of items

# Initialize hidden state r with zeros
r = torch.zeros(hidden_size)
decoded_orientations_before = []

# Simulate the RNN over a given number of time steps and decode the memory at each step
for t in range(T):
    u_t = generate_input(strength, theta, noise_level=0.1, stimuli_present=(t < T // 2))
    r = model(r, u_t)
    decoded_memory = model.decode(r)
    orientation = extract_orientation(decoded_memory)  # Get orientation from decoded memory
    decoded_orientations_before.append(orientation[0].item())  # Store only the first item’s orientation





# Example training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()

    u_0 = generate_input(strength, theta, noise_level=0.1, stimuli_present=True)
    
    # Initialize hidden state r with zeros
    r = torch.zeros(hidden_size)
    
    # Simulate the RNN over a given number of time steps
    for t in range(T):
        u_t = generate_input(strength, theta, noise_level=0.1, stimuli_present=(t < T//2))
        r = model(r, u_t)
    
    # Calculate the memory error loss at the end of the simulation
    loss = memory_loss(model.F, r, u_0)
    loss.backward()
    optimizer.step()
    
    # Print loss for tracking
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")


# Initialize hidden state r with zeros
r = torch.zeros(hidden_size)
decoded_orientations_after = []

# Simulate the RNN over a given number of time steps and decode the memory at each step
for t in range(T):
    u_t = generate_input(strength, theta, noise_level=0.1, stimuli_present=(t < T // 2))
    r = model(r, u_t)
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




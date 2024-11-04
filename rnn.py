import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


device = 'mps'

# torch.manual_seed(42)
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
        # normalise initialization so each neuron recieve same amount of recurrent input 
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size)/hidden_size**0.5) 
        self.B = nn.Parameter(torch.randn(hidden_size, max_item_num*2))
        self.F = nn.Parameter(torch.randn(max_item_num*2, hidden_size)/(hidden_size)**0.5)

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
def generate_input(strength, theta, noise_level=0.0, stimuli_present=True):
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
max_item_num = 8
hidden_size = 20
tau = 10  # ms
dt = 1 # ms

encode_noise = 0.0
process_noise = 0.0
decode_noise = 0.0

T_stimi = 100 #ms
T_delay = 800 #ms
T_decode = 100
T_simul = T_stimi + T_delay + T_decode
simul_steps = int(T_simul/dt)

item_num = 2


# Training parameters
train_rnn = True
num_epochs = 100
eta = 0.05 # learning_rate
lambda_reg=0.1 # coeff for activity penalty
num_trials = 10 # Number of trials per epoch

# Create the RNN model
model = RNNMemoryModel(max_item_num=max_item_num, hidden_size=hidden_size, tau=tau, dt=dt, noise_level=process_noise)

if train_rnn == True:
        
    optimizer = optim.Adam(model.parameters(), lr=eta)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        total_loss = 0
        total_activ_penal = 0
        
        for trial in range(num_trials):
            input_item = torch.zeros(max_item_num)  # Initialize a zero vector
            one_hot_index = torch.randperm(max_item_num)[:item_num]  # Randomly pick items
            input_item[one_hot_index] = 1

            input_theta = (torch.rand(max_item_num) * 2 * torch.pi) - torch.pi

            # print('input_item',input_item)
            # print('input_theta',input_theta)
            # Initialize hidden state r with zeros
            r = torch.zeros(hidden_size)
            # List to store r at each time step for this trial
            r_list = []

            # Simulate the RNN over the given number of time steps
            for step in range(simul_steps):
                time = step * dt
                u_t = generate_input(input_item, input_theta, noise_level=encode_noise, stimuli_present=(time < T_stimi))
                r = model(r, u_t)
                if time > (T_stimi + T_delay):
                    r_list.append(r.clone())  # Store r at this time step

            # Initial input for ground truth without noise
            u_0 = generate_input(input_item, input_theta, stimuli_present=True)

            # Calculate the memory error loss for this trial
            trial_loss, activ_penal = memory_loss_integral(model.F, r_list, u_0, lambda_reg=lambda_reg, dt=dt)

            # Accumulate loss values
            total_loss += trial_loss
            total_activ_penal += activ_penal

            # Accumulate gradients for this trial without optimizer step
            trial_loss.backward()

        # After all trials, perform the optimizer step for averaged gradient
        # Scale the gradients by the number of trials
        for param in model.parameters():
            param.grad /= num_trials
        optimizer.step()  # Update the model parameters

        # Print averaged loss for tracking
        avg_loss = total_loss.item() / num_trials
        avg_activ_penal = total_activ_penal.item() / num_trials
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Error loss = {avg_loss - avg_activ_penal}, Active penalty = {avg_activ_penal}")


    torch.save(model.state_dict(), 'model_weights.pth')
    
model = RNNMemoryModel(max_item_num=max_item_num, hidden_size=hidden_size, tau=tau, dt=dt, noise_level=process_noise)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # Set to evaluation mode if not training


# Initialize hidden state r with zeros
r = torch.zeros(hidden_size)
decoded_orientations_after = []


# Stimuli
strength = torch.tensor([1, 0.0, 0.0, 0, 0, 1, 0, 0])  # Strengths of items
theta = torch.tensor([0.8, 0.0, 0.0, 0.0, 0, 0, 0, 0])  # Orientations of items

# Simulate the RNN over the given number of time steps
for step in range(simul_steps):
    time = step * dt
    u_t = generate_input(strength, theta, noise_level=encode_noise, stimuli_present=(time < T_stimi))
    r = model(r, u_t)
    decoded_memory = model.decode(r)
    orientation = extract_orientation(decoded_memory)  # Get orientation from decoded memory
    decoded_orientations_after.append(orientation[0].item())  # Store only the first item’s orientation

# # Plot the decoded memory orientation vs. time
# plt.figure(figsize=(10, 6))
# plt.plot(range(T), decoded_orientations_before, marker='o', linestyle='-', color='b')
# plt.title('before - Decoded Memory Orientation vs. Time')
# plt.xlabel('Time Steps')
# plt.ylabel('Orientation (radians)')
# plt.grid(True)

# Plot the decoded memory orientation vs. time
plt.figure(figsize=(10, 6))
plt.plot(range(simul_steps), decoded_orientations_after, marker='o', linestyle='-', color='b')
plt.title('after - Decoded Memory Orientation vs. Time')
plt.xlabel('Time Steps')
plt.ylabel('Orientation (radians)')
plt.grid(True)
plt.show()
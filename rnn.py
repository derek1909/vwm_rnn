import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# use GPU would make it x20 slower - no complex matrix calculation requires GPU.

# Define the RNN model class
class RNNMemoryModel(nn.Module):
    def __init__(self, max_item_num, hidden_size, tau=1.0, dt=0.1, noise_level=0.0):
        super(RNNMemoryModel, self).__init__()
        self.hidden_size = hidden_size
        self.tau = tau  # unit: ms
        self.dt = dt   # unit: ms
        self.noise_level = noise_level

        # Learnable parameters
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size**0.5) 
        self.B = nn.Parameter(torch.randn(hidden_size, max_item_num*2))
        self.F = nn.Parameter(torch.randn(max_item_num*2, hidden_size) / hidden_size**0.5)

    def activation_function(self, x):
        return 400 * (1 + torch.tanh(0.4 * x - 3)) / self.tau
    
    def forward(self, r, u):
        # RNN dynamics: τ * dr/dt + r = Φ(W * r + B * u)
        r_dot = (-r + self.activation_function(self.W @ r + self.B @ u)) / self.tau
        r = r + self.dt * r_dot + self.noise_level * torch.randn_like(r)
        return r
    
    def decode(self, r):
        return self.F @ r

def generate_input(strength, theta, noise_level=0.0, stimuli_present=True):
    theta = theta + noise_level * torch.randn_like(theta)
    item_num = strength.shape[0]
    u_0 = torch.zeros(2 * item_num)
    for i in range(item_num):
        u_0[2 * i] = strength[i] * torch.cos(theta[i])
        u_0[2 * i + 1] = strength[i] * torch.sin(theta[i])
    u_t = u_0 * (1 if stimuli_present else 0) 
    return u_t

def memory_loss(F, r, u_0, lambda_reg=0.1):
    u_hat = F @ r
    error = u_0 - u_hat
    error_loss = torch.norm(error, p=2) ** 2
    activation_penalty = lambda_reg * torch.norm(r, p=1)
    return error_loss + activation_penalty, activation_penalty

def memory_loss_integral(F, r_list, u_0, lambda_reg=0.1, dt=0.1):
    total_error_loss = 0.0
    total_activation_penalty = 0.0
    for r_t in r_list:
        u_hat = F @ r_t
        error = u_0 - u_hat
        total_error_loss += torch.norm(error, p=2) ** 2
        total_activation_penalty += torch.norm(r_t, p=1)
    total_activation_penalty = lambda_reg * total_activation_penalty / len(r_list)
    total_error_loss = dt * total_error_loss / len(r_list)
    total_loss = total_error_loss + total_activation_penalty
    return total_loss, total_activation_penalty

def extract_orientation(decoded_memory):
    decoded_memory = decoded_memory.view(-1, 2)
    orientation = torch.atan2(decoded_memory[:, 1], decoded_memory[:, 0])
    return orientation


# Model parameters
max_item_num = 8
hidden_size = 50
tau = 10
dt = 1
encode_noise = 0.0
process_noise = 0.0
decode_noise = 0.0
T_stimi = 100
T_delay = 0
T_decode = 500
T_simul = T_stimi + T_delay + T_decode
simul_steps = int(T_simul/dt)
item_num = 2

# Training parameters
train_rnn = True
num_epochs = 100
eta = 0.05 # learning_rate
lambda_reg = 0.1 # coeff for activity penalty
num_trials = 20 # Number of trials per epoch
# num_cores = 2

# Run a single trial, calculates and returns the loss and activation penalty
def run_a_trial(model, input_item, input_theta):
    r = torch.zeros(hidden_size)
    r_list = []

    # Simulate the RNN over the time steps
    for step in range(simul_steps):
        time = step * dt
        u_t = generate_input(input_item, input_theta, stimuli_present=(time < T_stimi))
        r = model(r, u_t)
        if time > (T_stimi + T_delay):
            r_list.append(r.clone())

    # Calculate the memory loss for this trial
    u_0 = generate_input(input_item, input_theta, stimuli_present=True)
    trial_loss, activ_penal = memory_loss_integral(model.F, r_list, u_0, lambda_reg=lambda_reg, dt=dt)
    return trial_loss, activ_penal


def main():
    # Create the RNN model
    model = RNNMemoryModel(max_item_num=max_item_num, hidden_size=hidden_size, tau=tau, dt=dt, noise_level=process_noise)
    model.load_state_dict(torch.load('model_weights.pth')) # to continue training

    if train_rnn:
        optimizer = optim.Adam(model.parameters(), lr=eta)
        with tqdm(total=num_epochs, desc="Training Progress", unit="epoch") as pbar_epoch:
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                total_loss = 0
                total_activ_penal = 0

                # Generate inputs for each trial beforehand
                input_items = torch.zeros(num_trials, max_item_num)
                one_hot_indices = torch.stack([torch.randperm(max_item_num)[:item_num] for _ in range(num_trials)]) # ensuring no repeating indices in each trial
                input_items.scatter_(1, one_hot_indices, 1)
                input_thetas = (torch.rand(num_trials, max_item_num) * 2 * torch.pi) - torch.pi

                # Run trials in parallel - not implemented yet :(
                results = [run_a_trial(model, input_items[i], input_thetas[i]) for i in range(num_trials)]

                # Accumulate results
                for trial_loss, activ_penal in results:
                    total_loss += trial_loss
                    total_activ_penal += activ_penal
                    trial_loss.backward() # accumulate grad

                # Average the gradients and update model parameters
                for param in model.parameters():
                    param.grad /= num_trials
                optimizer.step()

                avg_loss = total_loss.item() / num_trials
                avg_activ_penal = total_activ_penal.item() / num_trials
                error_loss = avg_loss - avg_activ_penal

                # Update the tqdm progress bar with current metrics
                pbar_epoch.set_postfix({
                    "Error loss": f"{error_loss:.4f}",
                    "Active penalty": f"{avg_activ_penal:.4f}"
                })

                pbar_epoch.update(1)

        torch.save(model.state_dict(), 'model_weights.pth')
        
    # Load the trained model for evaluation
    model = RNNMemoryModel(max_item_num=max_item_num, hidden_size=hidden_size, tau=tau, dt=dt, noise_level=process_noise)
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()  # Set to evaluation mode if not training

    r = torch.zeros(hidden_size)
    decoded_orientations_after = []

    strength = torch.tensor([1, 0.0, 0.0, 0, 0, 1, 0, 0])
    theta = torch.tensor([-1, 0.0, 0.0, 0.0, 0, 0, 0, 0])

    for step in range(simul_steps):
        time = step * dt
        u_t = generate_input(strength, theta, noise_level=encode_noise, stimuli_present=(time < T_stimi))
        r = model(r, u_t)
        decoded_memory = model.decode(r)
        orientation = extract_orientation(decoded_memory)  # Get orientation from decoded memory
        decoded_orientations_after.append(orientation[0].item())  # Store only the first item’s orientation

    # Plot the decoded memory orientation vs. time
    plt.figure(figsize=(10, 6))
    plt.plot(range(simul_steps), decoded_orientations_after, marker='o', linestyle='-', color='b')
    plt.title('Decoded Memory Orientation vs. Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Orientation (radians)')
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()
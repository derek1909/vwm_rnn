import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# use GPU would make it x20 slower - no complex matrix calculation requires GPU.

# Define the RNN model class
class RNNMemoryModel(nn.Module):
    def __init__(self, max_item_num, num_neurons, tau=1.0, dt=0.1, noise_level=0.0):
        super(RNNMemoryModel, self).__init__()
        self.num_neurons = num_neurons
        self.tau = tau  # unit: ms
        self.dt = dt   # unit: ms
        self.noise_level = noise_level

        # Learnable parameters
        self.W = nn.Parameter(torch.randn(num_neurons, num_neurons) / num_neurons**0.5) 
        self.B = nn.Parameter(torch.randn(num_neurons, max_item_num*2)*10)
        # self.b = nn.Parameter(torch.randn(num_neurons, 1))
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

def generate_input(presence, theta, noise_level=0.0, stimuli_present=True):
    theta = theta + noise_level * torch.randn_like(theta)
    max_item_num = presence.shape[1]
    u_0 = torch.zeros(presence.size(0), 2 * max_item_num)
    for i in range(max_item_num):
        u_0[:, 2 * i] = presence[:, i] * torch.cos(theta[:, i])
        u_0[:, 2 * i + 1] = presence[:, i] * torch.sin(theta[:, i])
    u_t = u_0 * (1 if stimuli_present else 0) 
    return u_t

def memory_loss_integral(F, r_stack, u_0, presence, lambda_err=1.0, lambda_reg=0.1):
    # Vectorized memory loss over all time steps and trials
    # r_stack = (steps, trials, neurons)
    # u_0 = (trials, max_items*2)
    # presence = (trials, max_items)
    num_steps, num_trials, num_neurons = r_stack.shape
    u_hat_stack = decode(F, r_stack.reshape(-1, num_neurons)).reshape(num_steps, num_trials, -1)


    error = (u_0 - u_hat_stack * presence.repeat_interleave(2, dim=1)).pow(2) / torch.sum(presence,dim=1).unsqueeze(-1) #average over num of presented item (might be different for each trial)
    error = lambda_err * error.sum() / (num_steps * num_trials) # average over trials and steps
    
    activation_penalty = lambda_reg * r_stack.abs().mean()
    total_loss = error + activation_penalty

    return total_loss, activation_penalty

# def extract_orientation(decoded_memory):
#     decoded_memory = decoded_memory.view(decoded_memory.size(0), -1, 2)
#     orientation = torch.atan2(decoded_memory[:, :, 1], decoded_memory[:, :, 0])
#     return orientation


# Model parameters
max_item_num = 1
num_neurons = 64
tau = 50
dt = 5
encode_noise = 0.001
process_noise = 0.001
decode_noise = 0.0
T_init = 100
T_stimi = 400
T_delay = 200
T_decode = 800
T_simul = T_init + T_stimi + T_delay + T_decode
simul_steps = int(T_simul/dt)
item_num = 1

# Training parameters
train_rnn = True  # Set to True if training is required
train_from_scratch = False
num_epochs = 500
eta = 2e-4  # learning_rate
lambda_reg = 2e-5  # coeff for activity penalty
lambda_err = 1.0  # coeff for error penalty
num_trials = 64  # Number of trials per epoch


def main():
    model = RNNMemoryModel(
        max_item_num=max_item_num,
        num_neurons=num_neurons,
        tau=tau,
        dt=dt,
        noise_level=process_noise
    )
    if not train_from_scratch:
        model.load_state_dict(torch.load('model_weights.pth')) # to continue training
    
    # Generate input_thetas
    # input_thetas = ((torch.rand(num_trials, max_item_num) * 2 * torch.pi) - torch.pi).requires_grad_()
    input_thetas = torch.linspace(-torch.pi, torch.pi, num_trials).unsqueeze(1).repeat(1, max_item_num).requires_grad_()
    if train_rnn:
        optimizer = optim.Adam(model.parameters(), lr=eta)
        error_per_epoch = []
        activation_penalty_per_epoch = []

        with tqdm(total=num_epochs, desc="Training Progress", unit="epoch") as pbar_epoch:
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                total_loss = 0
                total_activ_penal = 0

                # Generate input presence and theta for all trials
                input_presence = torch.zeros(num_trials, max_item_num, requires_grad=True)
                one_hot_indices = torch.stack([torch.randperm(max_item_num)[:item_num] for _ in range(num_trials)])
                input_presence = input_presence.scatter(1, one_hot_indices, 1)

                # Initialize hidden states and collect activations for each time step
                r = torch.zeros(num_trials, num_neurons)
                r_list = []

                # Simulate the RNN across all trials and time steps
                for step in range(simul_steps):
                    time = step * dt
                    u_t = generate_input(
                        input_presence,
                        input_thetas,
                        noise_level=encode_noise,
                        stimuli_present=(T_init < time < T_stimi + T_init)
                    )
                    r = model(r, u_t)
                    if time > (T_stimi + T_delay + T_init):
                        r_list.append(r.clone())

                # Calculate loss over all trials and time steps
                r_stack = torch.stack(r_list)
                u_0 = generate_input(input_presence, input_thetas, stimuli_present=True)  # u_0 has no noise
                total_loss, total_activ_penal = memory_loss_integral(
                    model.F, r_stack, u_0, input_presence,
                    lambda_err=lambda_err, lambda_reg=lambda_reg
                )
                total_loss.backward()

                # Update model parameters
                optimizer.step()

                # Track errors and penalties
                error_per_epoch.append((total_loss - total_activ_penal).item())
                activation_penalty_per_epoch.append((total_activ_penal / lambda_reg).item())

                # Update progress bar
                pbar_epoch.set_postfix({
                    "Error": f"{error_per_epoch[-1]:.4f}",
                    "Activation": f"{activation_penalty_per_epoch[-1]:.4f}"
                })
                pbar_epoch.update(1)

        torch.save(model.state_dict(), 'model_weights.pth')

        # Plot error and activation penalty
        plt.figure(1, figsize=(10, 6))
        plt.plot(error_per_epoch, label="Error", marker='o')
        plt.plot(activation_penalty_per_epoch, label="Activation Penalty", marker='x')
        plt.title('Training Error and Activation Penalty vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True)
        
    # Load the trained model for evaluation
    model = RNNMemoryModel(max_item_num=max_item_num, num_neurons=num_neurons, tau=tau, dt=dt, noise_level=process_noise)
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()  # Set to evaluation mode if not training

    r = torch.zeros(1, num_neurons)
    angle_targets = [-1.9, -0.1, 0.6, 1.8]  # List of different target angles to test
    item_idx = 0

    # Create a dictionary to store decoded orientations for each target angle
    decoded_orientations_dict = {}

    presence = torch.tensor([1,]).reshape(1, max_item_num)

    for angle_target in angle_targets:
        decoded_orientations_after = []
        theta = torch.tensor([angle_target,]).reshape(1, max_item_num)
        r = torch.zeros(1, num_neurons)  # Reset the RNN state for each test case

        for step in range(simul_steps):
            time = step * dt
            u_t = generate_input(presence, theta, noise_level=encode_noise, stimuli_present=(T_init < time < T_stimi + T_init))
            r = model(r, u_t)
            decoded_memory = decode(model.F, r)
            decoded_memory = decoded_memory.view(decoded_memory.size(0), -1, 2)
            orientation = torch.atan2(decoded_memory[:, :, 1], decoded_memory[:, :, 0])

            decoded_orientations_after.append(orientation[0, item_idx].item())  # Store only the first item’s orientation

        # Store the results for the current angle target
        decoded_orientations_dict[angle_target] = decoded_orientations_after

    # Plot the results for all angle targets
    plt.figure(2, figsize=(10, 6))
    time_steps = torch.tensor([step * dt for step in range(simul_steps)])

    # Plot response lines and target lines
    for angle_target, decoded_orientations in decoded_orientations_dict.items():
        # Plot the response line
        line, = plt.plot(
            time_steps,
            decoded_orientations,
            marker='o',
            linestyle='-',
            markersize=4,  # Set the marker size
            color=None  # Automatically assign a unique color
        )
        # Plot the target line
        plt.axhline(
            y=angle_target,
            color=line.get_color(),  # Use the same color as the response line
            linestyle='--'
        )

    response_legend = plt.Line2D(
        [0], [0], color='blue', marker='o', linestyle='-', markersize=4, label='Response'
    )
    target_legend = plt.Line2D(
        [0], [0], color='blue', linestyle='--', label='Target'
    )
    stimulus_period_legend = plt.Line2D(
        [0], [0], color='orange', lw=4, alpha=0.3, label="Stimulus period"
    )


    plt.axvspan(T_init, T_stimi + T_init, color='orange', alpha=0.3)
    plt.title('Decoded Memory Orientations vs. Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Orientation (radians)')
    plt.grid(True)

    # Combine and display the simplified legend
    plt.legend(handles=[stimulus_period_legend, response_legend, target_legend], loc='upper right')
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
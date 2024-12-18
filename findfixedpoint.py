import numpy as np
import torch
from rnn import *
from config import *
from utils import *
from train import *


from FixedPoints.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from FixedPoints.plot_utils import plot_fps

def analyze_fixed_points(model, hidden_states, input_dim):
    """
    Analyze and visualize fixed points of the trained RNN.

    Args:
        model: Trained RNN model (torch.nn.Module)
        hidden_states: Hidden states collected during simulation (steps, trials, neurons)
        input_dim: Dimensionality of the input space.

    Returns:
        unique_fps: Unique fixed points found.
    """
    # Hyperparameters for fixed point finder
    fpf_hps = {
        'max_iters': 10000,
        'lr_init': 1.0,
        'outlier_distance_scale': 10.0,
        'verbose': True,
        'super_verbose': True
    }

    NOISE_SCALE = 0.5  # Standard deviation of noise added to states
    N_INITS = 1024     # Number of initial states for optimization

    # (steps, trials, neurons) ->  [n_batch x n_time x n_states]
    initial_states = hidden_states.reshape(num_trials, -1, num_neurons)

    # Sample noisy initial states
    fpf = FixedPointFinder(model, **fpf_hps)

    # sampled_states has shape [n_inits x n_states]
    sampled_states = fpf.sample_states(initial_states, n_inits=N_INITS, noise_scale=NOISE_SCALE)

    # Inputs to analyze the RNN in the absence of external stimuli
    inputs = np.zeros([1, input_dim])

    # Find fixed points
    unique_fps, all_fps = fpf.find_fixed_points(sampled_states, inputs)

    # Visualization
    plot_fps(unique_fps, hidden_states.reshape(-1, hidden_states.shape[-1]),
             plot_batch_idx=list(range(min(30, hidden_states.shape[1]))),
             plot_start_time=10)

    return unique_fps


# Load model
model = RNNMemoryModel(max_item_num, num_neurons, tau, dt, process_noise)
model, history = load_model_and_history(model, model_dir)


# Simulate to collect hidden states
# Generate presence for each group
input_presence = torch.zeros(num_trials, max_item_num, requires_grad=True)    
trials_per_group = num_trials // len(item_num)  # Ensure equal split
remaining_trials = num_trials % len(item_num)  # Handle leftover trials
trial_counts = [trials_per_group + (1 if i < remaining_trials else 0) for i in range(len(item_num))]

start_index = 0
for i, count in enumerate(trial_counts):
    end_index = start_index + count
    one_hot_indices = torch.stack([torch.randperm(max_item_num)[:item_num[i]] for _ in range(count)])
    input_presence_temp = input_presence.clone()
    input_presence_temp[start_index:end_index] = input_presence_temp[start_index:end_index].scatter(1, one_hot_indices, 1)
    input_presence = input_presence_temp
    start_index = end_index

input_thetas = ((torch.rand(num_trials, max_item_num) * 2 * torch.pi) - torch.pi).requires_grad_()

r = torch.zeros(num_trials, num_neurons)
r_list = []

for step in range(simul_steps):
    time = step * dt
    u_t = generate_input(input_presence, input_thetas, noise_level=encode_noise,
                            stimuli_present=(T_init < time < T_stimi + T_init))
    r = model(r, u_t)
    if time > (T_init + T_stimi + T_delay):  # Collect hidden states after stimuli
        r_list.append(r.clone())

hidden_states = torch.stack(r_list).detach().cpu().numpy()  # Shape: (steps, trials, neurons)

# Run fixed point analysis
print("Running Fixed Point Analysis...")
unique_fps = analyze_fixed_points(model, hidden_states, input_dim=max_item_num*2)
print(f"Fixed points found: {len(unique_fps)}")

# Save final model and history
# save_model_and_history(model, history, model_dir)

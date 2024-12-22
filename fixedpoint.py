import numpy as np
import torch
import ipdb

from rnn import *
from config import *
from utils import *
from train import *


from FixedPoints.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from FixedPoints.plot_utils import plot_fps

def analyze_fixed_points(model, hidden_states):
    """
    Analyze and visualize fixed points of the trained RNN.

    Args:
        model: Trained RNN model (torch.nn.Module)
        hidden_states: Hidden states collected during simulation (steps, trials, neurons)

    Returns:
        unique_fps: Unique fixed points found.
    """

    # (steps, trials, neurons) -> [n_batch x n_time x n_states]
    initial_states = hidden_states.reshape(num_trials, -1, num_neurons)

    # Sample noisy initial states
    fpf = FixedPointFinder(model, **fpf_hps)

    # sampled_states has shape [n_inits x n_states]
    sampled_states = fpf.sample_states(initial_states, n_inits=fpf_N_init, noise_scale=fpf_noise_scale)

    # Inputs to analyze the RNN in the absence of external stimuli
    inputs = np.zeros([1, max_item_num * 2])

    # Find fixed points
    unique_fps, all_fps = fpf.find_fixed_points(sampled_states, inputs)

    # Visualization
    fig = plot_fps(
        unique_fps,
        hidden_states.cpu().numpy(),  # Convert back to CPU for plotting
        plot_batch_idx=list(range(min(30, hidden_states.shape[1]))),
        plot_start_time=T_init,
        save_path=model_dir,
    )

    return unique_fps


def fixed_points_finder(model):
    """
    Simulate the RNN to collect hidden states and find fixed points.
    """

    ## Simulate to collect hidden states ##
    # Generate presence for each group
    input_presence = torch.zeros(num_trials, max_item_num, requires_grad=True, device=device)
    trials_per_group = num_trials // len(item_num)  # Ensure equal split
    remaining_trials = num_trials % len(item_num)  # Handle leftover trials
    trial_counts = [trials_per_group + (1 if i < remaining_trials else 0) for i in range(len(item_num))]

    start_index = 0
    for i, count in enumerate(trial_counts):
        end_index = start_index + count
        one_hot_indices = torch.stack([torch.randperm(max_item_num, device=device)[:item_num[i]] for _ in range(count)])
        input_presence_temp = input_presence.clone()
        input_presence_temp[start_index:end_index] = input_presence_temp[start_index:end_index].scatter(1, one_hot_indices, 1)
        input_presence = input_presence_temp
        start_index = end_index

    input_thetas = ((torch.rand(num_trials, max_item_num, device=device) * 2 * torch.pi) - torch.pi)

    u_t = generate_input_all(
        presence=input_presence,
        theta=input_thetas,
        noise_level=encode_noise,
        T_init=T_init,
        T_stimi=T_stimi,
        T_delay=T_delay,
        T_decode=T_decode,
        dt=dt,
        alpha=positive_input,
    )

    r_output, _ = model(u_t, r0=None)  # (trial, steps, neuron)



    ## Run fixed point analysis ##
    print("Running Fixed Point Analysis...")
    hidden_states = r_output.detach().cpu()
    model = model.to('cpu')
    unique_fps = analyze_fixed_points(model, hidden_states)
    print(f"Fixed points found: {len(unique_fps)}")

    return unique_fps
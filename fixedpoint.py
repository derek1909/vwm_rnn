import numpy as np
import torch
import ipdb

from rnn import *
from config import *
from utils import *
from train import *


from FixedPoints.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from FixedPoints.plot_utils import plot_fps, plot_F_vs_PCA_1item

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

    # valid_bxt: [n_batch x n_time]
    if fpf_name == 'decode':
        start_t_idx = int((T_init + T_stimi + T_delay) / dt)
        end_t_idx = -1
    # elif fpf_name == 'delay':
    #     start_t_idx = int((T_init + T_stimi) / dt)
    #     end_t_idx = int((T_init + T_stimi + T_delay) / dt)
    elif fpf_name == 'stimuli':
        start_t_idx = int(T_init / dt)
        end_t_idx = int((T_init + T_stimi) / dt)
    elif fpf_name == 'init':
        start_t_idx = 0
        end_t_idx = int(T_init / dt)
    else:
        start_t_idx = 0
        end_t_idx = -1
    
    valid_bxt = np.zeros((num_trials, simul_steps))
    valid_bxt[:, start_t_idx:end_t_idx] = 1

    # sampled_states has shape [n_inits x n_states]
    sampled_states = fpf.sample_states(initial_states, n_inits=fpf_N_init, noise_scale=fpf_noise_scale, valid_bxt=valid_bxt)

    # Inputs to analyze the RNN in the absence of external stimuli
    inputs = np.zeros([1, max_item_num * 2])

    # Find fixed points
    unique_fps, all_fps = fpf.find_fixed_points(sampled_states, inputs)

    # Visualization
    # trials_to_plot = list(range(min(30, num_trials)))
    trials_to_plot = list(range(min(64, num_trials)))
    fig = plot_fps(
        unique_fps,
        state_traj=hidden_states.cpu().numpy(),
        plot_batch_idx=trials_to_plot,
        plot_start_time=T_init,
        save_path=f'{model_dir}/{fpf_name}',
        )

    return unique_fps


def fixed_points_finder(model):
    """
    Simulate the RNN to collect hidden states and find fixed points.
    """
    np.random.seed(42)
    torch.manual_seed(42)

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

    # input_thetas = ((torch.rand(num_trials, max_item_num, device=device) * 2 * torch.pi) - torch.pi)
    input_thetas = torch.linspace(-torch.pi, torch.pi, num_trials, device=device).unsqueeze(1) # for 1item

    u_t = generate_input_all(
        presence=input_presence,
        theta=input_thetas,
        noise_level=0.0,
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



    if fpf_pca_bool:
        plot_F_vs_PCA_1item(
            model.F.detach().cpu(),
            hidden_states[:,-1,:],
            save_path=f'{model_dir}'
        )

    return unique_fps
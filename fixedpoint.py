import numpy as np
import ipdb


from FixedPoints.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from utils_fpf import *

def analyze_fixed_points(model, input_states, hidden_states, fpf_name):
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
    trials_to_plot = list(range(min(64, num_trials)))
    fig = plot_fps(
        unique_fps,
        state_traj=hidden_states,
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
    u_t, hidden_states = prepare_state(model)
    model = model.to('cpu')

    for fpf_name in fpf_names:
        print(f"Running Fixed Point Analysis for {fpf_name}")
        unique_fps = analyze_fixed_points(model, u_t, hidden_states, fpf_name)
        print(f"Fixed points found: {len(unique_fps)}")

    if fpf_pca_bool:
        plot_F_vs_PCA_1item(
            model.F,
            hidden_states[:,-1,:],
            save_path=f'{model_dir}'
        )

    return unique_fps
import numpy as np
import ipdb


from Analysis.FixedPoints.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from Analysis.fixedpoint_utils import *

def analyze_fixed_points(model, input_states, hidden_states, fpf_name, iteration):
    """
    Analyze and visualize fixed points of the trained RNN.

    Args:
        model: Trained RNN model (torch.nn.Module)
            The recurrent neural network to analyze.
        input_states: The input vector (num_trials, 3 [or 2] * max_item_num)
        hidden_states: Hidden states collected during simulation (trials, steps, neuron)
        fpf_name: the time period for fixed point analysis (e.g., 'stimuli', 'decode').

    Returns:
        unique_fps: torch.Tensor
            The unique fixed points found during the analysis.
    """

    # Sample noisy initial states
    fpf = FixedPointFinder(model, **fpf_hps)

    # valid_bxt: [n_batch x n_time]
    if fpf_name == 'fpf_decode':
        start_t_idx = -2
        end_t_idx = -1
    # elif fpf_name == 'delay':
    #     start_t_idx = int((T_init + T_stimi) / dt)
    #     end_t_idx = int((T_init + T_stimi + T_delay) / dt)
    elif fpf_name == 'fpf_stimuli':
        start_t_idx = int(T_init / dt)
        end_t_idx = int((T_init + T_stimi) / dt)
    elif fpf_name == 'fpf_init':
        start_t_idx = 0
        end_t_idx = int(T_init / dt)
    else:
        start_t_idx = 0
        end_t_idx = -1
    
    valid_bxt = np.zeros((fpf_trials, simul_steps))
    valid_bxt[:, start_t_idx:end_t_idx] = 1

    # sampled_states has shape [n_inits x n_states]
    sampled_states, trial_indices = fpf.sample_states(hidden_states, n_inits=fpf_N_init, noise_scale=fpf_noise_scale, valid_bxt=valid_bxt)

    # Inputs to analyze the RNN in the absence of external stimuli
    if fpf_name == 'stimuli':
        inputs = input_states[trial_indices].numpy() # [n_inits x max_item_num* 3]
    else:
        inputs = np.zeros([1, input_states.shape[1]])

    # Find fixed points
    unique_fps, _ = fpf.find_fixed_points(sampled_states, inputs)

    # Visualization
    trials_to_plot = list(range(min(64, fpf_trials)))
    fig = plot_fps(
        unique_fps,
        state_traj=hidden_states,
        plot_batch_idx=trials_to_plot,
        plot_start_time=T_init,
        save_path=f'{model_dir}/{fpf_name}',
        iteration=iteration
        )

    return unique_fps


def fixed_points_finder(model, iteration=None):
    """
    Simulate the RNN to collect hidden states and find fixed points.
    """
    # np.random.seed(40)
    # torch.manual_seed(40)

    ## Simulate to collect hidden states ##
    # u_t: (trials, steps, neurons)
    # hidden_states: (trials, steps, neuron)
    u_t, hidden_states, thetas = prepare_state(model)

    if iteration is None:
        iteration='final'

    for fpf_name in fpf_names:
        # print(f"Running Fixed Point Analysis for {fpf_name}")
        unique_fps = analyze_fixed_points(model, u_t[:,int(T_init/dt+1),:], hidden_states, fpf_name, iteration=iteration)
        # print(f"Fixed points found: {len(unique_fps)}")

    if fpf_pca_bool:
        plot_F_vs_PCA(
            model.F.detach().cpu(),
            hidden_states[:,-1,:],
            thetas[:,0],
            pca_dir = f'{model_dir}/pca',
            iteration=iteration,
        )

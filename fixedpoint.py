import numpy as np
import ipdb


from FixedPoints.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from fixedpoint_utils import *

def analyze_fixed_points(model, input_states, hidden_states, fpf_name, epoch):
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
    if fpf_name == 'decode':
        start_t_idx = -2
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
        epoch=epoch
        )

    return unique_fps


def fixed_points_finder(model, epoch=None):
    """
    Simulate the RNN to collect hidden states and find fixed points.
    """
    np.random.seed(40)
    torch.manual_seed(40)

    ## Simulate to collect hidden states ##
    # u_t: (trials, steps, neurons)
    # hidden_states: (trials, steps, neuron)
    u_t, hidden_states, thetas = prepare_state(model)

    if epoch is None:
        epoch='final'

    for fpf_name in fpf_names:
        # print(f"Running Fixed Point Analysis for {fpf_name}")
        unique_fps = analyze_fixed_points(model, u_t[:,int(T_init/dt+1),:], hidden_states, fpf_name, epoch=epoch)
        # print(f"Fixed points found: {len(unique_fps)}")

    if fpf_pca_bool:
        plot_F_vs_PCA(
            model.F.detach().cpu(),
            hidden_states[:,-1,:],
            thetas[:,0],
            pca_dir = f'{model_dir}/pca',
            epoch=epoch,
        )

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import yaml
from scipy.stats import chi2

def plot_decode_trajectories(r_output, F, T_init, T_stimi, T_delay, dt, num_trials, result_dir):
    """
    Parameters:
      r_output: Neural network output, shape (total_trials, steps, neurons)
      F: Decoding matrix, shape (max_item_num*2, neurons). Only the first two dimensions are used as the decode point.
      T_init, T_stimi, T_delay: Duration (in seconds) of each experimental phase.
      dt: Time step (in seconds)
      num_trials: Number of randomly selected trials to plot

    Functionality:
      1. Compute the decoded points for each trial (only first 2 components), resulting in shape (trials, steps, 2).
      2. Randomly select a subset of trials and plot their decoded point trajectories.
      3. Use a scatter plot where the point color indicates the progression of time (derived using dt).
      4. Mark the boundary points on each trial corresponding to:
         - The start of the trial (initial location)
         - End of the initial phase (T_init)
         - End of the stimuli phase (T_init + T_stimi)
         - End of the delay phase (T_init + T_stimi + T_delay)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Compute the decoded points: shape -> (trials, steps, 2)
    decode_points = (np.matmul(r_output, F.T))[:, :, :2]
    
    total_trials = decode_points.shape[0]
    # Randomly select trial indices
    selected_indices = np.random.choice(total_trials, size=num_trials, replace=False)
    
    plt.figure(figsize=(10, 8))
    
    # Use label only for the first trial to avoid duplicate legend entries
    first_trial = True
    for trial_idx in selected_indices:
        traj = decode_points[trial_idx]  # Trajectory for the current trial, shape (steps, 2)
        steps = traj.shape[0]
        time_array = np.arange(steps) * dt  # Corresponding time array
        
        # Plot trajectory with scatter plot; color represents time (using 'viridis' colormap)
        sc = plt.scatter(traj[:, 0], traj[:, 1],
                         c=time_array, cmap='viridis', s=10,
                         label=f'Trial {trial_idx}' if first_trial else None)
        # 同时绘制连线，方便观察轨迹走向
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.5)
        
        # Calculate indices for phase boundaries (ensure indices are within trajectory length)
        phase0_idx = 0
        phase1_idx = int(T_init / dt)
        phase2_idx = int((T_init + T_stimi) / dt)
        phase3_idx = int((T_init + T_stimi + T_delay) / dt)
        phase4_idx = int((T_init + T_stimi + T_delay + T_decode) / dt) - 1
   
        # Mark the initial location (start of trial)
        if phase0_idx < steps:
            plt.scatter(traj[phase0_idx, 0], traj[phase0_idx, 1],
                        marker='s', color='magenta', s=50,
                        label='Initial Location' if first_trial else "")
        # Mark the end of the initial phase
        if phase1_idx < steps:
            plt.scatter(traj[phase1_idx, 0], traj[phase1_idx, 1],
                        marker='s', color='orange', s=50,
                        label='End of Init' if first_trial else "")
        # Mark the end of the stimuli phase
        if phase2_idx < steps:
            plt.scatter(traj[phase2_idx, 0], traj[phase2_idx, 1],
                        marker='s', color='cyan', s=50,
                        label='End of Stimuli' if first_trial else "")
        # Mark the end of the delay phase
        if phase3_idx < steps:
            plt.scatter(traj[phase3_idx, 0], traj[phase3_idx, 1],
                        marker='s', color='lime', s=50,
                        label='End of Delay' if first_trial else "")
        # Mark the end of the decode phase (final location)
        if phase4_idx < steps:
            plt.scatter(traj[phase4_idx, 0], traj[phase4_idx, 1],
                        marker='D', color='blue', s=50,
                        label='End of Decode' if first_trial else "")
        first_trial = False  # Only add legend labels once

    # Add a color bar to indicate time progression
    cbar = plt.colorbar(sc)
    cbar.set_label('Time (s)')
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Sampled Trajectories in the Decoding Plane')
    plt.legend()
    plt.grid(True)
    
    # Save the figure to the specified result directory
    image_path = f'{result_dir}/sample_trajectories.png'
    plt.savefig(image_path, dpi=300)
    plt.close()

def SNR_analysis(model):
    # Prepare the states
    u_t, r_output, thetas = prepare_state_snr(model)
    
    # 1. Get true orientation from the first trial and convert to a 2D point (unit circle)
    true_orientation = thetas[0, 0].item()
    true_point = [math.cos(true_orientation), math.sin(true_orientation)]
    
    step_threshold = int((T_init + T_stimi + T_delay) / dt)
    r_decode = r_output[:, step_threshold:, :].detach().numpy()  # (trials, steps, neuron)
    F = model.F.detach().cpu().numpy().squeeze()  # (max_item_num*2, neurons)
    
    # Decode points from the network (we only take the first 2 components)
    decode_points = (r_decode @ F.T)[:, :, :2]  # (trials, steps, 2)

    # 2. Calculate average signal: mean over both trials and steps (result is a 2D point)
    signal = decode_points.mean(axis=(0, 1))  # (2,)
    
    # 3. Calculate error: each decoded point minus the signal
    error = decode_points - signal  # (trials, steps, 2)
    
    # 4. Time averaged error: average error over steps for each trial
    time_avg_error = error.mean(axis=1)  # (trials, 2)
    
    # 5. Noise 1: flatten error (over both trials and steps) and compute Gaussian parameters
    error_flat = error.reshape(-1, 2)  # shape (trials*steps, 2)
    noise1_mean = error_flat.mean(axis=0)
    noise1_cov = np.cov(error_flat, rowvar=False)
    noise1_power = np.trace(noise1_cov)
    
    # 6. Noise 2: flatten the time averaged error (over trials) and compute Gaussian parameters
    noise2_mean = time_avg_error.mean(axis=0)
    noise2_cov = np.cov(time_avg_error, rowvar=False)
    noise2_power = np.trace(noise2_cov)
    
    # 7. Signal magnitude: the Euclidean norm of the signal vector
    signal_mag = np.linalg.norm(signal)
    
    # 8. SNR calculations: using the trace of the covariance as noise power
    SNR1 = (signal_mag ** 2) / noise1_power if noise1_power != 0 else np.inf
    SNR2 = (signal_mag ** 2) / noise2_power if noise2_power != 0 else np.inf

    # Convert SNR to dB
    SNR1_dB = 10 * np.log10(SNR1) if SNR1 not in [0, np.inf] else SNR1
    SNR2_dB = 10 * np.log10(SNR2) if SNR2 not in [0, np.inf] else SNR2
    '''
    Power of noise is to be carefully re-defined. Maybe only take certain direction.
    '''

    # Directory to save results
    snr_dir = f'{model_dir}/snr'
    if not os.path.exists(snr_dir):
        os.makedirs(snr_dir)
    
    plot_decode_trajectories(r_output, F, T_init, T_stimi, T_delay, dt, num_trials=1, result_dir=snr_dir)

    # -------- Plot 1: Non Time Averaged --------
    plt.figure(figsize=(8, 6))
    # Scatter plot of all decoded points (flatten trials and steps)
    plt.scatter(decode_points[..., 0].flatten(), decode_points[..., 1].flatten(),
                label='Decoded Points', marker=',', s=3, alpha=0.5)
    
    # Both lines (true orientation and signal) are drawn with length equal to signal magnitude
    true_line_end = [math.cos(true_orientation) * signal_mag,
                     math.sin(true_orientation) * signal_mag]
    plt.plot([0, true_line_end[0]], [0, true_line_end[1]],
             color='red', label='True Orientation', linestyle='-', linewidth=1.5)
    plt.plot([0, signal[0]], [0, signal[1]],
             color='green', label='Signal (Mean)', linestyle='-', linewidth=2)
    
    # Draw an ellipse to indicate Gaussian noise 1 (non-time-averaged)
    eigenvals1, eigenvecs1 = np.linalg.eig(noise1_cov)
    order1 = eigenvals1.argsort()[::-1]
    eigenvals1 = eigenvals1[order1]
    eigenvecs1 = eigenvecs1[:, order1]
    angle1 = np.degrees(np.arctan2(eigenvecs1[1, 0], eigenvecs1[0, 0]))
    chi2_val1 = chi2.ppf(0.95, 2)  # 95% confidence region, 2 sigma
    width1, height1 = 2 * np.sqrt(eigenvals1 * chi2_val1)
    ellipse1 = patches.Ellipse(xy=signal, width=width1, height=height1,
                               angle=angle1, edgecolor='purple', fc='None', lw=2,
                               label='Noise Ellipse')
    plt.gca().add_patch(ellipse1)
    
    # Add text annotations in the upper left corner
    textstr1 = '\n'.join((
        f'Signal Strength: {signal_mag:.2f}',
        f'Noise Strength: {noise1_power:.2f}',
        f'SNR: {SNR1_dB:.2f} dB'))
    plt.gca().text(0.05, 0.05, textstr1, transform=plt.gca().transAxes, fontsize=8,
                   verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    plt.axvline(0, color='black', linewidth=1, linestyle='-')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('SNR - Non Time Averaged')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    image_path1 = f'{snr_dir}/non_time_averaged.png'
    plt.savefig(image_path1, dpi=300)
    plt.close()
    
    # -------- Plot 2: Time Averaged --------
    # Average decoded points over time steps for each trial
    decode_points_time = decode_points.mean(axis=1)  # (trials, 2)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(decode_points_time[:, 0], decode_points_time[:, 1],
                label='Time Averaged Decoded Points', marker='o', s=10, alpha=0.7)
    
    # Draw lines for true orientation and signal using the norm of the time averaged signal
    plt.plot([0, true_line_end[0]], [0, true_line_end[1]],
             color='red', label='True Orientation', linestyle='-', linewidth=1.5)
    plt.plot([0, signal[0]], [0, signal[1]],
             color='green', label='Signal (Mean)', linestyle='-', linewidth=2)
    
    # Draw an ellipse to indicate Gaussian noise (time-averaged)
    eigenvals2, eigenvecs2 = np.linalg.eig(noise2_cov)
    order2 = eigenvals2.argsort()[::-1]
    eigenvals2 = eigenvals2[order2]
    eigenvecs2 = eigenvecs2[:, order2]
    angle2 = np.degrees(np.arctan2(eigenvecs2[1, 0], eigenvecs2[0, 0]))
    chi2_val2 = chi2.ppf(0.95, 2)
    width2, height2 = 2 * np.sqrt(eigenvals2 * chi2_val2)
    ellipse2 = patches.Ellipse(xy=signal, width=width2, height=height2,
                               angle=angle2, edgecolor='purple', fc='None', lw=2,
                               label='Noise Ellipse')
    plt.gca().add_patch(ellipse2)
    
    # Add text annotations for time-averaged data
    textstr2 = '\n'.join((
        f'Signal Strength: {signal_mag:.2f}',
        f'Noise Strength: {noise2_power:.2f}',
        f'SNR: {SNR2_dB:.2f} dB'))
    plt.gca().text(0.05, 0.05, textstr2, transform=plt.gca().transAxes, fontsize=8,
                   verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    plt.axvline(0, color='black', linewidth=1, linestyle='-')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('SNR - Time Averaged')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    image_path2 = f'{snr_dir}/time_averaged.png'
    plt.savefig(image_path2, dpi=300)
    plt.close()
    
    # -------- Save results to a YAML file --------
    results = {
        'true_orientation': true_orientation,
        'true_point': true_point,
        'signal': signal.tolist(),
        'signal_magnitude': signal_mag.item(),
        'noise1': {
            'mean': noise1_mean.tolist(),
            'covariance': noise1_cov.tolist(),
            'power': noise1_power.item(),
        },
        'noise2': {
            'mean': noise2_mean.tolist(),
            'covariance': noise2_cov.tolist(),
            'power': noise2_power.item(),
        },
        'SNR1': SNR1.item(),
        'SNR2': SNR2.item(),
        'SNR1_dB': SNR1_dB if SNR1_dB in [np.inf, -np.inf] else float(SNR1_dB),
        'SNR2_dB': SNR2_dB if SNR2_dB in [np.inf, -np.inf] else float(SNR2_dB),
    }
    yaml_path = f'{snr_dir}/results.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(results, f)
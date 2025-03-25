import numpy as np
import ipdb


from FixedPoints.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from fixedpoint_utils import *

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

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import yaml
from scipy.stats import chi2
import matplotlib.patches as patches
from matplotlib import cm

def plot_decode_trajectories(r_output, F, T_init, T_stimi, T_delay, T_decode, dt, num_trials, result_dir, true_orientation=None, num_neurons=5):
    """
    Parameters:
      r_output: Neural network output, shape (total_trials, steps, neurons)
      F: Decoding matrix, shape (max_item_num*2, neurons). Only the first two dimensions are used as the decode point.
      T_init, T_stimi, T_delay, T_decode: Durations (in seconds) of each experimental phase.
      dt: Time step (in seconds)
      num_trials: Number of randomly selected trials to plot trajectories
      true_orientation: Optional; a 2D point representing the true orientation.
      num_neurons: Number of neurons to randomly select (from one of the chosen trials) for activity plotting.
      
    Functionality:
      1. Compute the decoded points for each trial (only first 2 components), resulting in shape (trials, steps, 2).
      2. Randomly select a subset of trials and plot their decoded point trajectories.
         - Scatter plot: color indicates time (using the 'viridis' colormap).
         - Boundary markers for phase transitions: start, end of init, stimuli, delay, and decode.
      3. For one of the selected trials, randomly select several neurons and plot their firing rate (Hz) vs time.
         - Shaded backgrounds indicate the different experimental phases.
      4. Save both plots to the provided result directory.
    """

    # -------- Plot 1: Decoded Trajectories --------
    # Compute the decoded points: shape -> (trials, steps, 2)
    decode_points = (np.matmul(r_output, F.T))[:, :, :2]
    
    total_trials = decode_points.shape[0]
    # Randomly select trial indices for trajectory plotting
    selected_indices = np.random.choice(total_trials, size=num_trials, replace=False)
    
    plt.figure(figsize=(10, 8))
    first_trial = True  # flag to add legend labels only once
    
    for trial_idx in selected_indices:
        traj = decode_points[trial_idx]  # Trajectory for the current trial, shape (steps, 2)
        steps = traj.shape[0]
        time_array = np.arange(steps) * dt  # Corresponding time array
        
        # Plot trajectory with scatter; color indicates time progression
        sc = plt.scatter(traj[:, 0], traj[:, 1],
                         c=time_array, cmap='viridis', s=10 if first_trial else None)
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.5)
        
        # Calculate indices for phase boundaries (ensuring indices do not exceed steps)
        phase0_idx = 0
        phase1_idx = int(T_init / dt)
        phase2_idx = int((T_init + T_stimi) / dt)
        phase3_idx = int((T_init + T_stimi + T_delay) / dt)
        phase4_idx = int((T_init + T_stimi + T_delay + T_decode) / dt) - 1

        # Mark the initial location (start of trial)
        if phase0_idx < steps:
            plt.scatter(traj[phase0_idx, 0], traj[phase0_idx, 1],
                        marker='s', color='magenta', s=50,
                        label=f'Initial Location (t = {0:.2f} ms)' if first_trial else "")
        # Mark the end of the initial phase
        if phase1_idx < steps:
            t_phase1 = phase1_idx * dt
            plt.scatter(traj[phase1_idx, 0], traj[phase1_idx, 1],
                        marker='s', color='orange', s=50,
                        label=f'End of Init (t = {t_phase1:.2f} ms)' if first_trial else "")
        # Mark the end of the stimuli phase
        if phase2_idx < steps:
            t_phase2 = phase2_idx * dt
            plt.scatter(traj[phase2_idx, 0], traj[phase2_idx, 1],
                        marker='s', color='cyan', s=50,
                        label=f'End of Stimuli (t = {t_phase2:.2f} ms)' if first_trial else "")
        # Mark the end of the delay phase
        if phase3_idx < steps:
            t_phase3 = phase3_idx * dt
            plt.scatter(traj[phase3_idx, 0], traj[phase3_idx, 1],
                        marker='s', color='lime', s=50,
                        label=f'End of Delay (t = {t_phase3:.2f} ms)' if first_trial else "")
        # Mark the end of the decode phase
        if phase4_idx < steps:
            t_phase4 = phase4_idx * dt
            plt.scatter(traj[phase4_idx, 0], traj[phase4_idx, 1],
                        marker='D', color='blue', s=50,
                        label=f'End of Decode (t = {t_phase4:.2f} ms)' if first_trial else "")
        first_trial = False  # only add labels once

    if true_orientation is not None:
        # Use half the maximum norm of the trajectories from the selected trials for display
        true_orientation_length = max(np.max(np.linalg.norm(decode_points[i], axis=1)) for i in selected_indices) * 0.5
        true_line_end = [math.cos(true_orientation) * true_orientation_length,
                         math.sin(true_orientation) * true_orientation_length]
        plt.plot([0, true_line_end[0]], [0, true_line_end[1]],
                 color='red', label='True Orientation', linestyle='-', linewidth=1.5)
        plt.scatter(true_line_end[0], true_line_end[1],
                    marker='*', color='red', s=100)
    
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    plt.axvline(0, color='black', linewidth=1, linestyle='-')
    cbar = plt.colorbar(sc)
    cbar.set_label('Time (s)')
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Sampled Trajectories in the Decoding Plane')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Save the trajectory plot
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    traj_image_path = os.path.join(result_dir, 'sample_trajectories.png')
    plt.savefig(traj_image_path, dpi=300)
    plt.close()

    # -------- Plot 2: Neuron Activity vs. Time --------
    # For demonstration, take the first selected trial from the trajectory selection.
    trial_idx_for_neurons = selected_indices[0]
    activity = r_output[trial_idx_for_neurons]  # shape (steps, neurons)
    steps = activity.shape[0]
    time = np.arange(steps) * dt  # time vector in seconds

    total_neurons = activity.shape[1]
    # Randomly select several neurons
    selected_neurons = np.random.choice(total_neurons, size=num_neurons, replace=False)
    
    plt.figure(figsize=(10, 6))
    
    # Shade the different experimental phases
    # Phase boundaries in seconds:
    t_phase0 = 0
    t_phase1 = T_init
    t_phase2 = T_init + T_stimi
    t_phase3 = T_init + T_stimi + T_delay
    t_phase4 = T_init + T_stimi + T_delay + T_decode
    
    # Use axvspan for shading (only label the first occurrence to avoid duplicate legend entries)
    plt.axvspan(t_phase0, t_phase1, color='lightgray', alpha=0.5, label='Init Phase')
    plt.axvspan(t_phase1, t_phase2, color='lightblue', alpha=0.5, label='Stimuli Phase')
    plt.axvspan(t_phase2, t_phase3, color='lightgreen', alpha=0.5, label='Delay Phase')
    plt.axvspan(t_phase3, t_phase4, color='navajowhite', alpha=0.5, label='Decode Phase')
    
    # Plot firing rate vs. time for each selected neuron.
    for neuron in selected_neurons:
        plt.plot(time, activity[:, neuron], label=f'Neuron {neuron}', linewidth=1.5)
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title(f'Neuron Activity vs Time (Trial {trial_idx_for_neurons})')
    plt.legend()
    plt.grid(True)
    
    # Save the neuron activity plot
    neuron_image_path = os.path.join(result_dir, 'neuron_activity.png')
    plt.savefig(neuron_image_path, dpi=300)
    plt.close()
    

def process_snr_item(model, snr_item_num, T_init, T_stimi, T_delay, T_decode, dt, model_dir):
    """
    Processes the SNR analysis for a single item number.
    Prepares states, computes signal and noise statistics, generates plots,
    saves results to disk, and returns the SNR values in dB.
    """
    # Prepare the states
    u_t, r_output, thetas = prepare_state_snr(model, snr_item_num)
    
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
    snr_dir = f'{model_dir}/snr/{snr_item_num}item'
    if not os.path.exists(snr_dir):
        os.makedirs(snr_dir)

    # Plot sampled trajectories on the Fr plane
    plot_decode_trajectories(r_output, F, T_init, T_stimi, T_delay, T_decode, dt,
                             num_trials=1, result_dir=snr_dir, true_orientation=true_orientation)
    
    
    # -------- Plot 1: Non Time Averaged --------
    plt.figure(figsize=(8, 6))
    plt.scatter(decode_points[..., 0].flatten(), decode_points[..., 1].flatten(),
                label='Decoded Points', marker=',', s=3, alpha=0.3)
    
    true_line_end = [math.cos(true_orientation) * signal_mag,
                     math.sin(true_orientation) * signal_mag]
    plt.plot([0, true_line_end[0]], [0, true_line_end[1]],
             color='red', label='True Orientation', linestyle='-', linewidth=1.5)
    plt.plot([0, signal[0]], [0, signal[1]],
             color='green', label='Signal (Mean)', linestyle='-', linewidth=2)
    
    # Draw an ellipse for Gaussian noise 1 (non-time-averaged)
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
    plt.axis('equal')

    image_path1 = f'{snr_dir}/non_time_averaged.png'
    plt.savefig(image_path1, dpi=300)
    plt.close()
    
    # -------- Plot 2: Time Averaged --------
    # Average decoded points over time steps for each trial
    decode_points_time = decode_points.mean(axis=1)  # (trials, 2)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(decode_points_time[:, 0], decode_points_time[:, 1],
                label='Time Averaged Decoded Points', marker='o', s=10, alpha=0.3)
    plt.plot([0, true_line_end[0]], [0, true_line_end[1]],
             color='red', label='True Orientation', linestyle='-', linewidth=1.5)
    plt.plot([0, signal[0]], [0, signal[1]],
             color='green', label='Signal (Mean)', linestyle='-', linewidth=2)
    
    # Draw an ellipse for Gaussian noise (time-averaged)
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
    plt.axis('equal')

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
    
    return SNR1_dB, SNR2_dB

def SNR_analysis(model):
    """
    Runs SNR analysis over a range of item numbers.
    Calls process_snr_item for each item, collects SNR values,
    and plots SNR vs. item number for both non-time-averaged and time-averaged cases.
    """
    item_numbers = range(1, max_item_num + 1)
    SNR1_dB_list = []
    SNR2_dB_list = []
    
    for snr_item_num in item_numbers:
        SNR1_dB, SNR2_dB = process_snr_item(model, snr_item_num, T_init, T_stimi, T_delay, T_decode, dt, model_dir)
        SNR1_dB_list.append(SNR1_dB)
        SNR2_dB_list.append(SNR2_dB)
    
    # Plot SNR vs. item number for both metrics
    plt.figure(figsize=(8, 6))
    plt.plot(item_numbers, SNR1_dB_list, marker='o', label='SNR1 - Not Time Averaged')
    plt.plot(item_numbers, SNR2_dB_list, marker='s', label='SNR2 - Time Averaged')
    plt.xlabel('Item Number', fontsize=12)
    plt.ylabel('SNR (dB)', fontsize=12)
    plt.title('SNR vs. Item Number', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    image_path = f'{model_dir}/snr/snr_vs_items.png'
    plt.savefig(image_path, dpi=300)
    plt.close()

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import yaml
from scipy.stats import chi2
import matplotlib.patches as patches
from matplotlib import cm
import seaborn as sns
import pandas as pd

from rnn import *
from config import *
from train import *
from utils import *

def prepare_state_snr(model, snr_item_num, T_init, T_stimi, T_delay, T_decode, dt):
    snr_trial_num = 1000 # i.e. repeated times of the same 
    
    # np.random.seed(39)
    # torch.manual_seed(39)

    input_presence = torch.zeros(max_item_num, device=device)
    input_presence[0] = 1  # Always include the first item

    if snr_item_num > 1:
        # Sample the remaining indices from 1 to max_item_num-1
        random_indices = torch.randperm(max_item_num - 1, device=device)[:(snr_item_num - 1)] + 1
        input_presence[random_indices.tolist()] = 1
        
    input_thetas = (torch.rand(max_item_num , device=device) * 2 * torch.pi) - torch.pi

    # print(f"SNR theta: {input_thetas}")
    # print(f"SNR presence: {input_presence}")

    # repeat same input snr_trial_num times，
    input_presence = input_presence.repeat(snr_trial_num, 1)
    input_thetas = input_thetas.repeat(snr_trial_num, 1)

    u_t = generate_input(
        presence=input_presence,
        theta=input_thetas,
        input_strength=input_strength,
        noise_level=0.0,
        T_init=T_init,
        T_stimi=T_stimi,
        T_delay=T_delay,
        T_decode=T_decode,
        dt=dt,
    )
    
    r_output, _ = model(u_t, r0=None)  # (trial, steps, neuron)
    observed_r_output = model.observed_r(r_output)

    return u_t.detach().cpu().numpy(), observed_r_output.detach().cpu().numpy(), input_thetas.detach().cpu().numpy()

def plot_input_n_activity(
    r: np.ndarray,
    u: np.ndarray,
    W: np.ndarray,
    B: np.ndarray,
    T_init: int,
    T_stimi: int,
    T_delay: int,
    T_decode: int,
    result_dir: str,
    dt: float
) -> None:
    """
    Compute and save:
      1) A 2x3 grid of histograms showing statistics (mean and std) of neuron activity,
         recurrent input, and external input during specific task periods.
      2) A time series plot of neuron activity for a single randomly selected trial.

    Args:
        r (np.ndarray): Firing rates with shape (trials, steps, neurons).
        u (np.ndarray): External inputs with shape (trials, steps, input_dim).
        W (np.ndarray): Recurrent weight matrix of shape (neurons, neurons).
        B (np.ndarray): Input weight matrix of shape (neurons, input_dim).
        T_init (int): Number of timesteps before stimulus onset.
        T_stimi (int): Duration of stimulus interval in timesteps.
        T_delay (int): Duration of delay interval in timesteps.
        T_decode (int): Duration of decoding interval in timesteps.
        result_dir (str): Directory to save output figures.
        dt (float): Duration of each timestep in seconds (for x-axis).

    Returns:
        None
    """

    # Time window indices
    delay_start = int((T_init + T_stimi)/dt)
    delay_end   = delay_start + int(T_delay/dt)
    stim_start  = int(T_init/dt)
    stim_end    = stim_start + int(T_stimi/dt)

    # Reshape r_output to (steps, trials, neurons) for time-major operations
    trials, steps, neurons = r.shape

    # Compute recurrent input: W @ r
    R_rec = np.dot(r.reshape(-1, neurons), W.T).reshape(steps, trials, neurons)

    # Compute external input: B @ u
    R_ext = np.dot(u.reshape(-1, u.shape[2]), B.T).reshape(steps, trials, neurons)

    # Compute statistics over selected time periods
    mean_r_delay   = r[delay_start:delay_end].mean(axis=(0, 1))
    std_r_delay    = r[delay_start:delay_end].std(axis=(0, 1))
    mean_rec_delay = R_rec[delay_start:delay_end].mean(axis=(0, 1))
    std_rec_delay  = R_rec[delay_start:delay_end].std(axis=(0, 1))
    mean_ext_stim  = R_ext[stim_start:stim_end].mean(axis=(0, 1))
    std_ext_stim   = R_ext[stim_start:stim_end].std(axis=(0, 1))

    # Organize all stats into a dataframe for easier handling
    data_dict = {
        'Mean firing rate (delay)': mean_r_delay,
        'Std firing rate (delay)': std_r_delay,
        'Mean recurrent input (delay)': mean_rec_delay,
        'Std recurrent input (delay)': std_rec_delay,
        'Mean external input (stim)': mean_ext_stim,
        'Std external input (stim)': std_ext_stim
    }

    # Convert to DataFrame
    df = pd.DataFrame({
        'firing_mean': data_dict['Mean firing rate (delay)'],
        'firing_std': data_dict['Std firing rate (delay)'],
        'recur_mean': data_dict['Mean recurrent input (delay)'],
        'recur_std': data_dict['Std recurrent input (delay)'],
        'ext_mean': data_dict['Mean external input (stim)'],
        'ext_std': data_dict['Std external input (stim)'],
    })

    # Create and save each jointplot
    os.makedirs(result_dir, exist_ok=True)
    plot_specs = [
        ('firing_mean', 'firing_std', 'Firing Rate (Delay)'),
        ('recur_mean', 'recur_std', 'Recurrent Input (Delay)'),
        ('ext_mean', 'ext_std', 'External Input (Stim)')
    ]

    for x_col, y_col, title in plot_specs:
        g = sns.jointplot(
            data=df, x=x_col, y=y_col,
            kind='scatter', height=5, marginal_kws=dict(bins=30, fill=True)
        )
        g.figure.suptitle(title, fontsize=14)
        g.figure.tight_layout()
        g.figure.subplots_adjust(top=0.95)  # Ensure title isn't clipped
        g.savefig(os.path.join(result_dir, f'{x_col}_vs_{y_col}_jointplot.png'))
        plt.close(g.figure)

    # -------- Plot 2: Neuron Activity vs. Time (single trial) --------
    trial_idx = np.random.choice(trials)
    activity = r[trial_idx]  # shape (steps, neurons)
    time = np.arange(steps) * dt

    selected_neurons = np.random.choice(activity.shape[1], size=5, replace=False)

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Shade the four phases
    t0 = 0
    t1 = T_init
    t2 = (T_init + T_stimi)
    t3 = (T_init + T_stimi + T_delay)
    t4 = (T_init + T_stimi + T_delay + T_decode)
    ax2.axvspan(t0, t1, color='lightgray', alpha=0.5, label='Init Phase')
    ax2.axvspan(t1, t2, color='lightblue', alpha=0.5, label='Stimuli Phase')
    ax2.axvspan(t2, t3, color='lightgreen', alpha=0.5, label='Delay Phase')
    ax2.axvspan(t3, t4, color='navajowhite', alpha=0.5, label='Decode Phase')

    # Plot selected neurons
    for n in selected_neurons:
        ax2.plot(time, activity[:, n], linewidth=1.5, label=f'Neuron {n}')
    ax2.set(
        xlabel='Time (s)',
        ylabel='Firing Rate (Hz)',
        title=f'Neuron Activity vs. Time (Trial {trial_idx})'
    )
    ax2.legend()
    ax2.grid(True)

    neuron_path = os.path.join(result_dir, 'neuron_activity.png')
    fig2.savefig(neuron_path, dpi=300)
    plt.close(fig2)

    return

def plot_decode_trajectories(r_output, F, T_init, T_stimi, T_delay, T_decode, dt, num_trials, result_dir, true_orientation=None):
    """
    Parameters:
      r_output: Neural network output, shape (total_trials, steps, neurons)
      F: Decoding matrix, shape (max_item_num*2, neurons). Only the first two dimensions are used as the decode point.
      T_init, T_stimi, T_delay, T_decode: Durations (in seconds) of each experimental phase.
      dt: Time step (in seconds)
      num_trials: Number of randomly selected trials to plot trajectories
      true_orientation: Optional; a 2D point representing the true orientation.
      
    Functionality:
      1. Compute the decoded points for each trial (only first 2 components), resulting in shape (trials, steps, 2).
      2. Randomly select a subset of trials and plot their decoded point trajectories.
         - Scatter plot: color indicates time (using the 'viridis' colormap).
         - Boundary markers for phase transitions: start, end of init, stimuli, delay, and decode.
      3. For one of the selected trials, randomly select several neurons and plot their firing rate (Hz) vs time.
         - Shaded backgrounds indicate the different experimental phases.
      4. Save both plots to the provided result directory.
    """

    # -------- Plot Decoded Trajectories --------
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

def process_snr_item(model, snr_item_num, T_init, T_stimi, T_delay, T_decode, dt, model_dir):
    """
    Processes the SNR analysis for a single item number.
    Prepares states, computes signal and noise statistics, generates plots,
    saves results to disk, and returns the SNR values in dB.
    """
    # Prepare the states
    u_t, observed_r_output, thetas = prepare_state_snr(model, snr_item_num, T_init, T_stimi, T_delay, T_decode, dt)
    
    # 1. Get true orientation from the first trial and convert to a 2D point (unit circle)
    true_orientation = thetas[0, 0].item()
    true_point = [math.cos(true_orientation), math.sin(true_orientation)]
    
    step_threshold = int((T_init + T_stimi + T_delay) / dt)
    r_decode = observed_r_output[:, step_threshold:, :]  # (trials, steps, neuron)
    F = model.F.detach().cpu().numpy()  # (max_item_num*2, neurons)
    W = (model.W * model.dales_sign.view(1, -1)).detach().cpu().numpy()
    B = model.B.detach().cpu().numpy()
    
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
    plot_decode_trajectories(observed_r_output, F, T_init, T_stimi, T_delay, T_decode, dt, num_trials=1,
                             result_dir=snr_dir, true_orientation=true_orientation)
    plot_input_n_activity(observed_r_output, u_t, W, B, T_init, T_stimi, T_delay, T_decode, snr_dir, dt)
    
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
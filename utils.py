import torch
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from math import sqrt

from rnn import *
from config import *

def generate_target(presence, theta, stimuli_present=True):
    max_item_num = presence.shape[1]
    u_0 = torch.zeros(presence.size(0), 2 * max_item_num, device=device)
    for i in range(max_item_num):
        u_0[:, 2 * i] = presence[:, i] * ( torch.cos(theta[:, i]) )
        u_0[:, 2 * i + 1] = presence[:, i] * ( torch.sin(theta[:, i]) )
    u_t = u_0 * (1 if stimuli_present else 0) 
    return u_t

def generate_input(presence, theta, input_strength=40, noise_level=0.0, T_init=0, T_stimi=400, T_delay=0, T_decode=800, dt=10):
    """
    Generate a 3D input tensor of shape (steps, num_trials, 3 * max_item_num) without loops.

    Args:
        presence: (num_trials, max_item_num) binary tensor indicating presence of items.
        theta: (num_trials, max_item_num) tensor of angles.
        noise_level: Noise level to be added to theta.
        T_init, T_stimi, T_delay, T_decode: Timings for each phase (in ms).
        dt: Time step size (in ms).

    Returns:
        u_t_stack: (num_trials, steps, 3 * max_item_num) tensor of input vectors over time.
    """
    # Total simulation time and steps
    T_simul = T_init + T_stimi + T_delay + T_decode
    steps = int(T_simul / dt)
    num_trials, max_item_num = presence.shape

    # Add noise to theta
    theta_noisy = theta + noise_level * torch.randn(
        (num_trials, max_item_num), device=device
    )
    theta_noisy = (theta_noisy + torch.pi) % (2 * torch.pi) - torch.pi

    # Compute the 2D positions (cos and sin components) for all items
    cos_theta = torch.cos(theta_noisy)  # (num_trials, max_item_num)
    sin_theta = torch.sin(theta_noisy)  # (num_trials, max_item_num)

    # Stack cos and sin into a single tensor along the last dimension
    # Then multiply by presence to zero-out absent items
    if positive_input:
        u_0 = ( torch.stack((
                    1 + cos_theta / sqrt(2) + sin_theta / sqrt(6), 
                    1 - cos_theta / sqrt(2) + sin_theta / sqrt(6),
                    1 - 2 * sin_theta / sqrt(6),
                ), dim=-1) ) * presence.unsqueeze(-1) # (num_trials, max_item_num, 3)
    else:
        u_0 = ( torch.stack((cos_theta, sin_theta), dim=-1) ) * presence.unsqueeze(-1) # (num_trials, max_item_num, 2)

    # Reshape to match output shape (combine cos and sin into one dimension)
    u_0 = u_0.view(num_trials, -1)  # (num_trials, 3 * max_item_num)

    # Create a mask for stimuli presence at each time step
    stimuli_present_mask = (torch.arange(steps, device=device) * dt >= T_init) & \
                           (torch.arange(steps, device=device) * dt < T_init + T_stimi)
    stimuli_present_mask = stimuli_present_mask.float().unsqueeze(-1).unsqueeze(-1)  # (steps, 1, 1)

    # Apply the stimuli mask
    u_t_stack = u_0.unsqueeze(0) * stimuli_present_mask  # (steps, num_trials, 3 * max_item_num)

    # Swap dimensions 0 and 1 to get (num_trials, steps, 3 * max_item_num)
    u_t_stack = u_t_stack.transpose(0, 1) * input_strength / max_item_num

    return u_t_stack

def plot_results(decoded_orientations_dict):
    plt.figure(figsize=(5,4))
    time_steps = torch.tensor([step * dt for step in range(simul_steps)], device=device)

    # Plot response lines and target curves
    for angle_target, decoded_orientations in decoded_orientations_dict.items():
        line, = plt.plot(time_steps.cpu(), decoded_orientations, marker='o', linestyle='-', markersize=3)
        plt.axhline(y=angle_target, color=line.get_color(), linestyle='--')

    response_legend = plt.Line2D([0], [0], color='blue', marker='o', linestyle='-', markersize=4, label='Response')
    target_legend = plt.Line2D([0], [0], color='blue', linestyle='--', label='Target')
    stimulus_period_legend = plt.Line2D([0], [0], color='orange', lw=5, alpha=0.4, label="Stimulus period")
    decode_period_legend = plt.Line2D([0], [0], color='green', lw=5, alpha=0.3, label="Decoding period")

    plt.axvspan(T_init, T_stimi + T_init, color='orange', alpha=0.15)
    plt.axvspan(T_stimi + T_init + T_delay, T_simul, color='green', alpha=0.1)
    plt.title('Decoded Memory Orientations vs. Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Orientation (radians)')
    plt.grid(True)
    plt.legend(handles=[stimulus_period_legend, decode_period_legend, response_legend, target_legend], loc='upper right')
    # plt.show()

def plot_training_history(error_per_iteration, error_std_per_iteration, activation_per_iteration):
    """
    Plots training curves for error (with error bar) and activation penalty.
    """
    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(5, 4))
    iterations = np.arange(1, len(error_per_iteration) + 1)

    # Plot Error curve on the left y-axis with error bars
    error_color = 'blue'
    ax1.plot(iterations, error_per_iteration, label="Error", color=error_color, marker='o', markersize=2)
    ax1.fill_between(
        iterations,
        np.array(error_per_iteration) - np.array(error_std_per_iteration),
        np.array(error_per_iteration) + np.array(error_std_per_iteration),
        color=error_color,
        alpha=0.2,
        label="Error (± std)"
    )
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('Error Loss', color=error_color)
    ax1.tick_params(axis='y', labelcolor=error_color)
    # ax1.set_yscale('log')
    # ax1.set_ylim(1e-3, 4)
    ax1.grid(True)

    # Calculate and annotate the average of the last 50 Error values
    ax1.axhline(y=error_per_iteration[-1], color=error_color, linestyle='--', alpha=0.7)
    ax1.annotate(
        f"{error_per_iteration[-1]:.3f}",  # Format the annotation to 3 decimal places
        xy=(iterations[-1], error_per_iteration[-1]),  # Position it at the last iteration's error value
        xytext=(5, 0), textcoords="offset points",  # Offset slightly for clarity
        color=error_color, fontsize=9, fontweight="bold"
    )

    # Create a second y-axis for Activation Penalty
    activation_color = 'orange'
    ax2 = ax1.twinx()
    ax2.plot(iterations, activation_per_iteration, label="Activation", color=activation_color, marker='o', markersize=2)
    ax2.set_ylabel('Ave Firing Rate (Hz)', color=activation_color)
    ax2.tick_params(axis='y', labelcolor=activation_color)

    # Calculate and annotate the average of the last 50 Activation Penalty values
    ax2.axhline(y=activation_per_iteration[-1], color=activation_color, linestyle='--', alpha=0.7)
    ax2.annotate(
        f"{activation_per_iteration[-1]:.3f}Hz",  # Format the annotation to 3 decimal places
        xy=(iterations[-1], activation_per_iteration[-1]),  # Position it at the last iteration's error value
        xytext=(5, 0), textcoords="offset points",  # Offset slightly for clarity
        color=activation_color, fontsize=9, fontweight="bold"
    )
    
    plt.title('Training Error and Activation vs Iteration')
    fig.tight_layout()
    # plt.show()

def plot_group_training_history(iterations, group_errors, group_stds, group_activ, item_num, logging_period):
    """
    Plots the error and error bars for each group across iterations, with each group in a separate subplot,
    and annotates the end value for each group.

    Parameters:
    - iterations: epch number
    - group_errors: List of lists, where each sublist contains mean errors for a group over iterations.
    - group_stds: List of lists, where each sublist contains standard deviations of errors for a group over iterations.
    - group_activ: List of lists, where each sublist contains average activation penalties for a group over iterations.
    - item_num: List of the number of items in each group (e.g., [1, 2, 3, 4] for 4 groups).
    - logging_period: The interval of iterations at which the errors are recorded.
    """
    num_groups = len(group_errors)

    fig, axes = plt.subplots(num_groups, 1, figsize=(8,2*num_groups), sharex=True)
    if num_groups == 1:  # if only one group exists
        axes = [axes]

    colormap = plt.cm.tab10  # Change colormap if desired

    err_color = colormap(1 % 10)  # Ensure color reuse for more than 10 groups
    activ_color = colormap(4 % 10)  # Ensure color reuse for more than 10 groups

    # Plot each group in its subplot
    for i, (errors, stds, activ) in enumerate(zip(group_errors, group_stds, group_activ)):
        errors = np.array(errors)
        stds = np.array(stds)

        # Plot errors
        line_error, = axes[i].plot(
            iterations, errors, 
            label="Error",
            color=err_color,
            # marker='o', markersize=3
        )
        # Plot error bars
        axes[i].fill_between(
            iterations,
            errors - stds,
            errors + stds,
            color=err_color,
            alpha=0.2,
            label="Error ± std"
        )

        axes[i].set_ylabel("Absolute Error (rad)",color=err_color)
        axes[i].set_title(f"{item_num[i]} item. Loss and Activation vs Iteration")
        axes[i].tick_params(axis='y', labelcolor=err_color)
        axes[i].grid(True)
        axes[i].set_ylim(0, 2)


        # Plot activations
        ax2 = axes[i].twinx()
        line_activ, = ax2.plot(
            iterations, activ,
            label="Activation",
            color=activ_color,
            # marker='o', markersize=
        )
        ax2.set_ylabel('Ave Firing Rate (Hz)', color=activ_color)
        ax2.tick_params(axis='y', labelcolor=activ_color)

        # Add the end value annotation for activation
        ax2.annotate(
            f"{activ[-1]:.3f} Hz",  # Format the annotation to 3 decimal places
            xy=(iterations[-1], activ[-1]),  # Position it at the last iteration's error value
            xytext=(-20, -20), textcoords="offset points",  # Offset slightly for clarity
            color=activ_color, fontsize=10, fontweight="bold",
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.3')  # White background
        )
        # Add the end value annotation
        axes[i].annotate(
            f"{errors[-1]:.3f} rad",  # Format the annotation to 3 decimal places
            xy=(iterations[-1], errors[-1]),  # Position it at the last iteration's error value
            xytext=(-20, -15), textcoords="offset points",  # Offset slightly for clarity
            color=err_color, fontsize=10, fontweight="bold",
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.3')  # White background
        )


    # Create a global legend
    fig.legend(
        handles=[
            line_error,  # Line for Error
            plt.Line2D([0], [0], color=err_color, alpha=0.2, lw=10, label="Error ± std"),  # Legend for Error ± std
            line_activ,  # Line for Activation
        ],
        labels=["Error",  "Error ± std", "Activation"],
        loc='lower center',
        ncol=3  # Updated number of columns to fit the new entry
    )

    # Set the x-axis label for the last subplot
    axes[-1].set_xlabel("Iterations")

    file_path = os.path.join(model_dir, f'training_history.png')
    plt.savefig(file_path, dpi=300)

    ## Summary Plot ##
    final_errors = [errors[-1] for errors in group_errors]
    final_activations = [activ[-1] for activ in group_activ]

    # Save the final plot with two different y-axes
    final_plot_path = os.path.join(model_dir, 'error_activ_vs_itemnum.png')
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Plot error on the primary y-axis (ax1)
    line1, = ax1.plot(item_num, final_errors, marker='o', color=err_color, label='Last Error (rad)')
    ax1.set_xlabel('Item Number')
    ax1.set_ylabel('Last Error (rad)', color=err_color)
    ax1.tick_params(axis='y', labelcolor=err_color)
    ax1.grid(True)
    ax1.set_ylim(bottom=0)  # Set lower limit to zero for ax1

    # Create a secondary y-axis for activation
    ax2 = ax1.twinx()
    line2, = ax2.plot(item_num, final_activations, marker='s', color=activ_color, label='Activation (Hz)')
    ax2.set_ylabel('Activation (Hz)', color=activ_color)
    ax2.tick_params(axis='y', labelcolor=activ_color)
    ax2.set_ylim(bottom=0)  # Set lower limit to zero for ax2

    # Combine the legends from both axes
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc='lower center', ncol=2)
    
    plt.title('Error and Activation vs. Item Number')
    plt.savefig(final_plot_path, dpi=300)
    plt.close()

def save_model_and_history(model, history, model_dir, model_name="model", history_name="training_history.yaml"):
    """Saves the model state and training history to the specified directory."""
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    if use_scripted_model:
        model_path = f'{model_dir}/models/scripted_{model_name}.pt'
        model.save(model_path)
    else:
        model_path = f'{model_dir}/models/{model_name}.pth'
        torch.save(model.state_dict(), model_path)

    # Save training history
    history_path = f'{model_dir}/{history_name}'
    with open(history_path, 'w') as f:
        yaml.dump(history, f, default_flow_style=False)

def load_model_and_history(model, model_dir, model_name="model", history_name="training_history.yaml"):
    """Loads the model state and training history from the specified directory."""
    history_path = f'{model_dir}/{history_name}'

    if os.path.exists(history_path):
        # Load model
        with open(history_path, 'r') as f:
            history = yaml.safe_load(f)
        iteration = history['iterations'][-1]
        model_name = f'model_iteration{iteration}'

        if use_scripted_model:
            model_path = f'{model_dir}/models/scripted_{model_name}.pt'
            if os.path.exists(model_path):
                model = torch.jit.load(model_path, map_location=device)
                model.device = device
        else:
            model_path = f'{model_dir}/models/{model_name}.pth'
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
                model.device = device
    else:
        history = None

    return model, history

def plot_weights(model):
    """
    Plots the weight matrices B (input to neurons), W (recurrent), and F (neurons to output) side by side.

    Args:
        model (torch.nn.Module): RNN model containing weight matrices B, W, and F.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Convert tensors to NumPy
    B_np = model.B.detach().cpu().numpy()
    F_np = model.F.detach().cpu().numpy()

    effective_W = model.W * model.dales_sign.view(1, -1)
    effective_W_np = effective_W.detach().cpu().numpy()

    # fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 2.5, 1]})

    # Plot B (Input to Neurons)
    im0 = axes[0].imshow(B_np, cmap="seismic", vmin=-np.max(np.abs(B_np)), vmax=np.max(np.abs(B_np)))
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=-0.04)
    axes[0].set_title("Input-to-Neurons (B)", fontsize=14)
    axes[0].set_xlabel(f"Inputs ({B_np.shape[1]})", fontsize=12)
    axes[0].set_ylabel(f"Neurons ({B_np.shape[0]})", fontsize=12)

    # Plot W (Recurrent Weights)
    im1 = axes[1].imshow(effective_W_np, cmap="seismic", vmin=-np.max(np.abs(effective_W_np)), vmax=np.max(np.abs(effective_W_np)))
    fig.colorbar(im1, ax=axes[1], fraction=0.023, pad=0.04)
    axes[1].set_title("Recurrent Weights (W)", fontsize=14)
    axes[1].set_xlabel(f"Neurons ({effective_W_np.shape[1]})", fontsize=12)
    axes[1].set_ylabel(f"Neurons ({effective_W_np.shape[0]})", fontsize=12)

    # Plot F (Neurons to Output)
    im2 = axes[2].imshow(F_np.T, cmap="seismic", vmin=-np.max(np.abs(F_np)), vmax=np.max(np.abs(F_np)))
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=-0.04)
    axes[2].set_title("Neurons-to-Output (F.T)", fontsize=14)
    axes[2].set_xlabel(f"Outputs ({F_np.shape[0]})", fontsize=12)
    axes[2].set_ylabel(f"Neurons ({F_np.shape[1]})", fontsize=12)

    # Remove ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_anchor("C")

    # Save figure
    save_path = os.path.join(model_dir, f"weights.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"All weight matrices plot saved at: {save_path}")

def plot_error_dist(model):
    """
    Plot the distribution of decoding errors for different item numbers after training.
    This function visualizes the model's performance by comparing decoded angles with input angles.
    """
    num_trials = 6000
    if max_item_num == 1:
        item_num = [1]
    elif max_item_num == 2:
        item_num = [1,2]
    elif max_item_num < 8:
        item_num = list(range(1, max_item_num+1, 2))
    else:
        item_num = [8, 4, 2, 1]

    # Split num_trials into len(item_num) groups
    trials_per_group = num_trials // len(item_num)
    remaining_trials = num_trials % len(item_num)
    trial_counts = [trials_per_group + (1 if i < remaining_trials else 0) for i in range(len(item_num))]
    
    # Generate random presence indicators
    # device = next(model.parameters()).device  # Use model device
    input_presence = torch.zeros(num_trials, max_item_num, device=device, requires_grad=False)
    start_index = 0
    for i, count in enumerate(trial_counts):
        end_index = start_index + count
        one_hot_indices = torch.stack([
            torch.randperm(max_item_num, device=device)[:item_num[i]] for _ in range(count)
        ])
        input_presence[start_index:end_index] = input_presence[start_index:end_index].scatter(1, one_hot_indices, 1)
        start_index = end_index
    
    # Generate random input angles
    input_thetas = (torch.rand(num_trials, max_item_num, device=device) * 2 * torch.pi) - torch.pi
    
    # Generate input tensor
    u_t = generate_input(input_presence, input_thetas, input_strength, ILC_noise, T_init, T_stimi, T_delay, T_decode, dt)

    # Run simulation and slice the output
    r_output, _ = model(u_t, r0=None)  # (trial, steps, neuron)
    step_threshold = int((T_init + T_stimi + T_delay) / dt)
    r_decode = r_output[:, step_threshold:, :].transpose_(0, 1).clone()  # (steps_for_loss, trial, neuron)
    del r_output    # delete r_output as it can be up to 8GB
    torch.cuda.empty_cache()

    # Decode the output
    u_hat = model.decode(r_decode.reshape(-1, num_neurons)).reshape(r_decode.shape[0], num_trials, -1)
    u_hat_reshaped = u_hat.view(u_hat.shape[0], u_hat.shape[1], -1, 2)
    
    # Compute angles diff
    cos_thetas = u_hat_reshaped[..., 0]  # (steps, trials, max_items)
    sin_thetas = u_hat_reshaped[..., 1]  # (steps, trials, max_items)
    decoded_thetas = torch.atan2(sin_thetas, cos_thetas)  # (steps, trials, max_items)
    angular_diff = (input_thetas - decoded_thetas.mean(dim=0) + torch.pi) % (2 * torch.pi) - torch.pi  # (trials,items)

    # Plot error distribution
    plt.figure(figsize=(6, 5))
    x_values = np.linspace(-np.pi, np.pi, 100)
    
    start_index = 0
    for i, count in enumerate(trial_counts):
        end_index = start_index + count
        mask = input_presence[start_index:end_index].bool()
        sliced_angular_diff = angular_diff[start_index:end_index]
        err = sliced_angular_diff[mask].detach().cpu().numpy()
          
        hist, bins = np.histogram(err, bins=x_values, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.plot(bin_centers, hist, label=f'{item_num[i]} item(s)')

        start_index = end_index
    
    plt.xlim(-np.pi, np.pi)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
            [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    plt.xlabel('Angular Error (radians)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title('Distribution of Decoding Errors')
    plt.ylim(bottom=0)
    
    # Save the plot
    file_path = os.path.join(model_dir, 'error_distrib.png')
    plt.savefig(file_path, dpi=300)
    plt.close()

    print(f"Error Distribution Plot saved at: {file_path}")
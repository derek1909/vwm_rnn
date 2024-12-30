import torch
import matplotlib.pyplot as plt
import os
import json
import numpy as np

from rnn import *
from config import *

def generate_target(presence, theta, noise_level=0.0, stimuli_present=True, alpha=0):
    theta = theta + noise_level * torch.randn_like(theta, device=device)
    max_item_num = presence.shape[1]
    u_0 = torch.zeros(presence.size(0), 2 * max_item_num, device=device)
    for i in range(max_item_num):
        u_0[:, 2 * i] = presence[:, i] * ( torch.cos(theta[:, i]) + alpha )
        u_0[:, 2 * i + 1] = presence[:, i] * ( torch.sin(theta[:, i]) + alpha )
    u_t = u_0 * (1 if stimuli_present else 0) 
    return u_t

def generate_input(presence, theta, noise_level=0.0, T_init=0, T_stimi=400, T_delay=0, T_decode=800, dt=10, alpha=0):
    """
    Generate a 3D input tensor of shape (steps, num_trials, 2 * max_item_num) without loops.

    Args:
        presence: (num_trials, max_item_num) binary tensor indicating presence of items.
        theta: (num_trials, max_item_num) tensor of angles.
        noise_level: Noise level to be added to theta.
        T_init, T_stimi, T_delay, T_decode: Timings for each phase (in ms).
        dt: Time step size (in ms).

    Returns:
        u_t_stack: (num_trials, steps, 2 * max_item_num) tensor of input vectors over time.
    """
    # Total simulation time and steps
    T_simul = T_init + T_stimi + T_delay + T_decode
    steps = int(T_simul / dt)
    num_trials, max_item_num = presence.shape

    # Add noise to theta
    theta_noisy = theta.unsqueeze(0) + noise_level * torch.randn(
        (steps, num_trials, max_item_num), device=device
    )

    # Compute the 2D positions (cos and sin components) for all items
    cos_theta = torch.cos(theta_noisy/2)  # (steps, num_trials, max_item_num)
    sin_theta = torch.sin(theta_noisy/2)  # (steps, num_trials, max_item_num)

    # Stack cos and sin into a single tensor along the last dimension
    # Then multiply by presence to zero-out absent items
    u_0 = ( torch.stack((cos_theta, sin_theta), dim=-1) + alpha ) * presence.unsqueeze(0).unsqueeze(-1) # (steps, num_trials, max_item_num, 2)


    # Reshape to match output shape (combine cos and sin into one dimension)
    u_0 = u_0.view(steps, num_trials, -1)  # (steps, num_trials, 2 * max_item_num)

    # Create a mask for stimuli presence at each time step
    stimuli_present_mask = (torch.arange(steps, device=device) * dt >= T_init) & \
                           (torch.arange(steps, device=device) * dt < T_init + T_stimi)
    stimuli_present_mask = stimuli_present_mask.float().unsqueeze(-1).unsqueeze(-1)  # (steps, 1, 1)

    # Apply the stimuli mask
    u_t_stack = u_0 * stimuli_present_mask  # (steps, num_trials, 2 * max_item_num)

    # Swap dimensions 0 and 1 to get (num_trials, steps, 2 * max_item_num)
    u_t_stack = u_t_stack.transpose(0, 1)

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

def plot_training_history(error_per_epoch, error_std_per_epoch, activation_per_epoch):
    """
    Plots training curves for error (with error bar) and activation penalty.
    """
    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(5, 4))
    epochs = np.arange(1, len(error_per_epoch) + 1)

    # Plot Error curve on the left y-axis with error bars
    error_color = 'blue'
    ax1.plot(epochs, error_per_epoch, label="Error", color=error_color, marker='o', markersize=2)
    ax1.fill_between(
        epochs,
        np.array(error_per_epoch) - np.array(error_std_per_epoch),
        np.array(error_per_epoch) + np.array(error_std_per_epoch),
        color=error_color,
        alpha=0.2,
        label="Error (± std)"
    )
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error Loss', color=error_color)
    ax1.tick_params(axis='y', labelcolor=error_color)
    # ax1.set_yscale('log')
    # ax1.set_ylim(1e-3, 4)
    ax1.grid(True)

    # Calculate and annotate the average of the last 50 Error values
    ax1.axhline(y=error_per_epoch[-1], color=error_color, linestyle='--', alpha=0.7)
    ax1.annotate(
        f"{error_per_epoch[-1]:.3f}",  # Format the annotation to 3 decimal places
        xy=(epochs[-1], error_per_epoch[-1]),  # Position it at the last epoch's error value
        xytext=(5, 0), textcoords="offset points",  # Offset slightly for clarity
        color=error_color, fontsize=9, fontweight="bold"
    )

    # Create a second y-axis for Activation Penalty
    activation_color = 'orange'
    ax2 = ax1.twinx()
    ax2.plot(epochs, activation_per_epoch, label="Activation", color=activation_color, marker='o', markersize=2)
    ax2.set_ylabel('Ave Firing Rate (Hz)', color=activation_color)
    ax2.tick_params(axis='y', labelcolor=activation_color)

    # Calculate and annotate the average of the last 50 Activation Penalty values
    ax2.axhline(y=activation_per_epoch[-1], color=activation_color, linestyle='--', alpha=0.7)
    ax2.annotate(
        f"{activation_per_epoch[-1]:.3f}Hz",  # Format the annotation to 3 decimal places
        xy=(epochs[-1], activation_per_epoch[-1]),  # Position it at the last epoch's error value
        xytext=(5, 0), textcoords="offset points",  # Offset slightly for clarity
        color=activation_color, fontsize=9, fontweight="bold"
    )
    
    plt.title('Training Error and Activation vs Epoch')
    fig.tight_layout()
    # plt.show()

def plot_group_training_history(group_errors, group_stds, group_activ, item_num, logging_period):
    """
    Plots the error and error bars for each group across epochs, with each group in a separate subplot,
    and annotates the end value for each group.

    Parameters:
    - group_errors: List of lists, where each sublist contains mean errors for a group over epochs.
    - group_stds: List of lists, where each sublist contains standard deviations of errors for a group over epochs.
    - group_activ: List of lists, where each sublist contains average activation penalties for a group over epochs.
    - item_num: List of the number of items in each group (e.g., [1, 2, 3, 4] for 4 groups).
    - logging_period: The interval of epochs at which the errors are recorded.
    """
    num_groups = len(group_errors)
    epochs = np.arange(1, len(group_errors[0]) + 1) * logging_period

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
            epochs, errors, 
            label="Error",
            color=err_color,
            # marker='o', markersize=3
        )
        # Plot error bars
        axes[i].fill_between(
            epochs,
            errors - stds,
            errors + stds,
            color=err_color,
            alpha=0.2,
            label="Error ± std"
        )

        axes[i].set_ylabel("Error",color=err_color)
        axes[i].set_title(f"{item_num[i]} item. Loss and Activation vs Epoch")
        axes[i].tick_params(axis='y', labelcolor=err_color)
        axes[i].grid(True)


        # Plot activations
        ax2 = axes[i].twinx()
        line_activ, = ax2.plot(
            epochs, activ,
            label="Activation",
            color=activ_color,
            # marker='o', markersize=
        )
        ax2.set_ylabel('Ave Firing Rate (Hz)', color=activ_color)
        ax2.tick_params(axis='y', labelcolor=activ_color)

        # Add the end value annotation for activation
        ax2.axhline(y=activ[-1], color=activ_color, linestyle='--', alpha=0.7)
        ax2.annotate(
            f"{activ[-1]:.3f} Hz",  # Format the annotation to 3 decimal places
            xy=(epochs[-1], activ[-1]),  # Position it at the last epoch's error value
            xytext=(-20, -20), textcoords="offset points",  # Offset slightly for clarity
            color=activ_color, fontsize=10, fontweight="bold",
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.3')  # White background
        )
        # Add the end value annotation
        axes[i].axhline(y=errors[-1], color=err_color, linestyle='--', alpha=0.7)
        axes[i].annotate(
            f"{errors[-1]:.3f}",  # Format the annotation to 3 decimal places
            xy=(epochs[-1], errors[-1]),  # Position it at the last epoch's error value
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
    axes[-1].set_xlabel("Epochs")

    file_path = os.path.join(model_dir, f'training_history_{rnn_name}.png')
    plt.savefig(file_path, dpi=300, bbox_inches="tight")

def save_model_and_history(model, history, model_dir, model_name="model_path.pth", history_name="training_history.json"):
    """Saves the model state and training history to the specified directory."""
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)

    # Save training history
    history_path = os.path.join(model_dir, history_name)
    with open(history_path, 'w') as f:
        json.dump(history, f)

def load_model_and_history(model, model_dir, model_name="model_path.pth", history_name="training_history.json"):
    """Loads the model state and training history from the specified directory."""
    model_path = os.path.join(model_dir, model_name)
    history_path = os.path.join(model_dir, history_name)

    # Load model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))

    # Load history
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = {
            "error_per_epoch": [],  # Overall mean error per epoch
            "error_std_per_epoch": [],  # std of overall error per epoch
            "activation_per_epoch": [],
            "group_errors": [[] for _ in item_num],  # List to store errors for each 'set size' group
            "group_std": [[] for _ in item_num],  # List to store std of errors for each group
            "group_activ": [[] for _ in item_num],
        }

    return model, history
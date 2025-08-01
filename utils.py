"""
Utility functions for the visual working memory RNN.

This module provides essential utility functions for data generation, visualization,
model saving/loading, and result analysis. It includes functions for generating
input stimuli, target representations, plotting training history, and managing
model checkpoints.

Author: Derek Jinyu Dong
Date: 2024-2025
"""

import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import math
import os, gc, yaml

from rnn import *
from config import *

def generate_target(presence, theta, stimuli_present=True):
    """Generate target output vectors from orientations and presence masks."""
    max_item_num = presence.shape[1]
    u_0 = torch.zeros(presence.size(0), 2 * max_item_num, device=device)
    
    # Convert orientations to cosine/sine pairs for each item
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
        # Transform to positive input space using 3D encoding
        u_0 = ( torch.stack((
                    1 + cos_theta / math.sqrt(2) + sin_theta / math.sqrt(6), 
                    1 - cos_theta / math.sqrt(2) + sin_theta / math.sqrt(6),
                    1 - 2 * sin_theta / math.sqrt(6),
                ), dim=-1) ) * presence.unsqueeze(-1) # (num_trials, max_item_num, 3)
    else:
        # Standard 2D cosine/sine encoding
        u_0 = ( torch.stack((cos_theta, sin_theta), dim=-1) ) * presence.unsqueeze(-1) # (num_trials, max_item_num, 2)

    # Reshape to match output shape (combine cos and sin into one dimension)
    u_0 = u_0.view(num_trials, -1)  # (num_trials, 3 * max_item_num)

    # Create temporal mask: stimuli only present during stimulus period
    stimuli_present_mask = (torch.arange(steps, device=device) * dt >= T_init) & \
                           (torch.arange(steps, device=device) * dt < T_init + T_stimi)
    stimuli_present_mask = stimuli_present_mask.float().unsqueeze(-1).unsqueeze(-1)  # (steps, 1, 1)

    # Apply temporal mask and normalize by number of items
    u_t_stack = u_0.unsqueeze(0) * stimuli_present_mask  # (steps, num_trials, 3 * max_item_num)
    u_t_stack = u_t_stack.transpose(0, 1) * input_strength / max_item_num  # (num_trials, steps, 3 * max_item_num)

    return u_t_stack

def plot_results(decoded_orientations_dict):
    """Plot decoded orientations over time with target references."""
    plt.figure(figsize=(5,4))
    time_steps = torch.tensor([step * dt for step in range(simul_steps)], device=device)

    # Plot response trajectories and target horizontal lines
    for angle_target, decoded_orientations in decoded_orientations_dict.items():
        line, = plt.plot(time_steps.cpu(), decoded_orientations, marker='o', linestyle='-', markersize=3)
        plt.axhline(y=angle_target, color=line.get_color(), linestyle='--')

    # Create legend elements
    response_legend = plt.Line2D([0], [0], color='blue', marker='o', linestyle='-', markersize=4, label='Response')
    target_legend = plt.Line2D([0], [0], color='blue', linestyle='--', label='Target')
    stimulus_period_legend = plt.Line2D([0], [0], color='orange', lw=5, alpha=0.4, label="Stimulus period")
    decode_period_legend = plt.Line2D([0], [0], color='green', lw=5, alpha=0.3, label="Decoding period")

    # Highlight task periods
    plt.axvspan(T_init, T_stimi + T_init, color='orange', alpha=0.15)        # Stimulus
    plt.axvspan(T_stimi + T_init + T_delay, T_simul, color='green', alpha=0.1)  # Decode
    plt.title('Decoded Memory Orientations vs. Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Orientation (radians)')
    plt.grid(True)
    plt.legend(handles=[stimulus_period_legend, decode_period_legend, response_legend, target_legend], loc='upper right')
    # plt.show()

def plot_training_history(error_per_iteration, error_std_per_iteration, activation_per_iteration):
    """Plot training curves for error (with std) and average firing rate."""
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(5, 4))
    iterations = np.arange(1, len(error_per_iteration) + 1)

    # Plot error curve with confidence bands
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
    ax1.grid(True)

    # Annotate final error value
    ax1.axhline(y=error_per_iteration[-1], color=error_color, linestyle='--', alpha=0.7)
    ax1.annotate(
        f"{error_per_iteration[-1]:.3f}",
        xy=(iterations[-1], error_per_iteration[-1]),
        xytext=(5, 0), textcoords="offset points",
        color=error_color, fontsize=9, fontweight="bold"
    )

    # Plot activation on secondary y-axis
    activation_color = 'orange'
    ax2 = ax1.twinx()
    ax2.plot(iterations, activation_per_iteration, label="Activation", color=activation_color, marker='o', markersize=2)
    ax2.set_ylabel('Ave Firing Rate (Hz)', color=activation_color)
    ax2.tick_params(axis='y', labelcolor=activation_color)

    # Annotate final activation value
    ax2.axhline(y=activation_per_iteration[-1], color=activation_color, linestyle='--', alpha=0.7)
    ax2.annotate(
        f"{activation_per_iteration[-1]:.3f}Hz",
        xy=(iterations[-1], activation_per_iteration[-1]),
        xytext=(5, 0), textcoords="offset points",
        color=activation_color, fontsize=9, fontweight="bold"
    )
    
    plt.title('Training Error and Activation vs Iteration')
    fig.tight_layout()
    # plt.show()

def plot_group_training_history(iterations, group_errors, group_stds, group_activ, item_num, logging_period):
    """Plot training history separately for each set size with dual y-axes."""
    num_groups = len(group_errors)

    fig, axes = plt.subplots(num_groups, 1, figsize=(8,2*num_groups), sharex=True)
    if num_groups == 1:  # Handle single group case
        axes = [axes]

    colormap = plt.cm.tab10
    err_color = colormap(1 % 10)    # Blue for errors
    activ_color = colormap(4 % 10)  # Orange for activations

    # Plot each set size in separate subplot
    for i, (errors, stds, activ) in enumerate(zip(group_errors, group_stds, group_activ)):
        errors = np.array(errors)
        stds = np.array(stds)

        # Plot error with confidence bands
        line_error, = axes[i].plot(
            iterations, errors, 
            label="Error",
            color=err_color,
        )
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

        # Plot firing rates on secondary axis
        ax2 = axes[i].twinx()
        line_activ, = ax2.plot(
            iterations, activ,
            label="Activation",
            color=activ_color,
        )
        ax2.set_ylabel('Ave Firing Rate (Hz)', color=activ_color)
        ax2.tick_params(axis='y', labelcolor=activ_color)

        # Annotate final values with background boxes
        ax2.annotate(
            f"{activ[-1]:.3f} Hz",
            xy=(iterations[-1], activ[-1]),
            xytext=(-20, -20), textcoords="offset points",
            color=activ_color, fontsize=10, fontweight="bold",
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.3')
        )
        axes[i].annotate(
            f"{errors[-1]:.3f} rad",
            xy=(iterations[-1], errors[-1]),
            xytext=(-20, -15), textcoords="offset points",
            color=err_color, fontsize=10, fontweight="bold",
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.3')
        )


    # Create global legend at bottom
    fig.legend(
        handles=[
            line_error,  # Line for Error
            plt.Line2D([0], [0], color=err_color, alpha=0.2, lw=10, label="Error ± std"),  # Legend for Error ± std
            line_activ,  # Line for Activation
        ],
        labels=["Error",  "Error ± std", "Activation"],
        loc='lower center',
        ncol=3
    )

    # Set x-axis label for bottom subplot
    axes[-1].set_xlabel("Iterations")

    # Save training history plot
    file_path = os.path.join(model_dir, f'training_history.png')
    plt.savefig(file_path, dpi=300)

    def plot_error_activ_vs_itemnum(errors, activs, plot_path):
        """Helper function to plot final values vs set size."""
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        # Plot error on primary y-axis
        line1, = ax1.plot(item_num, errors, marker='o', color=err_color, label='Error (rad)')
        ax1.set_xlabel('Item Number')
        ax1.set_ylabel('Error (rad)', color=err_color)
        ax1.tick_params(axis='y', labelcolor=err_color)
        ax1.grid(True)
        ax1.set_ylim(bottom=0)

        # Plot activation on secondary y-axis
        ax2 = ax1.twinx()
        line2, = ax2.plot(item_num, activs, marker='s', color=activ_color, label='Activation (Hz)')
        ax2.set_ylabel('Activation (Hz)', color=activ_color)
        ax2.tick_params(axis='y', labelcolor=activ_color)
        ax2.set_ylim(bottom=0)

        # Combined legend
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        fig.legend(lines, labels, loc='lower center', ncol=2)
        
        plt.title('Error and Activation vs. Item Number')
        plt.savefig(plot_path, dpi=300)
        plt.close()
    
    # Generate comparison plots for first and final training states
    final_errors = [errors[-1] for errors in group_errors]
    final_activations = [activ[-1] for activ in group_activ]
    final_plot_path = os.path.join(model_dir, 'Final_error_activ_vs_itemnum.png')
    plot_error_activ_vs_itemnum(final_errors, final_activations, final_plot_path)
    
    first_errors = [errors[0] for errors in group_errors]
    first_activations = [activ[0] for activ in group_activ]
    first_plot_path = os.path.join(model_dir, 'First_error_activ_vs_itemnum.png')
    plot_error_activ_vs_itemnum(first_errors, first_activations, first_plot_path)
    

def save_model_and_history(model, history, model_dir, model_name="model", history_name="training_history.yaml"):
    """Save model state dict and training history to specified directory."""
    os.makedirs(model_dir, exist_ok=True)

    # Save model checkpoint
    if use_scripted_model:
        model_path = f'{model_dir}/models/scripted_{model_name}.pt'
        model.save(model_path)
    else:
        model_path = f'{model_dir}/models/{model_name}.pth'
        torch.save(model.state_dict(), model_path)

    # Save training metrics as YAML
    history_path = f'{model_dir}/{history_name}'
    with open(history_path, 'w') as f:
        yaml.dump(history, f, default_flow_style=False)

def load_model_and_history(model, model_dir, model_name="model", history_name="training_history.yaml"):
    """Load model checkpoint and training history from directory."""
    history_path = f'{model_dir}/{history_name}'

    if os.path.exists(history_path):
        # Load training history to get latest iteration
        with open(history_path, 'r') as f:
            history = yaml.safe_load(f)
        iteration = history['iterations'][-1]
        model_name = f'model_iteration{iteration}'

        # Load model checkpoint
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

import torch
import matplotlib.pyplot as plt
import os
import json
import numpy as np

from rnn import *
from config import *

def generate_input(presence, theta, noise_level=0.0, stimuli_present=True):
    theta = theta + noise_level * torch.randn_like(theta)
    max_item_num = presence.shape[1]
    u_0 = torch.zeros(presence.size(0), 2 * max_item_num)
    for i in range(max_item_num):
        u_0[:, 2 * i] = presence[:, i] * torch.cos(theta[:, i])
        u_0[:, 2 * i + 1] = presence[:, i] * torch.sin(theta[:, i])
    u_t = u_0 * (1 if stimuli_present else 0) 
    return u_t

def evaluate(model, angle_targets):
    """
    Evaluates the model's decoded orientations for given target angles.

    Args:
        model: The RNN memory model.
        angle_targets (list): List of target angles for evaluation.å
    Returns:å
        dict: A dictionary mapping each angle target to its decoded orientations over time.
    """
    decoded_orientations_dict = {}
    presence = torch.tensor([1,]).reshape(1, max_item_num)

    for angle_target in angle_targets:
        decoded_orientations_after = []
        theta = torch.tensor([angle_target,]).reshape(1, max_item_num)
        r = torch.zeros(1, num_neurons)

        for step in range(simul_steps):
            time = step * dt
            u_t = generate_input(presence, theta, noise_level=encode_noise, stimuli_present=(T_init < time < T_stimi + T_init))
            r = model(r, u_t)
            decoded_memory = decode(model.F, r)
            decoded_memory = decoded_memory.view(decoded_memory.size(0), -1, 2)
            orientation = torch.atan2(decoded_memory[:, :, 1], decoded_memory[:, :, 0])
            decoded_orientations_after.append(orientation[0, 0].item())

        decoded_orientations_dict[angle_target] = decoded_orientations_after

    return decoded_orientations_dict

def plot_results(decoded_orientations_dict):
    plt.figure(figsize=(5,4))
    time_steps = torch.tensor([step * dt for step in range(simul_steps)])

    # Plot response lines and target curves
    for angle_target, decoded_orientations in decoded_orientations_dict.items():
        line, = plt.plot(time_steps, decoded_orientations, marker='o', linestyle='-', markersize=3)
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

def plot_training_curves(error_per_epoch, activation_penalty_per_epoch):

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(5, 4))
    epochs = range(1, len(error_per_epoch) + 1)

    # Plot Error curve on the left y-axis
    ax1.plot(epochs, error_per_epoch, label="Error", color='blue', marker='o', markersize=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # Calculate and annotate the average of the last 20 Error values
    avg_error = np.mean(error_per_epoch[-50:])
    ax1.axhline(y=avg_error, color='blue', linestyle='--', alpha=0.7)
    ax1.text(len(epochs), avg_error, f"End: {avg_error:.3f}", color='blue', ha='left', va='center')

    # Create a second y-axis for Activation Penalty
    ax2 = ax1.twinx()
    ax2.plot(epochs, activation_penalty_per_epoch, label="Activation Penalty", color='orange', marker='o', markersize=2)
    ax2.set_ylabel('Activation Penalty', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Calculate and annotate the average of the last 20 Activation Penalty values
    avg_penalty = np.mean(activation_penalty_per_epoch[-50:])
    ax2.axhline(y=avg_penalty, color='orange', linestyle='--', alpha=0.7)
    ax2.text(len(epochs), avg_penalty, f"End: {avg_penalty:.3f}", color='orange', ha='left', va='center')

    plt.title('Training Error and Activation Penalty vs Epoch')
    fig.tight_layout()
    # plt.show()

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
        model.load_state_dict(torch.load(model_path))
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load history
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = {"error_per_epoch": [], "activation_penalty_per_epoch": []}

    return model, history
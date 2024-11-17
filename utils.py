import torch
import matplotlib.pyplot as plt
from rnn import *
from config import *
from train import *

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
    plt.figure(figsize=(10, 6))
    time_steps = torch.tensor([step * dt for step in range(simul_steps)])
    for angle_target, decoded_orientations in decoded_orientations_dict.items():
        line, = plt.plot(time_steps, decoded_orientations, marker='o', linestyle='-')
        plt.axhline(y=angle_target, color=line.get_color(), linestyle='--')

    plt.axvspan(T_init, T_stimi + T_init, color='orange', alpha=0.3)
    plt.title('Decoded Memory Orientations vs. Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Orientation (radians)')
    plt.grid(True)
    plt.legend(["Response", "Target"])
    plt.show()

def plot_training_curves(error_per_epoch, activation_penalty_per_epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(error_per_epoch, label="Error", marker='o')
    plt.plot(activation_penalty_per_epoch, label="Activation Penalty", marker='x')
    plt.title('Training Error and Activation Penalty vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.show()
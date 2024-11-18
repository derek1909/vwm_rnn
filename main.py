import torch
import os
import json

from rnn import *
from config import *
from train import *
from utils import *


if __name__ == "__main__":
    model = RNNMemoryModel(max_item_num, num_neurons, tau, dt, process_noise)

    # Load model and history if training from a previous checkpoint
    if not train_from_scratch:
        model, history = load_model_and_history(model, model_dir)
    else:
        history = None  # Start fresh

    # Train the model
    if train_rnn:
        history = train(model, model_dir, history)

    # Plot training curves
    if history:
        plot_training_curves(history["error_per_epoch"], history["error_std_per_epoch"], history["activation_per_epoch"])

    # # Plot a few trials
    # angle_targets = [-1.9, -0.1, 0.6, 1.8]
    # decoded_orientations = evaluate(model, angle_targets)
    # plot_results(decoded_orientations)

    plt.show()
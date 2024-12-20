import torch
import os
import json

from rnn import *
from config import *
from train import *
from utils import *
from fixedpoint import fixed_points_finder


if __name__ == "__main__":
    model = RNNMemoryModel(max_item_num, num_neurons, tau, dt, process_noise).to(device)

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
        plot_group_training_history(history["group_errors"], history["group_std"], history["group_activ"], item_num)

    if find_fixed_points:
        fixed_points_finder(model)

    # plt.show()
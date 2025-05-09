# import torch
# import os
# import json
# import argparse

from rnn import *
from config import *
from train import *
from utils import *
from fixedpoint import fixed_points_finder, SNR_analysis

if __name__ == "__main__":
    model = RNNMemoryModel(max_item_num, num_neurons, dt, tau_min, tau_max, spike_noise_type, 
                           spike_noise_factor, saturation_firing_rate, device, positive_input, dales_law)
    if use_scripted_model:
        model = torch.jit.script(model)

    # Load model and history if training from a previous checkpoint
    if load_history:
        model, history = load_model_and_history(model, model_dir)
    else:
        history = None  # Start fresh

    # Train the model
    if train_rnn:
        history = train(model, model_dir, history)

    # Fixed Point Analysis
    if fpf_bool:
        print(f"Running final Fixed Point Analysis...")
        fixed_points_finder(model)

    with torch.no_grad():
        # Plot weights
        if plot_weights_bool:
            plot_weights(model)

        # Plot error distribution plots
        if error_dist_bool:
            plot_error_dist(model)

        # Plot training curves
        if history:
            plot_group_training_history(history["iterations"], history["group_errors"], history["group_std"], history["group_activ"], item_num, logging_period)

        if SNR_analy_bool:
            print(f"Running Signal to Noise Ratio Analysis...")
            SNR_analysis(model)

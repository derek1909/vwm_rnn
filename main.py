"""
Main execution script for the visual working memory RNN.

This module provides the main entry point for training and analyzing the 
biologically plausible RNN model. It orchestrates model training, fixed point 
analysis, and various neural analysis methods including SNR analysis, error 
distribution analysis, and mixed selectivity analysis.

Author: Derek Jinyu Dong
Date: 2024-2025
"""

# import torch
# import os
# import json
# import argparse

from rnn import *
from config import *
from train import *
from utils import *
from analysis.fixedpoint import fixed_points_finder
from analysis.snr_analysis import SNR_analysis
from analysis.error_dist_analysis import error_dist_analysis
from analysis.mixed_selectivity import mixed_selectivity_analysis
from analysis.dn_analysis import divisive_normalisation_analysis

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

    # Plot error distribution plots
    if error_dist_bool:
        print(f"Running Error Distribution Analysis...")
        error_dist_analysis(model)

    with torch.no_grad():
        print(f"Running Divisive Normalisation Analysis (set size = 1)...")
        divisive_normalisation_analysis(model)

        # Plot weights
        if plot_weights_bool:
            plot_weights(model)

        # Plot training curves
        if history:
            plot_group_training_history(history["iterations"], history["group_errors"], history["group_std"], history["group_activ"], item_num, logging_period)

        if snr_analy_bool and (spike_noise_factor>0):
            print(f"Running Signal to Noise Ratio Analysis...")
            SNR_analysis(model)

        if mixed_selec_bool:
            print(f"Running Mixed Selectivity Analysis (set size = 1)...")
            mixed_selectivity_analysis(model)
    

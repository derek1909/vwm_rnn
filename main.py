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


import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from rnn import *
from config import *
from train import *
from utils import *
from analysis.fixedpoint import fixed_points_finder
from analysis.snr_analysis import SNR_analysis
from analysis.error_dist_analysis import error_dist_analysis
from analysis.mixed_selectivity import mixed_selectivity_analysis
from analysis.dn_analysis import divisive_normalisation_analysis


def main_worker(rank, local_rank, world_size):

    # torchrun sets MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Set device for each process
    torch.cuda.set_device(local_rank)
    local_device = f'cuda:{local_rank}'

    # Build model and move to local GPU
    model = RNNMemoryModel(max_item_num, num_neurons, dt, tau_min, tau_max, spike_noise_type, 
                           final_spike_noise_factor, saturation_firing_rate, local_device, positive_input, dales_law)
    model = model.to(local_device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])


    # Only the main process (rank 0) creates directories, copies files, and prints info
    if rank == 0:
        import shutil
        os.makedirs(model_dir, exist_ok=True)
        destination_path = os.path.join(model_dir, os.path.basename(config_path))
        if os.path.realpath(config_path) != os.path.realpath(destination_path):
            shutil.copyfile(config_path, destination_path)
        print(f"rnn_name: {rnn_name}")
        print(f"Model directory: {model_dir}")
        print(f"Using DistributedDataParallel on {world_size} GPUs.")

    # Load model and training history if needed
    if load_history:
        model, history = load_model_and_history(model, model_dir, device=local_device)
    else:
        history = None

    # Training
    if train_rnn:
        # print("[Main] Multi-stage training (always enabled). Using multi_stage_train().")
        history = train(model, model_dir, history, rank=rank, world_size=world_size)

    # Only the main process (rank 0) runs analysis and plotting
    if rank == 0:
        if fpf_bool:
            print(f"Running final Fixed Point Analysis...")
            fixed_points_finder(model.module)

        if error_dist_bool:
            print(f"Running Error Distribution Analysis...")
            error_dist_analysis(model.module)

        with torch.no_grad():
            print(f"Running Divisive Normalisation Analysis (set size = 1)...")
            divisive_normalisation_analysis(model.module)

            if plot_weights_bool:
                plot_weights(model.module)

            if history:
                plot_group_training_history(history["iterations"], history["group_errors"], history["group_std"], history["group_activ"], item_num, logging_period)

            if snr_analy_bool and (final_spike_noise_factor > 0):
                print(f"Running Signal to Noise Ratio Analysis...")
                SNR_analysis(model.module)

            if mixed_selec_bool:
                print(f"Running Mixed Selectivity Analysis (set size = 1)...")
                mixed_selectivity_analysis(model.module)

    dist.destroy_process_group()


if __name__ == "__main__":
    import torch
    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    main_worker(rank, local_rank, world_size)


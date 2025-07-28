#!/usr/bin/env python
import os
import sys
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2
import concurrent.futures
from tqdm import tqdm
from threading import Lock
import ipdb
import gc
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

## Example usage: 
# (torch2) jd976@cblgpu07:/scratches/kolmogorov_2/jd976/working/vwm_rnn$ 
# python model_evolution_analysis.py --config /homes/jd976/working/vwm_rnn/rnn_models/exp_RegStrength_MultiItem_wPoiNoiseHeterTau_longer/lambda_reg-0.10000000_n512item10PI1k0.005/config.yaml
# python model_evolution_analysis.py --config /homes/jd976/working/vwm_rnn/rnn_models/PsychoSimil_4_2_n256item10PI1gamma0.05/config.yaml

# Add project root to Python path
sys.path.append(os.path.abspath("."))

# Import your model, configuration, and utility functions.
from rnn import RNNMemoryModel
from config import *
from utils import generate_input

# Set basic paths
base_folder = os.path.dirname(config_path)
model_folder = os.path.join(base_folder, "models")

# ===== Option Settings =====
plot_first_n = 0    # Only select first N models; 0 means no filtering.
plot_step = 0       # Select every X-th model; 0 means no filtering.
max_workers = 1
# ===========================

def get_model_files():
    """Return a sorted list of model filenames applying filtering options."""
    model_files = [f for f in os.listdir(model_folder) if re.match(r'model_iteration\d+\.pth', f)]
    model_files = sorted(model_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    if plot_first_n > 0:
        model_files = model_files[:plot_first_n]
    if plot_step > 0:
        model_files = model_files[::plot_step]
    return model_files

# Create a global lock for plotting code.
plot_lock = Lock()

# ---------------------------
# Analysis Functions
# ---------------------------
def analyze_activation(model, iteration_num, common_input, save_folder):
    """
    Run the model on common_input, compute average firing rate and plot it versus time constant τ.
    
    Args:
        model (torch.nn.Module): Loaded model.
        iteration_num (int): Current iteration number.
        common_input (Tensor): Input tensor (u_t) required for activation analysis.
        save_folder (str): Folder to save the resulting plot.
    
    Returns:
        str: Path to the saved PNG.
    """
    with torch.no_grad():
        r_output, _ = model(common_input, r0=None)
    avg_firing_rate = r_output.mean(dim=(0, 1))
    taus = model.tau

    # Convert tensors to numpy arrays for plotting.
    avg_firing_rate_np = avg_firing_rate.detach().cpu().numpy()
    taus_np = taus.detach().cpu().numpy()

    with plot_lock:
        plt.figure(figsize=(6, 4))
        plt.scatter(taus_np, avg_firing_rate_np, alpha=0.7)
        plt.xlabel("Time constant τ (ms)")
        plt.ylabel("Average firing rate (Hz)")
        plt.title(f"Activation at iteration {iteration_num}")
        plt.grid(True)
        plt.ylim(0, 10)
        plt.xscale("log")
        
        png_filename = f"activation_iteration{iteration_num}.png"
        save_path = os.path.join(save_folder, png_filename)
        plt.savefig(save_path, dpi=150)
        plt.close('all')
    return save_path

def analyze_weights(model, iteration_num, save_folder):
    """
    Plot weight matrices (B, W, F) side by side.
    
    Args:
        model (torch.nn.Module): Loaded model.
        iteration_num (int): Current iteration number.
        save_folder (str): Folder to save the resulting plot.
    
    Returns:
        str: File path to the saved weights PNG.
    """
    # Convert weight tensors to numpy arrays.
    B_np = model.B.detach().cpu().numpy()
    W_np = model.W.detach().cpu().numpy()
    F_np = model.F.detach().cpu().numpy()
    effective_W = model.W * model.dales_sign.view(1, -1)
    effective_W_np = effective_W.detach().cpu().numpy()

    with plot_lock:
        # Create a subplot with 1 row and 3 columns.
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 2.5, 1]})
        
        # Add a suptitle with the iteration number.
        fig.suptitle(f"Weight Evolution at iteration {iteration_num}", fontsize=16)

        # Plot B (Input-to-Neurons)
        im0 = axes[0].imshow(B_np, cmap="seismic", vmin=-np.max(np.abs(B_np)), vmax=np.max(np.abs(B_np)))
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=-0.04)
        axes[0].set_title("Input-to-Neurons (B)", fontsize=14)
        axes[0].set_xlabel(f"Inputs ({B_np.shape[1]})", fontsize=12)
        axes[0].set_ylabel(f"Neurons ({B_np.shape[0]})", fontsize=12)

        # Plot W (Recurrent Weights)
        im1 = axes[1].imshow(effective_W_np, cmap="seismic", vmin=-np.max(np.abs(W_np)), vmax=np.max(np.abs(W_np)))
        fig.colorbar(im1, ax=axes[1], fraction=0.023, pad=0.04)
        axes[1].set_title("Recurrent Weights (W)", fontsize=14)
        axes[1].set_xlabel(f"Neurons ({W_np.shape[1]})", fontsize=12)
        axes[1].set_ylabel(f"Neurons ({W_np.shape[0]})", fontsize=12)

        # Plot F (Neurons-to-Output, transposed for display)
        im2 = axes[2].imshow(F_np.T, cmap="seismic", vmin=-np.max(np.abs(F_np)), vmax=np.max(np.abs(F_np)))
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=-0.04)
        axes[2].set_title("Neurons-to-Output (F.T)", fontsize=14)
        axes[2].set_xlabel(f"Outputs ({F_np.shape[0]})", fontsize=12)
        axes[2].set_ylabel(f"Neurons ({F_np.shape[1]})", fontsize=12)
        
        # Remove axis ticks for clarity.
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_anchor("C")

        png_filename = f"weights_iteration{iteration_num}.png"
        save_path = os.path.join(save_folder, png_filename)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close('all')
    return save_path

torch.cuda.empty_cache()
def analyze_error_dist(model, iteration_num, save_folder):
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
    u_hat = model.readout(r_decode.reshape(-1, num_neurons)).reshape(r_decode.shape[0], num_trials, -1)
    decoded_thetas =  model.decode(u_hat)  # (trials, max_items)
    angular_diff = (input_thetas - decoded_thetas + torch.pi) % (2 * torch.pi) - torch.pi  # (trials,items)

    with plot_lock:
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
        plt.title(f'Distribution of Decoding Errors (iteration {iteration_num})', fontsize=16)
        plt.ylim(bottom=0)
    
        # Save the plot
        save_path = os.path.join(save_folder, f'error_distrib_iteration{iteration_num}.png')
        plt.savefig(save_path, dpi=200)
        plt.close('all')

    del input_presence, input_thetas, u_t
    del r_decode, u_hat, decoded_thetas, angular_diff
    torch.cuda.empty_cache()
    gc.collect()
    return save_path

        
# Dictionary to register analyses.
# Each key is the analysis name, and the value is a dict with:
#   - 'enabled': a boolean to toggle the analysis
#   - 'function': the analysis function
#   - 'save_folder': the folder to store its outputs
#   - 'gif_name': the name of the final GIF
#   - 'requires_input': whether the analysis function needs the common input u_t.
ANALYSES = {
    "activation": {
        "enabled": False,
        "function": analyze_activation,
        "save_folder": os.path.join(model_folder, "ActivVsTau"),
        "gif_name": "Activ_vs_Tau.gif",
        "requires_input": True,
    },
    "weights": {
        "enabled": False,
        "function": analyze_weights,
        "save_folder": os.path.join(model_folder, "WeightsEvolution"),
        "gif_name": "WeightsEvolution.gif",
        "requires_input": False,
    },
    "error_dist": {
        "enabled": True,
        "function": analyze_error_dist,
        "save_folder": os.path.join(model_folder, "ErrorDistEvolution"),
        "gif_name": "ErrorDistEvolution.gif",
        "requires_input": False,
    },
}

def create_common_input():
    """
    Creates a common input tensor (u_t) for analyses that require it.
    """
    num_trials = 500
    trials_per_group = num_trials // len(item_num)
    remaining_trials = num_trials % len(item_num)
    trial_counts = [trials_per_group + (1 if i < remaining_trials else 0) for i in range(len(item_num))]
    input_presence = torch.zeros(num_trials, max_item_num, device=device, requires_grad=True)
    start_index = 0
    for i, count in enumerate(trial_counts):
        end_index = start_index + count
        one_hot_indices = torch.stack([torch.randperm(max_item_num, device=device)[:item_num[i]] for _ in range(count)])
        input_presence_temp = input_presence.clone()
        input_presence_temp[start_index:end_index] = input_presence_temp[start_index:end_index].scatter(1, one_hot_indices, 1)
        input_presence = input_presence_temp
        start_index = end_index
    input_thetas = ((torch.rand(num_trials, max_item_num, device=device) * 2 * torch.pi) - torch.pi)
    u_t = generate_input(
        presence=input_presence,
        theta=input_thetas,
        input_strength=input_strength,
        noise_level=ILC_noise,
        T_init=T_init,
        T_stimi=T_stimi,
        T_delay=T_delay,
        T_decode=T_decode,
        dt=dt,
    )
    return u_t

def process_model_file(model_file, common_input):
    """
    Loads a single model, runs all enabled analyses, and returns iteration and a dict mapping analysis names to PNG paths.
    """
    iteration_num = int(re.findall(r'\d+', model_file)[0])
    model_path = os.path.join(model_folder, model_file)
    model = RNNMemoryModel(max_item_num, num_neurons, dt, tau_min, tau_max, spike_noise_type, 
                           spike_noise_factor, saturation_firing_rate, device, positive_input, dales_law)
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
    model.device = device
    model.eval()

    results = {}
    for name, analysis in ANALYSES.items():
        if not analysis["enabled"]:
            continue
        func = analysis["function"]
        if analysis["requires_input"]:
            png_path = func(model, iteration_num, common_input, analysis["save_folder"])
        else:
            png_path = func(model, iteration_num, analysis["save_folder"])
        results[name] = png_path
    return (iteration_num, results)

def main():
    """
    For each model file, load the model once, run all enabled analyses in parallel,
    and then create GIFs for each analysis.
    """
    for key, analysis in ANALYSES.items():
        if analysis["enabled"]:
            os.makedirs(analysis["save_folder"], exist_ok=True)

    common_input = create_common_input() if any( (a["requires_input"] and a["enabled"]) for a in ANALYSES.values()) else None
    model_files = get_model_files()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_model_file, model_file, common_input) for model_file in model_files]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing models"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing a model file: {e}")

    results = sorted(results, key=lambda x: x[0])
    png_paths = {name: [] for name, a in ANALYSES.items() if a["enabled"]}
    for iteration_num, result in results:
        for name in result:
            png_paths[name].append(result[name])

    for name, analysis in ANALYSES.items():
        if not analysis["enabled"]:
            continue
        gif_name = analysis["gif_name"]
        frames = [imageio.v2.imread(f) for f in png_paths[name]]
        gif_path = os.path.join(analysis["save_folder"], gif_name)
        imageio.v2.mimsave(gif_path, frames, duration=1)
        print(f"Created {name} evolution GIF: {gif_path}")

if __name__ == "__main__":
    main()
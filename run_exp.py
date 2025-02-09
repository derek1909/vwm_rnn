import yaml
import subprocess
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import sys

def setup_experiment(exp_name, config_source="./config.yaml"):
    """Setup experiment directory and copy config file."""
    exp_dir = f"./rnns/exp_{exp_name}"
    config_dest = f"{exp_dir}/config.yaml"
    
    os.makedirs(exp_dir, exist_ok=True)
    shutil.copy2(config_source, config_dest)

    """Save a copy of the current script into the experiment folder."""
    # sys.argv[0] contains the current script path.
    current_script = sys.argv[0]
    dest_path = os.path.join(exp_dir, os.path.basename(current_script))
    shutil.copy2(current_script, dest_path)
    
    print(f"Created experiment folder: {exp_dir}")
    print(f"Please check the configuration file at {config_dest} before continuing.")
    input("Press Enter to continue or Ctrl+C to abort...")
    
    return exp_dir, config_dest

def run_experiments(exp_name, config_path, para_list, para_cat, para_name, main_script="python main.py"):
    """Run experiments for different RNN sizes."""
    for para in para_list:
        # Load and modify config
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            
        def format_param(param):
            if isinstance(param, float):
                return f"{param:.8f}"
            return f"{param}"
        
        data["model_and_logging_params"]["rnn_name"] = f"exp_{exp_name}/{para_name}-{format_param(para)}"
        data[para_cat][para_name] = para
        
        # Save modified config
        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
            
        print(f"Running main.py with {para_name} = {para}.")
        subprocess.run(f"{main_script} --config {config_path}", shell=True)

def collect_results(exp_dir, para_name):
    """Collect results from all completed runs."""
    final_results = {}
    
    for rnn_dir in os.listdir(exp_dir):
        if not os.path.isdir(os.path.join(exp_dir, rnn_dir)):
            continue
            
        # Extract num_neurons from directory name
        parts = rnn_dir.replace('_', '-').split('-')
        try:
            para = parts[-2]
        except ValueError:
            print(f"Skipping directory {rnn_dir} - cannot extract {para_name}")
            continue
        
        # Load training history
        history_path = f"{exp_dir}/{rnn_dir}/training_history.yaml"
        try:
            with open(history_path, 'r') as f:
                history = yaml.safe_load(f)
                
            final_results[para] = {
                'final_epoch': history['epochs'][-1],
                'final_error': history['error_per_epoch'][-1],
                'final_error_std': history['error_std_per_epoch'][-1],
                'final_activation': history['activation_per_epoch'][-1],
                'group_errors': [errors[-1] for errors in history['group_errors']],
                'group_std': [std[-1] for std in history['group_std']],
                'group_activ': [activ[-1] for activ in history['group_activ']]
            }
        except FileNotFoundError:
            print(f"Warning: Could not find history file for {rnn_dir}")
            continue
            
    return final_results

def plot_results_vs_neurons(results, save_path, fig_size=(8, 3)):
    """Plot and save performance curves."""
    neuron_sizes = sorted(list(results.keys()))
    setsizes = np.arange(1, 11)
    
    # Prepare data
    errors = {size: results[size]['group_errors'] for size in neuron_sizes}
    activations = {size: results[size]['group_activ'] for size in neuron_sizes}
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=fig_size, sharex=True)
    
    # Error plot
    for neurons in neuron_sizes:
        axes[0].plot(setsizes, errors[neurons], '-', label=f"N={neurons}")
    axes[0].set_xlabel('Item number')
    axes[0].set_ylabel('Error (rad)')
    axes[0].set_title('Error vs Item Number')
    axes[0].grid(True)
    
    # Activation plot
    for neurons in neuron_sizes:
        axes[1].plot(setsizes, activations[neurons], '-', label=f"N={neurons}")
    axes[1].set_xlabel('Item number')
    axes[1].set_ylabel('Activation (Hz)')
    axes[1].set_title('Average Firing Rate')
    axes[1].grid(True)
    
    axes[1].legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Parameters
    EXP_NAME = "ILC_NoDelay"
    PARA_CATA = "model_params"
    PARA_NAME = "ILC_noise"
    PARA_LIST = [0, 1e-3, 1e-2, 0.1, 0.5, 1, 2, 5]
    
    # Setup experiment
    exp_dir, config_path = setup_experiment(EXP_NAME)
    
    # Run experiments (uncomment to run)
    run_experiments(EXP_NAME, config_path, PARA_LIST, PARA_CATA, PARA_NAME)
    
    # Collect and save results
    results = collect_results(exp_dir, PARA_NAME)
    results_path = f"{exp_dir}/final_results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"Final results saved to {results_path}")
    
    # Plot results
    # plot_path = f"{exp_dir}/performance_curves.png"
    # plot_results_vs_neurons(results, plot_path)
    # print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
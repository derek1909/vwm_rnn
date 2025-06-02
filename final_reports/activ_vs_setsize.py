import matplotlib.pyplot as plt
import yaml
import os

def load_yaml_group_activation(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    curves = {}
    for noise_level, stats in data.items():
        group_activ = stats['group_activ']
        curves[float(noise_level)] = group_activ
    return curves

def plot_activation_vs_setsize(yaml_paths, net_sizes):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    set_sizes = list(range(1, 11))  # set size 1 to 10

    for ax, path, size in zip(axes, yaml_paths, net_sizes):
        curves = load_yaml_group_activation(path)
        for noise, activ in sorted(curves.items()):
            ax.plot(set_sizes, activ, label=f"Noise={noise:.1f}")
        ax.set_title(f"Network Size: {size}")
        ax.set_xlabel("Set Size")
        ax.set_ylabel("Mean Activation")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.suptitle("Activation vs. Set Size under Varying Poisson Noise", fontsize=16, y=1.05)

    output_folder = "./final_reports/"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "activation_vs_setsize.png")
    plt.savefig(output_path)
    plt.close()

# Example usage:
yaml_paths = [
    "/homes/jd976/working/vwm_rnn/rnn_models/exp_Neuron64Input120/final_results.yaml",
    "/homes/jd976/working/vwm_rnn/rnn_models/exp_Neuron128Input120/final_results.yaml",
    "/homes/jd976/working/vwm_rnn/rnn_models/exp_Neuron256Input120/final_results.yaml",
]

net_sizes = [64, 128, 256]

plot_activation_vs_setsize(yaml_paths, net_sizes)

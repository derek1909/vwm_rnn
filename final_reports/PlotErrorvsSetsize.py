import matplotlib.pyplot as plt
import yaml
import os

def load_yaml_group_errors(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    curves = {}
    for noise_level, stats in data.items():
        group_errors = stats['group_errors']
        curves[float(noise_level)] = group_errors
    return curves

def plot_error_vs_setsize(yaml_paths, net_sizes):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True)
    set_sizes = list(range(1, 11))  # set size 1 to 10

    handles_labels = []

    for ax, path, size in zip(axes, yaml_paths, net_sizes):
        curves = load_yaml_group_errors(path)
        lines = []
        for noise, errors in sorted(curves.items()):
            line, = ax.plot(set_sizes, errors, label=f"Noise={noise:.1f}", linewidth=2)
            lines.append(line)
        ax.set_title(f"Network Size: {size}")
        ax.set_xlabel("Set Size")
        ax.set_ylabel("Error (rad)")
        ax.grid(True)
        if not handles_labels:  # collect legend handles from the first plot
            handles_labels = lines

    # Shared legend
    fig.legend(handles_labels, [line.get_label() for line in handles_labels],
               loc='lower center', ncol=len(handles_labels), frameon=False)

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space for legend at bottom
    output_folder = "./final_reports/"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "error_vs_setsize.png")
    plt.savefig(output_path, dpi=200)
    plt.close()


# Example usage:
yaml_paths = [
    "/homes/jd976/working/vwm_rnn/rnn_models/exp_Neuron64Input120/final_results.yaml",
    "/homes/jd976/working/vwm_rnn/rnn_models/exp_Neuron128Input120/final_results.yaml",
    "/homes/jd976/working/vwm_rnn/rnn_models/exp_Neuron256Input120/final_results.yaml",
]

net_sizes = [64, 128, 256]

plot_error_vs_setsize(yaml_paths, net_sizes)

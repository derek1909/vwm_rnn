import matplotlib.pyplot as plt
import os

# Set size
set_sizes = list(range(1, 11))

# Data: Angular model
angular_inputs = [2.9642, 6.0850, 9.0225, 11.8500, 14.4959, 17.4947, 20.6645, 23.5586, 26.4222, 29.7114]
angular_activs = [14.2744, 17.0799, 15.7316, 16.6982, 17.6330, 16.2428, 18.4952, 18.2309, 17.6157, 20.8593]

# Data: Euclidean model
euclidean_inputs = [2.8533, 5.7505, 9.0616, 11.3643, 14.6329, 17.1520, 19.7697, 23.1205, 26.3719, 28.9488]
euclidean_activs = [7.6163, 9.0615, 10.2579, 10.5008, 10.8563, 11.8823, 11.7113, 13.0876, 12.2808, 12.9843]

# Combined data
model_data = {
    "Angular Error": {
        "inputs": angular_inputs,
        "activs": angular_activs
    },
    "Euclidean Error": {
        "inputs": euclidean_inputs,
        "activs": euclidean_activs
    }
}

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharex=True)

lines = []

# Left plot: Mean Input vs Set Size
for label, values in model_data.items():
    line, = axes[0].plot(set_sizes, values["inputs"], label=label, linewidth=2)
    lines.append(line)
# axes[0].set_title("Mean Input vs Set Size")
axes[0].set_xlabel("Set Size")
axes[0].set_ylabel("Mean Input (Hz)")
axes[0].set_ylim([0,32])
axes[0].grid(True)

# Right plot: Mean Activation vs Set Size
for _, values in model_data.items():
    axes[1].plot(set_sizes, values["activs"], linewidth=2)
# axes[1].set_title("Mean Activity vs Set Size")
axes[1].set_xlabel("Set Size")
axes[1].set_ylabel("Mean Activation (Hz)")
axes[1].set_ylim([0,32])
axes[1].grid(True)

# Shared legend outside
fig.legend(lines, [line.get_label() for line in lines], loc='lower center', ncol=2, frameon=False)
plt.tight_layout(rect=[0, 0.1, 1, 1])

# Save figure
output_folder = "./final_reports/"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "input_activation_vs_setsize.png")
plt.savefig(output_path, dpi=200)
plt.close()

output_path

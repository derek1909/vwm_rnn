import os
import yaml
import matplotlib.pyplot as plt

# Path to the YAML file
yaml_file = "/homes/jd976/working/vwm_rnn/rnns/exp_PoiNoise/final_results.yaml"

# Load the YAML data
with open(yaml_file, 'r') as f:
    data = yaml.safe_load(f)

# The YAML keys represent poisson noise factors. Sort them numerically.
sorted_keys = sorted(data.keys(), key=lambda x: float(x))

# Filter out any zero (or negative) values because log scale cannot include zero.
poisson_vals = []
activation_vals = []
error_vals = []
for key in sorted_keys:
    poisson_value = float(key)
    if poisson_value < 0:
        continue  # Skip zero (or negative) values for log-scaled x-axis.
    poisson_vals.append(poisson_value)
    activation_vals.append(data[key]['final_activation'])
    error_vals.append(data[key]['final_error'])

# Create two subplots side-by-side with shared x-axis.
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# Plot final_activation vs. poisson noise factor with x-axis on a logarithmic scale.
axes[0].plot(poisson_vals, activation_vals, '-o', label='Activation')
axes[0].set_xscale('log')
axes[0].set_xlabel('Poisson Noise Factor')
axes[0].set_ylabel('Activation (Hz)')
axes[0].set_title('Activation vs Poisson Noise Factor')
axes[0].grid(True, which='both')
axes[0].legend()

# Plot final_error vs. poisson noise factor with x-axis on a logarithmic scale.
axes[1].plot(poisson_vals, error_vals, '-o', color='r', label='Error')
axes[1].set_xscale('log')
axes[1].set_xlabel('Poisson Noise Factor')
axes[1].set_ylabel('Error (rad)')
axes[1].set_title('Error vs Poisson Noise Factor')
axes[1].grid(True, which='both')
axes[1].legend()

plt.tight_layout()

# Save the figure to the same folder as the YAML file and then close the plot.
folder = os.path.dirname(yaml_file)
output_path = os.path.join(folder, "final_results_poisson_plot.png")
plt.savefig(output_path)
plt.close()
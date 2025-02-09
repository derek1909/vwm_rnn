import os
import yaml
import matplotlib.pyplot as plt

# Path to the YAML file
yaml_file = '/homes/jd976/working/vwm_rnn/rnns/exp_RegStrength/final_results.yaml'

# Load the YAML data
with open(yaml_file, 'r') as f:
    data = yaml.safe_load(f)

# Sort the keys (reg_lambda values) numerically and convert to floats.
sorted_keys = sorted(data.keys(), key=lambda x: float(x))
reg_lambda = [float(k) for k in sorted_keys]

# Extract final_activation and final_error values corresponding to each reg_lambda
final_activation = [data[k]['final_activation'] for k in sorted_keys]
final_error = [data[k]['final_error'] for k in sorted_keys]

# Create two side-by-side subplots with shared x-axis
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# Plot final_activation vs. reg_lambda on a log-scaled x-axis
axes[0].plot(reg_lambda, final_activation, '-o', label='Activation')
axes[0].set_xscale('log')
axes[0].set_xlabel('reg_lambda')
axes[0].set_ylabel('Activation')
axes[0].set_title('Activation vs reg_lambda')
axes[0].grid(True, which='both')
axes[0].legend()

# Plot final_error vs. reg_lambda on a log-scaled x-axis
axes[1].plot(reg_lambda, final_error, '-o', color='r', label='Error')
axes[1].set_xscale('log')
axes[1].set_xlabel('reg_lambda')
axes[1].set_ylabel('Error')
axes[1].set_title('Error vs reg_lambda')
axes[1].grid(True, which='both')
axes[1].legend()

plt.tight_layout()

# Save the figure to the same folder as the YAML file and close the plot.
folder = os.path.dirname(yaml_file)
output_path = os.path.join(folder, "final_results_reg_lambda_plot.png")
plt.savefig(output_path)
plt.close()
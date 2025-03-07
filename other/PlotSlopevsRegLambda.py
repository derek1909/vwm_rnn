import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Path to the YAML file
yaml_file = "/homes/jd976/working/vwm_rnn/rnns/exp_RegStrength_MultiItem_wPoiNoiseHeterTau_longer/final_results.yaml"

# Load the YAML data
with open(yaml_file, 'r') as f:
    data = yaml.safe_load(f)

# Sort the keys (reg_lambda values) numerically and convert to floats.
sorted_keys = sorted(data.keys(), key=lambda x: float(x))
reg_lambda = [float(k) for k in sorted_keys]

# Initialize lists to store slopes for activations and errors.
activation_slopes = []
error_slopes = []

# Compute the slope (linear regression coefficient) for each lambda value.
for k in sorted_keys:
    group_activ = data[k]['group_activ']
    group_errors = data[k]['group_errors']
    
    # Use the index as the independent variable.
    x = np.arange(len(group_activ))
    
    # Compute linear fit (slope and intercept) for activation values.
    slope_activ, _ = np.polyfit(x, group_activ, 1)
    activation_slopes.append(slope_activ)
    
    # Compute linear fit (slope and intercept) for error values.
    slope_error, _ = np.polyfit(x, group_errors, 1)
    error_slopes.append(slope_error)

# Create two side-by-side subplots with shared x-axis.
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# Plot activation slope vs. reg_lambda on a log-scaled x-axis.
axes[0].plot(reg_lambda, activation_slopes, '-o', label='Activation Slope')
axes[0].set_xscale('log')
axes[0].set_xlabel('reg_lambda')
axes[0].set_ylabel('Slope')
axes[0].set_title('Activation Slope vs reg_lambda')
axes[0].grid(True, which='both')
axes[0].legend()

# Plot error slope vs. reg_lambda on a log-scaled x-axis.
axes[1].plot(reg_lambda, error_slopes, '-o', color='r', label='Error Slope')
axes[1].set_xscale('log')
axes[1].set_xlabel('reg_lambda')
axes[1].set_ylabel('Slope')
axes[1].set_title('Error Slope vs reg_lambda')
axes[1].grid(True, which='both')
axes[1].legend()

plt.tight_layout()

# Save the figure to the same folder as the YAML file and close the plot.
folder = os.path.dirname(yaml_file)
output_path = os.path.join(folder, "final_results_reg_lambda_slopes_plot.png")
plt.savefig(output_path)
plt.close()
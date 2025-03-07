import os
import yaml
import matplotlib.pyplot as plt

# Path to the YAML file
yaml_file = '/homes/jd976/working/vwm_rnn/rnns/exp_Delay_MultiItem_HeterTau/final_results.yaml'

# Load the YAML data
with open(yaml_file, 'r') as f:
    data = yaml.safe_load(f)

# Extract T-delay values and corresponding final activation and final error.
# Sort keys numerically (they may be strings, so convert them to integers for sorting)
delays = sorted(data.keys(), key=lambda x: int(x))
final_activation = [data[d]['final_activation'] for d in delays]
final_error = [data[d]['final_error'] for d in delays]

# Create two side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

# Plot Activation vs. T-delay
axes[0].plot(delays, final_activation, '-o', label='Activation')
axes[0].set_xlabel('T-delay')
axes[0].set_ylabel('Activation')
axes[0].set_title('Activation vs T-delay')
axes[0].grid(True)
axes[0].legend()

# Plot Error vs. T-delay
axes[1].plot(delays, final_error, '-o', label='Error')
axes[1].set_xlabel('T-delay')
axes[1].set_ylabel('Error')
axes[1].set_title('Error vs T-delay')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()

# Save the figure in the same folder as the YAML file and then close it
folder = os.path.dirname(yaml_file)
output_path = os.path.join(folder, "final_results_plot.png")
plt.savefig(output_path)
plt.close()
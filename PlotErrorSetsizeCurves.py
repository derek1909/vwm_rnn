import matplotlib.pyplot as plt
import numpy as np

# Data for baseline (64 neurons)
setsizes = [1, 3, 5, 7]
baseline_error_e2 = [0.25, 1.04, 1.27, 1.36]
baseline_error_degree = [29, 61, 69, 71]
baseline_avg_firing_rate = [5.44, 5.42, 5.49, 5.56]

# Data for more neurons (128 neurons)
more_neurons_error_e2 = [0.065, 0.49, 0.75, 0.89]
more_neurons_error_degree = [14.6, 41, 51, 56]
more_neurons_avg_firing_rate = [5.11, 5.21, 5.26, 5.20]


# Plot Error (e2) vs Set Size
plt.figure(figsize=(6, 4))
plt.plot(setsizes, baseline_error_e2, 'o-', label="Baseline (64 neurons)")
plt.plot(setsizes, more_neurons_error_e2, 's-', label="More neurons (128 neurons)")
plt.xlabel('Set Size')
plt.ylabel('Error (e2)')
plt.title('Error (e2) vs Set Size')
plt.legend()
plt.grid(True)

# Plot Error (rad) vs Set Size
plt.figure(figsize=(6, 4))
plt.plot(setsizes, baseline_error_degree, 'o-', label="Baseline (64 neurons)")
plt.plot(setsizes, more_neurons_error_degree, 's-', label="More neurons (128 neurons)")
plt.xlabel('Set Size')
plt.ylabel('Error (degree)')
plt.title('Error (degree) vs Set Size')
plt.legend()
plt.grid(True)

# Plot Avg Firing Rate vs Set Size
plt.figure(figsize=(6, 4))
plt.plot(setsizes, baseline_avg_firing_rate, 'o-', label="Baseline (64 neurons)")
plt.plot(setsizes, more_neurons_avg_firing_rate, 's-', label="More neurons (128 neurons)")
plt.xlabel('Set Size')
plt.ylabel('Average Firing Rate (Hz)')
plt.title('Average Firing Rate vs Set Size')
plt.legend()
plt.grid(True)
plt.show()
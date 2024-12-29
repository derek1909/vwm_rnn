import matplotlib.pyplot as plt
import numpy as np

# Re-defining data
setsizes = [1, 3, 5, 7]

# Lambda = 2e-5
# Data for baseline (64 neurons)
baseline_error_degree = [29, 61, 69, 71]
baseline_avg_firing_rate = [5.44, 5.42, 5.49, 5.56]

# Data for more neurons (128 neurons)
more_neurons_error_degree = [14.6, 41, 51, 56]
more_neurons_avg_firing_rate = [5.11, 5.21, 5.26, 5.20]

# Data for more neurons (256 neurons)
more_neurons_256_error_degree = [11.5, 30.9, 39.2, 46.1]
more_neurons_256_avg_firing_rate = [5.23, 5.38, 5.35, 5.43]

# Lambda = 5e-4
errors_e2 = {
    64: [0.227, 0.957, 1.186, 1.325],
    128: [0.094, 0.753, 0.970, 1.175],
    256: [0.046, 0.392, 0.656, 0.740],
    1024: [0.013, 0.089, 0.155, 0.218],
}

# Calculate degree error using the formula
degree_errors = {
    neurons: [np.arccos(1 - 0.5 * e2) / np.pi * 180 for e2 in errors]
    for neurons, errors in errors_e2.items()
}

firing_rates = {
    64: [4.757, 4.767, 4.757, 4.800],
    128: [5.039, 5.011, 4.999, 5.001],
    256: [4.327, 4.380, 4.396, 4.382],
    1024: [4.160, 4.213, 4.237, 4.271],
}

# 9
fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharex=True)

# Plot Error (degree) vs Set Size
for neurons, errors in degree_errors.items():
    axes[0].plot(setsizes, errors, marker='o', label=f"N={neurons}")
# axes[0].plot(setsizes, baseline_error_degree, 'o-', label="N=64")
# axes[0].plot(setsizes, more_neurons_error_degree, 's-', label="N=128")
# axes[0].plot(setsizes, more_neurons_256_error_degree, 'd-', label="N=256")
axes[0].set_xlabel('Item number')
axes[0].set_ylabel('Error (degrees)')
axes[0].set_title('Error (degree) vs Item number')
axes[0].legend()
axes[0].grid(True)

# Plot Average Firing Rate vs Set Size
for neurons, rates in firing_rates.items():
    axes[1].plot(setsizes, rates, marker='o', label=f"N={neurons}")
# axes[1].plot(setsizes, baseline_avg_firing_rate, 'o-', label="N=64")
# axes[1].plot(setsizes, more_neurons_avg_firing_rate, 's-', label="N=128")
# axes[1].plot(setsizes, more_neurons_256_avg_firing_rate, 'd-', label="N=256")
axes[1].set_xlabel('Item number')
axes[1].set_ylabel('Average Firing Rate (Hz)')
axes[1].set_title('Average Firing Rate vs Item number')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
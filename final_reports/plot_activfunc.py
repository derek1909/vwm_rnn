import numpy as np
import matplotlib.pyplot as plt
import os

# ----- Parameters -----
gain = 0.14
theta = gain * 30

# ----- Activation Functions -----

# Original tanh-based
def phi_tanh(x):
    return 30 * (1 + np.tanh(gain * x - theta))

# Supralinear SSN-style
def phi_supralinear(x):
    return 1.2 * np.maximum(x - 10, 0)

# Input range
x = np.linspace(-2, 60, 500)
y_tanh = phi_tanh(x)
y_ssn = phi_supralinear(x)

# ----- Plotting -----
plt.figure(figsize=(5, 4))
plt.plot(x, y_tanh, label='Saturated', linewidth=2)
plt.plot(x, y_ssn, label='Supralinear (SSN)', linewidth=2, linestyle='--')

# Annotations and style
plt.axhline(0, color='k', linestyle='--', linewidth=0.7)
plt.axvline(0, color='k', linestyle='--', linewidth=0.7)
plt.xlabel(r'$x$')
plt.ylabel('Firing rate (Hz)')
plt.ylim([0,100])
# plt.title('Activation Function Comparison')
plt.legend(loc='upper left', frameon=False)
plt.tight_layout()

# Save the figure
output_folder = './final_reports/'
os.makedirs(output_folder, exist_ok=True)
save_path = os.path.join(output_folder, 'activ_func_compare.png')
plt.savefig(save_path, dpi=300)
print(f"Saved plot to {save_path}")
plt.close()

import numpy as np
import matplotlib.pyplot as plt
import os
# Parameters
gain = 0.13
theta = gain * 30

# Define the activation function Φ(x)
def phi(x):
    return  30 * (1 + np.tanh(gain * x - theta))

# Input range
x = np.linspace(-2, 60, 500)
y = phi(x)

# Plot
plt.figure(figsize=(4, 4))
plt.plot(x, y, 'b')

# Annotations and style
plt.axhline(0, color='k', linestyle='--', linewidth=0.7)
plt.axvline(0, color='k', linestyle='--', linewidth=0.7)
plt.xlabel(r'$x$')
plt.ylabel('Firing rate (Hz)')
# plt.legend(loc='lower right', frameon=True)
plt.title(rf'$\Phi(x) = 30 \cdot (1 + \tanh({gain}x - {theta}))$')
plt.tight_layout()
plt.axis('equal')

# Save the figure
os.makedirs('./other', exist_ok=True)
save_path = f'./other/activ_func_theta={theta}_gain={gain}.png'
plt.savefig(save_path, dpi=300)
print(f"Saved plot to {save_path}")
plt.close()
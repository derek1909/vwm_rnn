import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os
from math import sqrt
# from mpl_toolkits.mplot3d import Axes3D
# parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# sys.path.append(parent_folder)
# from utils import generate_input
device = 'cpu'

def generate_input(presence, theta, noise_level=0.0, T_init=0, T_stimi=400, T_delay=0, T_decode=800, dt=10, alpha=0):
    """
    Generate a 3D input tensor of shape (steps, num_trials, 2 * max_item_num) without loops.

    Args:
        presence: (num_trials, max_item_num) binary tensor indicating presence of items.
        theta: (num_trials, max_item_num) tensor of angles.
        noise_level: Noise level to be added to theta.
        T_init, T_stimi, T_delay, T_decode: Timings for each phase (in ms).
        dt: Time step size (in ms).

    Returns:
        u_t_stack: (num_trials, steps, 2 * max_item_num) tensor of input vectors over time.
    """
    # Total simulation time and steps
    T_simul = T_init + T_stimi + T_delay + T_decode
    steps = int(T_simul / dt)
    num_trials, max_item_num = presence.shape

    # Add noise to theta
    theta_noisy = theta.unsqueeze(0) + noise_level * torch.randn(
        (steps, num_trials, max_item_num), device=device
    )
    theta_noisy = (theta_noisy + torch.pi) % (2 * torch.pi) - torch.pi

    # # Compute the 2D positions (cos and sin components) for all items
    # cos_theta = torch.cos(theta_noisy/4+torch.pi/4)  # (steps, num_trials, max_item_num)
    # sin_theta = torch.sin(theta_noisy/4+torch.pi/4)  # (steps, num_trials, max_item_num)

    # Compute the 2D positions (cos and sin components) for all items
    cos_theta = torch.cos(theta_noisy)  # (steps, num_trials, max_item_num)
    sin_theta = torch.sin(theta_noisy)  # (steps, num_trials, max_item_num)

    # Stack cos and sin into a single tensor along the last dimension
    # Then multiply by presence to zero-out absent items
    # u_0 = ( torch.stack((cos_theta, sin_theta), dim=-1) + alpha ) * presence.unsqueeze(0).unsqueeze(-1) # (steps, num_trials, max_item_num, 2)
    u_0 = ( torch.stack((
                1 + cos_theta / sqrt(2) + sin_theta / sqrt(6), 
                1 - cos_theta / sqrt(2) + sin_theta / sqrt(6),
                1 - 2 * sin_theta / sqrt(6),
            ), dim=-1) ) * presence.unsqueeze(0).unsqueeze(-1) # (steps, num_trials, max_item_num, 3)


    # Reshape to match output shape (combine cos and sin into one dimension)
    u_0 = u_0.view(steps, num_trials, -1)  # (steps, num_trials, 2 * max_item_num)

    # Create a mask for stimuli presence at each time step
    stimuli_present_mask = (torch.arange(steps, device=device) * dt >= T_init) & \
                           (torch.arange(steps, device=device) * dt < T_init + T_stimi)
    stimuli_present_mask = stimuli_present_mask.float().unsqueeze(-1).unsqueeze(-1)  # (steps, 1, 1)

    # Apply the stimuli mask
    u_t_stack = u_0 * stimuli_present_mask  # (steps, num_trials, 2 * max_item_num)

    # Swap dimensions 0 and 1 to get (num_trials, steps, 2 * max_item_num)
    u_t_stack = u_t_stack.transpose(0, 1)

    return u_t_stack


num_trials = 128
input_presence = torch.ones(num_trials,1)
input_thetas = torch.linspace(-torch.pi, torch.pi, num_trials).unsqueeze(1) # for 1item

u_t = generate_input(
    presence=input_presence,
    theta=input_thetas,
    noise_level=0.0,
    T_init=0,
    T_stimi=1,
    T_delay=0,
    T_decode=0,
    dt=1,
    alpha=0,
)

# Squeeze dimensions to make them easier to work with
u_t = u_t.squeeze()  # Shape becomes (64, 2)
theta = input_thetas.squeeze()  # Shape becomes (64,)

# Extract x and y coordinates
x = u_t[:, 0]
y = u_t[:, 1]
z = u_t[:, 2]



fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, c=theta, cmap='rainbow', label="Circle")
ax.scatter(1, 1, 1, color='r', marker='x', label="Center (1,1,1)")
plt.colorbar(scatter, label="Theta Values")  # Add colorbar to indicate theta values

# Highlight 3D Axes
ax.plot([0, 2], [0, 0], [0, 0], color='black', linewidth=1, linestyle='-')  # X-axis
ax.plot([0, 0], [0, 2], [0, 0], color='black', linewidth=1, linestyle='-')  # Y-axis
ax.plot([0, 0], [0, 0], [0, 2], color='black', linewidth=1, linestyle='-')  # Z-axis

# Set labels
# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# ax.set_zlabel("Z-axis")


ax.set_xlim(0,2)
ax.set_ylim(0,2)
ax.set_zlim(0,2)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(elev=15, azim=-80)  # Elevation and azimuth angle

# ax.set_axis_off()  # Removes all axis elements (frame, ticks, labels)
ax.grid(False)  # Removes grid

ax.xaxis.line.set_color((1,1,1,0))  # Hide X-axis line
ax.yaxis.line.set_color((1,1,1,0))  # Hide Y-axis line
ax.zaxis.line.set_color((1,1,1,0))  # Hide Z-axis line

# plt.show()
plt.savefig('./other/input_PI_3d.png', dpi=300, bbox_inches='tight')

# plt.show()

# # Save the plot
# plt.close()
'''
plot_utils.py
Supports FixedPointFinder
Written for Python 3.8.17
@ Matt Golub, October 2018
Please direct correspondence to mgolub@cs.washington.edu
'''

import numpy as np
import ipdb
import os
from tqdm import tqdm
import torch

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

from rnn import *
from config import *
from utils import *
from train import *

def plot_fps(fps,
	state_traj=None,
	plot_batch_idx=None,
	plot_start_time=0,
	plot_stop_time=None,
	mode_scale=0.25,
	fig=None,
	save_path=None,
	iteration=None,
	plot_fps=False,
	thetas=None):

	'''Plots a visualization and analysis of the unique fixed points.

	1) Finds a low-dimensional subspace for visualization via PCA. If
	state_traj is provided, PCA is fit to [all of] those RNN state
	trajectories. Otherwise, PCA is fit to the identified unique fixed
	points. This subspace is 3-dimensional if the RNN state dimensionality
	is >= 3.

	2) Plots the PCA representation of the stable unique fixed points as
	black dots.

	3) Plots the PCA representation of the unstable unique fixed points as
	red dots.

	4) Plots the PCA representation of the modes of the Jacobian at each
	fixed point. By default, only unstable modes are plotted.

	5) (optional) Plots example RNN state trajectories as blue lines.

	Args:
		fps: a FixedPoints object. See FixedPoints.py.

		state_traj (optional): [n_batch x n_time x n_states] numpy
		array or LSTMStateTuple with .c and .h as
		[n_batch x n_time x n_states/2] numpy arrays. Contains example
		trials of RNN state trajectories.

		plot_batch_idx (optional): Indices specifying which trials in
		state_traj to plot on top of the fixed points. Default: plot all
		trials.

		plot_start_time (optional): int specifying the first timestep to
		plot in the example trials of state_traj. Default: 0.

		plot_stop_time (optional): int specifying the last timestep to
		plot in the example trials of stat_traj. Default: n_time.

		stop_time (optional):

		mode_scale (optional): Non-negative float specifying the scaling
		of the plotted eigenmodes. A value of 1.0 results in each mode
		plotted as a set of diametrically opposed line segments
		originating at a fixed point, with each segment's length specified
		by the magnitude of the corresponding eigenvalue.

		fig (optional): Matplotlib figure upon which to plot.

	Returns:
		None.
	'''
	thetas = thetas.detach().numpy().squeeze()

	## Prepare ax ##
	FONT_WEIGHT = 'bold'
	if fig is None:
		FIG_WIDTH = 6 # inches
		FIG_HEIGHT = 6 # inches
		fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT),
			tight_layout=True)
	# plt.ion()

	if state_traj is not None:
		
		state_traj_bxtxd = state_traj
		[n_batch, n_time, n_states] = state_traj_bxtxd.shape

		# Ensure plot_start_time >= 0
		plot_start_time = np.max([plot_start_time, 0])

		if plot_stop_time is None:
			plot_stop_time = n_time
		else:
			# Ensure plot_stop_time <= n_time
			plot_stop_time = np.min([plot_stop_time, n_time])

		plot_time_idx = list(range(plot_start_time, plot_stop_time))

	n_inits = fps.n
	n_states = fps.n_states

	pca = PCA(n_components=3)

	if state_traj is not None:
		state_traj_btxd = np.reshape(state_traj_bxtxd,
			(n_batch*n_time, n_states))
		pca.fit(state_traj_btxd)
	else:
		pca.fit(fps.xstar)

	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('PC 1', fontweight=FONT_WEIGHT)
	ax.set_zlabel('PC 3', fontweight=FONT_WEIGHT)
	ax.set_ylabel('PC 2', fontweight=FONT_WEIGHT)

	# ax.text(0.05, 0.95, f"iteration: {iteration}", transform=ax.transAxes, fontsize=12, fontweight='bold', color='darkred')
	if iteration is not None:
		ax.text2D(0.05, 0.95, f"iteration: {iteration}", transform=ax.transAxes, fontsize=12, fontweight='bold', color='darkred')

	## Plot blue trajectory from state_traj and plot_batch_idx ##
	if state_traj is not None:
		if plot_batch_idx is None:
			plot_batch_idx = list(range(n_batch))
		
		start_points = []
		end_points = []
		for batch_idx in plot_batch_idx:
			alpha = 1
			# if not (batch_idx == 0 or batch_idx==13):
			# 	alpha=0
			x_idx = state_traj_bxtxd[batch_idx]
			z_idx = pca.transform(x_idx[plot_time_idx, :])
			start_points.append(z_idx[int((T_init) / dt)]) 
			end_points.append(z_idx[-1])

			# Phase time indices
			stim_start_idx  = int(T_init / dt)
			stim_end_idx    = int((T_init + T_stimi) / dt)
			delay_end_idx   = int((T_init + T_stimi + T_delay) / dt)
			decode_end_idx  = int((T_init + T_stimi + T_delay + T_decode) / dt)

			# Stimuli phase: yellow
			ax.plot(z_idx[stim_start_idx:stim_end_idx, 0],
					z_idx[stim_start_idx:stim_end_idx, 1],
					z_idx[stim_start_idx:stim_end_idx, 2],
					color='gold', linewidth=0.6, label='Stimuli Phase' if (batch_idx == 0) and (not plot_fps) else None, alpha=alpha)

			# Delay phase: green
			ax.plot(z_idx[stim_end_idx:delay_end_idx, 0],
					z_idx[stim_end_idx:delay_end_idx, 1],
					z_idx[stim_end_idx:delay_end_idx, 2],
					color='green', linewidth=0.6, label='Delay Phase' if (batch_idx == 0) and (not plot_fps) else None, alpha=alpha)

			# Decode phase: blue
			ax.plot(z_idx[delay_end_idx:decode_end_idx, 0],
					z_idx[delay_end_idx:decode_end_idx, 1],
					z_idx[delay_end_idx:decode_end_idx, 2],
					color='blue', linewidth=0.6, label='Decode Phase' if (batch_idx == 0) and (not plot_fps) else None, alpha=alpha)

		if not plot_fps:
			start_points = np.array(start_points) 
			end_points = np.array(end_points) 
			sc1 = ax.scatter(start_points[:, 0], start_points[:, 1], start_points[:, 2],
							c=thetas, cmap='hsv', s=30, marker='x', label='Start of Stimuli Phase' if (not plot_fps) else None, alpha=alpha)
			sc2 = ax.scatter(end_points[:, 0], end_points[:, 1], end_points[:, 2],
							c=thetas, cmap='hsv', s=30, marker='^', label='End of Decoding Phase' if (not plot_fps) else None, alpha=alpha)
			cbar = plt.colorbar(sc2, ax=ax, shrink=0.5, aspect=15)
			cbar.set_label('Target Theta')

	## Plot fixed points (red, black) ##
	if plot_fps:
		for init_idx in range(n_inits):
			plot_fixed_point(
				ax,
				fps[init_idx],
				pca,
				scale=mode_scale)

	ax.legend()

	## Save the figure from multiple angles if save_base_path is provided ##
	if save_path is not None:
		if not os.path.exists(save_path):
			os.makedirs(save_path)

		angles = range(0, 360, 45)  # Angles at 0°, 45°, 90°, ..., 315°
		for angle in angles:
			ax.view_init(elev=30, azim=angle)  # Adjust elevation and azimuth as needed
			fig.savefig(f"{save_path}/fpf_angle_{angle}_iteration_{iteration}.png", format='png', dpi=300)
			plt.close()
		with open(f'{save_path}/fpf_3d.fig.pickle', 'wb') as file:
			pickle.dump(fig, file)  

	return fig


def plot_fixed_point(ax, fp, pca,
	scale=1.0,
	max_n_modes=3,
	do_plot_unstable_fps=False,
	do_plot_stable_modes=False, # (for unstable FPs)
	stable_color='k',
	stable_marker='.',
	unstable_color='r',
	unstable_marker=None,
	**kwargs):
	'''Plots a single fixed point and its dominant eigenmodes.

	Args:
		ax: Matplotlib figure axis on which to plot everything.

		fp: a FixedPoints object containing a single fixed point
		(i.e., fp.n == 1),

		pca: PCA object as returned by sklearn.decomposition.PCA. This
		is used to transform the high-d state space representations
		into 3-d for visualization.

		scale (optional): Scale factor for stretching (>1) or shrinking
		(<1) lines representing eigenmodes of the Jacobian. Default:
		1.0 (unity).

		max_n_modes (optional): Maximum number of eigenmodes to plot.
		Default: 3.

		do_plot_stable_modes (optional): bool indicating whether or
		not to plot lines representing stable modes (i.e.,
		eigenvectors of the Jacobian whose eigenvalue magnitude is
		less than one).

	Returns:
		None.
	'''

	# --- Static flags to avoid duplicate legend entries ---
	if not hasattr(plot_fixed_point, 'plotted_stable_label'):
		plot_fixed_point.plotted_stable_label = False
	if not hasattr(plot_fixed_point, 'plotted_unstable_label'):
		plot_fixed_point.plotted_unstable_label = False


	xstar = fp.xstar
	J = fp.J_xstar
	n_states = fp.n_states

	has_J = J is not None

	if has_J:

		if not fp.has_decomposed_jacobians:
			''' Ideally, never wind up here. Eigen decomposition is much faster in batch mode.'''
			print('Decomposing Jacobians, one fixed point at time.')
			print('\t warning: THIS CAN BE VERY SLOW.')
			fp.decompose_Jacobians()

		e_vals = fp.eigval_J_xstar[0]
		e_vecs = fp.eigvec_J_xstar[0]

		sorted_e_val_idx = np.argsort(np.abs(e_vals))

		if max_n_modes > n_states:
			max_n_modes = n_states

		# Determine stability of fixed points
		is_stable = np.all(np.abs(e_vals) < 1.0)

		if is_stable:
			color = stable_color
			marker = stable_marker
			label = "Stable Fixed Point" if not plot_fixed_point.plotted_stable_label else None
			plot_fixed_point.plotted_stable_label = True
		else:
			color = unstable_color
			marker = unstable_marker
			label = "Unstable Fixed Point" if not plot_fixed_point.plotted_unstable_label else None
			plot_fixed_point.plotted_unstable_label = True
	else:
		color = stable_color
		marker = stable_marker
		label = "Stable Fixed Point" if not plot_fixed_point.plotted_stable_label else None
		plot_fixed_point.plotted_stable_label = True

	do_plot = (not has_J) or is_stable or do_plot_unstable_fps

	if do_plot:
		if has_J:
			for mode_idx in range(max_n_modes):
				# -[1, 2, ..., max_n_modes]
				idx = sorted_e_val_idx[-(mode_idx+1)]

				# Magnitude of complex eigenvalue
				e_val_mag = np.abs(e_vals[idx])

				if e_val_mag > 1.0 or do_plot_stable_modes:

					# Already real. Cast to avoid warning.
					e_vec = np.real(e_vecs[:,idx])

					# [1 x d] numpy arrays
					xstar_plus = xstar + scale*e_val_mag*e_vec
					xstar_minus = xstar - scale*e_val_mag*e_vec

					# [3 x d] numpy array
					xstar_mode = np.vstack((xstar_minus, xstar, xstar_plus))

					if e_val_mag < 1.0:
						color = stable_color
					else:
						color = unstable_color

					if n_states >= 3 and pca is not None:
						# [3 x 3] numpy array
						zstar_mode = pca.transform(xstar_mode)
					else:
						zstar_mode = xstar_mode

					plot_123d(ax, zstar_mode,
								color=color,
								**kwargs)

		if n_states >= 3 and pca is not None:
			zstar = pca.transform(xstar)
		else:
			zstar = xstar

		plot_123d(ax, zstar,
					color=color,
					marker=marker,
					markersize=12,
					label=label,
					**kwargs)

def plot_123d(ax, z, **kwargs):
	'''Plots in 1D, 2D, or 3D.

	Args:
		ax: Matplotlib figure axis on which to plot everything.

		z: [n x n_states] numpy array containing data to be plotted,
		where n_states is 1, 2, or 3.

		any keyword arguments that can be passed to ax.plot(...).

	Returns:
		None.
	'''
	n_states = z.shape[1]
	if n_states == 3:
		ax.plot(z[:, 0], z[:, 1], z[:, 2], **kwargs)
	elif n_states == 2:
		ax.plot(z[:, 0], z[:, 1], **kwargs)
	elif n_states == 1:
		ax.plot(z, **kwargs)
	else:
		raise ValueError("z should have 1, 2, or 3 columns corresponding to 1D, 2D, or 3D data.")
	


def plot_F_vs_PCA(F, hidden_state_end, thetas, pca_dir, iteration):
	# thetas: (trials,1) -> (trials,)
	thetas = thetas.detach().numpy().squeeze()
	hidden_state_end = hidden_state_end.detach().numpy().squeeze()
	F = F.detach().numpy().squeeze()

	pca = PCA(n_components=2)
	pca.fit(hidden_state_end)
	pca_points = pca.transform(hidden_state_end)  # (trials, 2*max_items)

	decode_points = (hidden_state_end @ F.T)[:,:2]


	def normalize_radius(points, target_radius):
		radii = np.linalg.norm(points, axis=1)  # 原始半径
		
		valid_radii = np.where(radii > 1e-6, radii, 1)
		scale_factors = target_radius / valid_radii
		
		normalized = np.zeros_like(points)
		normalized[:,0] = scale_factors * points[:,0]
		normalized[:,1] = scale_factors * points[:,1]
		return normalized

	# pca_points = normalize_radius(pca_points, target_radius=40)
	# decode_points = normalize_radius(decode_points, target_radius=20)


	# 2D scatter plot
	plt.figure(figsize=(5, 4))
	scatter1 = plt.scatter(
		pca_points[:, 0], pca_points[:, 1], c=thetas, cmap='rainbow',
		label='PCA Points', marker='^', s=60, edgecolor='black', linewidth=0.5
	)
	scatter2 = plt.scatter(
		decode_points[:, 0], decode_points[:, 1], c=thetas, cmap='rainbow',
		label='Decoded Points', marker='o', s=40, edgecolor='black', linewidth=0.5
	)
	cbar = plt.colorbar(scatter1)
	cbar.set_label('Target Theta')
	
	plt.axis('equal')
	plt.xlabel('Component 1')
	plt.ylabel('Component 2')
	plt.title(f'PCA vs Decoded Points (2D) - iteration {iteration}')
	plt.legend()
	plt.axhline(0, color='black', linewidth=1, linestyle='-')  # Highlight x-axis
	plt.axvline(0, color='black', linewidth=1, linestyle='-')  # Highlight y-axis
	plt.grid(True, linestyle='--', alpha=0.6)

	if not os.path.exists(pca_dir):
		os.makedirs(pca_dir)
	file_path = f'{pca_dir}/pca_vs_F_iteration_{iteration}.png'
	plt.savefig(file_path, dpi=300)
	plt.close()



def prepare_state(model):
	fpf_item_num = [1]
	# Generate presence for each group
	# input_presence = torch.zeros(fpf_trials, max_item_num, device=device)
	# trials_per_group = fpf_trials // len(fpf_item_num)  
	# remaining_trials = fpf_trials % len(fpf_item_num)  
	# trial_counts = [trials_per_group + (1 if i < remaining_trials else 0) for i in range(len(fpf_item_num))]

	# start_index = 0
	# for i, count in enumerate(trial_counts):
	#     end_index = start_index + count
	#     # Ensuring the first item (index 0) is always included
	#     random_indices = torch.stack([torch.randperm(max_item_num - 1, device=device)[:fpf_item_num[i] - 1] + 1 for _ in range(count)])
	#     one_hot_indices = torch.cat([torch.zeros(count, 1, dtype=torch.long, device=device), random_indices], dim=1)
	#     input_presence[start_index:end_index] = input_presence[start_index:end_index].scatter(1, one_hot_indices, 1)
	#     start_index = end_index

	# # Generate the orientation of first item independently for each trial
	# first_item = torch.linspace(-torch.pi, torch.pi, fpf_trials, device=device).unsqueeze(1) # (trials,1)
	# if max_item_num == 1:
	#      input_thetas = first_item
	# elif max_item_num > 1:
	#     # Generate the remaining items, which are shared across all trials
	#     shared_items = (torch.rand(1, max_item_num - 1, device=device) * 2 * torch.pi) - torch.pi # (1, max_items-1)
	#     # Concatenate the first item (unique per trial) with the shared items (same for all trials)
	#     input_thetas = torch.cat((first_item, shared_items.expand(fpf_trials, -1)), dim=1) # (trials, max_items)


	input_presence = torch.zeros(fpf_trials, max_item_num, device=device)
	input_presence[:,0] = 1
	
	theta_base = torch.linspace(-torch.pi, torch.pi, steps=fpf_trials, device=device)
	input_thetas = theta_base.unsqueeze(1).expand(-1, max_item_num)

	u_t = generate_input(
		presence=input_presence,
		theta=input_thetas,
		input_strength=input_strength,
		noise_level=0.0,
		T_init=T_init,
		T_stimi=T_stimi,
		T_delay=T_delay,
		T_decode=T_decode,
		dt=dt,
	)
	
	r_output, _ = model(u_t, r0=None)  # (trial, steps, neuron)

	return u_t.detach().cpu(), r_output.detach().cpu(), input_thetas.detach().cpu()

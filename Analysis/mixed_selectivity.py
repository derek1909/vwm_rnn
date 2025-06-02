from rnn import *
from config import *
from train import *
from utils import *

# --------------- Helper Functions ---------------
def _calculate_vector_strength(r_decode: torch.Tensor,
                                        input_thetas: torch.Tensor,
                                        input_presence: torch.Tensor):
    """
    Helper function: Calculates the vector strength for each neuron for a specific item.
    Args:
        r_decode (torch.Tensor): Neural activity tensor.
            Shape: (steps_for_loss, num_trials, num_neurons).
        input_thetas (torch.Tensor): Orientations of items for each trial.
            Shape: (num_trials, max_item_num). Values in radians.
        input_presence (torch.Tensor): Indicates which item is present on each trial.
            Shape: (num_trials, max_item_num). Boolean or 0/1.    
    Returns:
        torch.Tensor: Tensor of shape (neuron, max_item_num) representing the
                    vector strength for each neuron for each item group.
    """
    avg_r_decode = torch.mean(r_decode, dim=0)

    vector_strength_output = torch.zeros((num_neurons, max_item_num), device=device)

    for k_item in range(max_item_num):
        # Mask for trials where the k-th item was presented
        present_trials_mask = input_presence[:, k_item].bool() if input_presence.dtype != torch.bool else input_presence[:, k_item] # (num_trials,)

        # Get responses for these trials for all neurons
        avg_r_decode_k = avg_r_decode[present_trials_mask, :] # (num_present_trials, num_neurons)

        # Get thetas for this item in these trials
        thetas_k = input_thetas[present_trials_mask, k_item] # shape: (num_present_trials,)

        # Numerator part: |sum(response * e^(j*theta))|
        # e^(j*theta) = cos(theta) + j*sin(theta)
        cos_thetas = torch.cos(thetas_k)  # shape: (num_present_trials,)
        sin_thetas = torch.sin(thetas_k)  # shape: (num_present_trials,)
        cos_thetas_expanded = cos_thetas.unsqueeze(1) # (num_present_trials, 1)
        sin_thetas_expanded = sin_thetas.unsqueeze(1) # (num_present_trials, 1)

        sum_r_cos_theta = torch.sum(avg_r_decode_k * cos_thetas_expanded, dim=0) # (num_neurons,)
        sum_r_sin_theta = torch.sum(avg_r_decode_k * sin_thetas_expanded, dim=0) # (num_neurons,)
        magnitude_of_complex_sum = torch.sqrt(sum_r_cos_theta**2 + sum_r_sin_theta**2) # (num_neurons,)

        # Denominator part: sum(response)
        sum_responses = torch.sum(avg_r_decode_k, dim=0) # (num_neurons,)

        # Calculate vector strength for the current item, for all neurons
        vector_strength_output[:, k_item] =  magnitude_of_complex_sum / sum_responses

    return vector_strength_output


def _calculate_ipr(vector_strength):
    """
    Calculates the Inverse Participation Ratio (IPR) for each neuron
    from its vector strength across items.

    Formula: IPR(R_i) = (sum_k R_ik)^2 / (M_items * sum_k R_ik^2)

    Args:
        vector_strength (torch.Tensor): Tensor of shape (neuron, max_item_num)
                                        representing the vector strength (R_ik).
                                        Values should be non-negative.

    Returns:
        torch.Tensor: Tensor of shape (neuron,) representing the IPR for each neuron.
    """
    # Numerator: (sum_k R_ik)^2
    sum_R_ik = torch.sum(vector_strength, dim=1) # (neuron,)

    # Denominator: M_items * sum_k R_ik^2
    R_ik_squared = vector_strength**2 # (neuron, max_item_num)
    sum_R_ik_squared = torch.sum(R_ik_squared, dim=1) # (neuron,)

    # Calculate IPR
    ipr = sum_R_ik**2 / (max_item_num * sum_R_ik_squared) # (neuron,)
    return ipr


def _visualize_R_matrix(R_matrix, title="Vector Strength (R_matrix) Heatmap", xlabel="Item Index", ylabel="Neuron Index", cmap="viridis", output_file="R_matrix.png"):
    """
    Visualizes the R_matrix (vector_strength tensor) as a heatmap and saves it to a file.

    Args:
        R_matrix (torch.Tensor): The 2D matrix to visualize, (neuron, max_item_num).
        title (str): The title for the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        cmap (str): Colormap to use for the heatmap.
        output_filename (str): The filename to save the plot to (e.g., "Rik.png").
    """

    R_matrix_np = R_matrix.detach().cpu().numpy()

    fig = plt.figure(figsize=(5, 4))
    plt.imshow(R_matrix_np, aspect='auto', cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Vector Strength')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    
    # Save the figure
    plt.savefig(output_file)
    plt.close(fig)


def _visualize_R_matrix_and_ipr(R_matrix, ipr_values, 
                                     title_heatmap="Vector Strength (R_matrix.T) Heatmap", 
                                     xlabel_heatmap="Neuron Index",  # X-axis of heatmap is Neuron Index due to transpose
                                     ylabel_heatmap="Item Index",    # Y-axis of heatmap is Item Index due to transpose
                                     cmap="viridis", 
                                     output_file="R_matrix_T_with_IPR.png"):
    """
    Visualizes R_matrix.T as a heatmap and IPR values as a line plot above it,
    in a single figure, aligned by neurons. Saves the figure to a file.

    Args:
        R_matrix (torch.Tensor): The 2D vector strength matrix (neuron, max_item_num).
        ipr_values (torch.Tensor): 1D tensor of IPR values (neuron,).
        title_heatmap (str): Title for the heatmap subplot.
        xlabel_heatmap (str): Label for the x-axis of the heatmap (shared with IPR plot).
        ylabel_heatmap (str): Label for the y-axis of the heatmap.
        cmap (str): Colormap for the heatmap.
        output_filename (str): Filename to save the plot.
    """
    # Prepare data for plotting
    R_matrix_T_np = R_matrix.detach().cpu().numpy().T  # Transpose for plotting
    ipr_values_np = ipr_values.detach().cpu().numpy()
    
    neuron_indices = np.arange(num_neurons)

    # Create figure with two subplots: IPR curve on top, R_matrix.T heatmap below
    # sharex=True links the x-axis (neuron index) of both subplots
    fig, axs = plt.subplots(2, 1, figsize=(5, 4), sharex=True, gridspec_kw={'height_ratios': [1, 2]}, constrained_layout=True)

    # Top subplot: IPR curve
    axs[0].plot(neuron_indices, ipr_values_np, color='dodgerblue')
    # axs[0].set_title('IPR per Neuron')
    axs[0].set_ylabel('IPR Value')
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    if num_neurons > 1 : # Avoid issues with xlim for single neuron
         axs[0].set_xlim([0, num_neurons - 1])


    # Bottom subplot: R_matrix.T heatmap
    # R_matrix_T_np has shape (max_item_num, num_neurons)
    im = axs[1].imshow(R_matrix_T_np, aspect='auto', cmap=cmap, interpolation='nearest')
    # axs[1].set_title(title_heatmap)
    axs[1].set_xlabel(xlabel_heatmap) # "Neuron Index"
    axs[1].set_ylabel(ylabel_heatmap) # "Item Index"
    
    # Add colorbar for the heatmap
    # Adjust fraction and pad to make colorbar fit nicely
    cbar = fig.colorbar(im, ax=axs[1], label='Vector Strength', orientation='vertical', fraction=0.046, pad=0.04)

    # plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle if needed, and prevent overlap
    
    # Save the figure
    plt.savefig(output_file, dpi=200)
    plt.close(fig)


def _visualize_ipr_histogram(ipr_values, title="Histogram of IPR Values", xlabel="IPR Value", ylabel="Number of Neurons (Count)", bins=20, output_file="ipr_histogram.png"):
    """
    Visualizes the IPR values as a histogram and saves it to a file.

    Args:
        ipr_values (torch.Tensor or np.ndarray): A 1D array or tensor of IPR values.
        title (str): The title for the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        bins (int or sequence or str): Number of histogram bins or bin specification.
        output_filename (str): The filename to save the plot to.
    """

    ipr_values_np = ipr_values.detach().cpu().numpy()

    # Create a new figure
    fig = plt.figure(figsize=(5, 4))
    plt.hist(ipr_values_np, bins=bins, color='skyblue', edgecolor='black')
    # plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75) # Add a light grid on y-axis
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file)
    plt.close(fig)

    
# --------------- Main Analysis Function ---------------
def mixed_selectivity_analysis(model):
    """
    Performs mixed selectivity analysis.
    Calculates Vector Strength (R_ik) for orientation tuning per item slot
    (assuming set size = 1), and then the IPR of these R_ik values per neuron.

    Returns:
        tuple: (R_matrix, ipr_R_values_per_neuron)
            - R_matrix (np.ndarray): Matrix of Vector Strengths.
                Shape: (num_neurons, max_item_num).
            - ipr_R_values_per_neuron (np.ndarray): Array of IPR values for each neuron's R_i.
                Shape: (num_neurons,).
    """

    # ------------------ bookkeeping / dirs ------------------
    out_dir = f'{model_dir}/mixed_selec'
    os.makedirs(out_dir, exist_ok=True)
    Rik_png = f'{out_dir}/R_matrixT_with_IPR.png'
    ipr_png = f'{out_dir}/ipr_hist.png'

    # ------------------ generate test data ------------------
    item_num = [1] # Assume set size = 1
    num_trials = 1000

    # Split num_trials into len(item_num) groups
    trials_per_group = num_trials // len(item_num)
    remaining_trials = num_trials % len(item_num)
    trial_counts = [trials_per_group + (1 if i < remaining_trials else 0) for i in range(len(item_num))]

    # ---- random presence matrix ----
    input_presence = torch.zeros(num_trials, max_item_num, device=device, requires_grad=False)
    start = 0
    for idx, cnt in enumerate(trial_counts):
        end   = start + cnt
        one_hot_inds  = torch.stack([
            torch.randperm(max_item_num, device=device)[:item_num[idx]]
            for _ in range(cnt)])
        input_presence[start:end] = input_presence[start:end].scatter(1, one_hot_inds, 1)
        start = end

    # ---- random orientation & input tensor ----
    input_thetas = (torch.rand(num_trials, max_item_num, device=device) * 2*torch.pi) - torch.pi
    u_t          = generate_input(input_presence, input_thetas,
                                  input_strength, ILC_noise,
                                  T_init, T_stimi, T_delay, T_decode, dt)

    # ------------------ forward pass ------------------
    with torch.no_grad():
        r_out, _   = model(u_t, r0=None)                 # (trial, steps, neuron)
    step_thr  = int((T_init + T_stimi + T_delay) / dt)
    r_decode  = r_out[:, step_thr:, :].permute(1, 0, 2).clone()  # (steps_for_loss, trial, neuron)
    del r_out, u_t; torch.cuda.empty_cache(); gc.collect()

    # ------------------ vector strength and IPR calculation ------------------
    R_matrix = _calculate_vector_strength(r_decode, input_thetas, input_presence)
    ipr = _calculate_ipr(R_matrix)
            
    # ------------------ visualisation and save ------------------
    _visualize_R_matrix_and_ipr(R_matrix, ipr, output_file=Rik_png)
    _visualize_ipr_histogram(ipr, output_file=ipr_png)
    return R_matrix, ipr
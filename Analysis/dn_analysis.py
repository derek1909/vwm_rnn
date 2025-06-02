from rnn import *
from config import *
from train import *
from utils import *


# --------------- Main Analysis Function ---------------
def divisive_normalisation_analysis(model):
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
    out_dir = f'{model_dir}/divis_normal'
    os.makedirs(out_dir, exist_ok=True)
    DN_b_png = f'{out_dir}/DN_sublinearity.png'
    # ipr_png = f'{out_dir}/ipr_hist.png'

    # ------------------ generate test data ------------------
    num_trials = 100
    # zeros-initialised templates
    input_presence1  = torch.zeros(num_trials, max_item_num, device=device, requires_grad=False)
    input_presence2  = torch.zeros_like(input_presence1, requires_grad=False)
    input_presence12 = torch.zeros_like(input_presence1, requires_grad=False)

    # set the desired positions to 1
    input_presence1[:, 0] = 1                      # item-1 only
    input_presence2[:, 1] = 1                      # item-2 only
    input_presence12[:, (0, 1)] = 1                # items-1 and 2

    # ---- random orientation & input tensor ----
    input_thetas = (torch.rand(num_trials, max_item_num, device=device) * 2*torch.pi) - torch.pi
    h1          = generate_input(input_presence1, input_thetas,
                                input_strength, ILC_noise,
                                T_init, T_stimi, T_delay, T_decode, dt)
    h2          = generate_input(input_presence2, input_thetas,
                                input_strength, ILC_noise,
                                T_init, T_stimi, T_delay, T_decode, dt)    
    h12          = generate_input(input_presence12, input_thetas,
                                input_strength, ILC_noise,
                                T_init, T_stimi, T_delay, T_decode, dt)

    # ------------------ forward pass ------------------
    with torch.no_grad():
        r_out1, _   = model(h1, r0=None)                 # (trial, steps, neuron)
        r_out2, _   = model(h2, r0=None)                 # (trial, steps, neuron)
        r_out12, _   = model(h12, r0=None)                 # (trial, steps, neuron)

    step_thr  = int((T_init + T_stimi + T_delay) / dt)
    r_decode1  = r_out1[:, step_thr:, :].permute(1, 0, 2).clone()  # (steps_for_loss, trial, neuron)
    r_decode2  = r_out2[:, step_thr:, :].permute(1, 0, 2).clone()  # (steps_for_loss, trial, neuron)
    r_decode12  = r_out12[:, step_thr:, :].permute(1, 0, 2).clone()  # (steps_for_loss, trial, neuron)
    del r_out1, r_out2, r_out12, h1, h2, h12 ; torch.cuda.empty_cache(); gc.collect()

    # Step 1: Average over time and trials
    r1_bar = r_decode1.mean(dim=0)  # shape: (trial, neurons,)
    r2_bar = r_decode2.mean(dim=0)  # shape: (trial, neurons,)
    r12_bar = r_decode12.mean(dim=0)  # shape: (trial, neurons,)

    # Step 2: Stack r1 and r2 to form X: shape (neurons, 2)
    X = torch.stack([r1_bar, r2_bar], dim=2)  # shape: (trial, neurons, 2)
    y = r12_bar.unsqueeze(2)                 # shape:  (trial, neurons, 1)
    
    # Step 3: Solve for W using least squares: W = (X^T X)^-1 X^T r12
    # This is equivalent to: W = torch.linalg.lstsq(X, r12_bar).solution
    # But for compatibility we use the closed form
    Xt = X.transpose(1, 2) #  (trial, neurons, 2)
    XtX = Xt @ X  # shape: (trial, 2, 2)
    Xt_y = Xt @ y # shape: (trial, 2, 1)

    W = torch.linalg.solve(XtX, Xt_y).squeeze(2)  # shape: (trial, 2)

    w1_vals = W[:, 0].cpu().numpy()
    w2_vals = W[:, 1].cpu().numpy()

    x_min = min(w1_vals.min(), 0)
    y_min = min(w2_vals.min(), 0)
    x_max = max(w1_vals.max(), 1.2)
    y_max = max(w2_vals.max(), 1.2)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Scatter plot
    plt.figure(figsize=(5, 5))
    plt.fill_betweenx([0, 1], 0, 1, color='gray', alpha=0.2, label='DN Region')
    plt.scatter(w1_vals, w2_vals, c='orange', alpha=0.7, edgecolor='k')

    # Highlight DN region: 0 ≤ w1 ≤ 1 and 0 ≤ w2 ≤ 1

    # Guideline: w1 + w2 = 1
    # plt.axvline(x=1, color='red', linestyle='--', linewidth=1, label=r'$w_1 = 1$')
    # plt.axhline(y=1, color='blue', linestyle='--', linewidth=1, label=r'$w_2 = 1$')

    plt.xlabel(r'$w_1$')
    plt.ylabel(r'$w_2$')
    # plt.title('Scatter of $(w_1, w_2)$ over $(\theta_1, \theta_2)$ combinations')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(DN_b_png, dpi=200)
    plt.close()

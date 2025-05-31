
from rnn import *
from config import *
from train import *
from utils import *
import yaml


# ---------- helper: VM + uniform negative-log-likelihood ----------
def _fit_vm_uniform(torch_errs, init_w=0.5, init_kappa=5.0,
                    lr=5e-2, epochs=800):
    raw_w     = torch.tensor([torch.logit(torch.tensor(init_w))],
                                requires_grad=True, device=device)
    raw_kappa = torch.tensor([init_kappa], device=device, requires_grad=True)

    optim = torch.optim.Adam([raw_w, raw_kappa], lr=lr)
    log_2pi = torch.log(torch.tensor(2 * math.pi, device=device))
    for _ in range(epochs):
        optim.zero_grad()
        w     = torch.sigmoid(raw_w)                        # (0,1)
        kappa = torch.nn.functional.softplus(raw_kappa)     # >0

        log_vm   = kappa*torch.cos(torch_errs) \
                    - log_2pi \
                    - torch.log(torch.special.i0(kappa))
        log_unif = -log_2pi

        a = torch.log(w)       + log_vm
        b = torch.log1p(-w)    + log_unif
        m = torch.max(a,b)
        logp = m + torch.log(torch.exp(a-m) + torch.exp(b-m))
        nll  = -logp.mean()
        nll.backward(); optim.step()

    return torch.sigmoid(raw_w).item(), torch.nn.functional.softplus(raw_kappa).item()

# ---------- helper: Gaussian + uniform negative-log-likelihood ----------
def _fit_gauss_uniform(torch_errs,
                    init_w=0.5,          # mixture weight on the Gaussian
                    init_sigma=0.5,      # Gaussian std-dev (rad)
                    lr=5e-2,
                    epochs=800):
    """
    Maximum-likelihood fit of a mixture model:
        p(err) = w  ·  N(0, σ²)  +  (1-w) · U(-π, π)

    Returns
    -------
    w, sigma : torch.Tensor scalars (on the current device)
    nll      : final mean negative log-likelihood
    """
    # ── unconstrained parameters ──────────────────────────────────────────
    raw_w     = torch.tensor([torch.logit(torch.tensor(init_w))],
                            requires_grad=True, device=torch_errs.device)
    raw_sigma = torch.tensor([init_sigma],
                            requires_grad=True, device=torch_errs.device)

    optimiser = torch.optim.Adam([raw_w, raw_sigma], lr=lr)
    log_2pi   = torch.log(torch.tensor(2 * math.pi, device=torch_errs.device))

    for _ in range(epochs):
        optimiser.zero_grad()

        # positive / bounded re-parameterisations
        w     = torch.sigmoid(raw_w)                 # 0 < w < 1
        sigma = torch.nn.functional.softplus(raw_sigma)  # σ > 0
        inv_2sigma2 = 0.5 / (sigma * sigma)

        # log-pdf of N(0,σ²)  (vectorised over err)
        log_gauss = (
            - inv_2sigma2 * torch_errs.pow(2)
            - torch.log(sigma)
            - 0.5 * log_2pi
        )

        # log-pdf of U(-π,π)
        log_unif = -log_2pi

        # numerically stable log-sum-exp for mixture
        a = torch.log(w)      + log_gauss
        b = torch.log1p(-w)   + log_unif
        m = torch.max(a, b)
        logp = m + torch.log(torch.exp(a - m) + torch.exp(b - m))

        nll = -logp.mean()
        nll.backward()
        optimiser.step()

    # return final parameters on the same device
    w_final     = torch.sigmoid(raw_w.detach())
    sigma_final = torch.nn.functional.softplus(raw_sigma.detach())
    return w_final.item(), sigma_final.item()
    
# ---------- helper: G circular SD from mean resultant length R̄ ----------
def _circ_sd(Rbar: float) -> float:
    Rbar = max(min(Rbar, 0.999999), 1e-12)      # clamp to (0,1)
    return math.sqrt(-2.0 * math.log(Rbar))

def error_dist_analysis(model):
    """
    Plot error histograms for each set size.  Optionally (fit_mixture_bool=True)
    fit a von-Mises + uniform mixture and save parameters + figure.
    """
    # ------------------ bookkeeping / dirs ------------------
    out_dir = f'{model_dir}/error_dist'
    os.makedirs(out_dir, exist_ok=True)
    hist_png  = f'{out_dir}/error_hist.png'
    var_png  = f'{out_dir}/error_var.png'
    kurt_png  = f'{out_dir}/error_kurt.png'
    # vm_fit_png   = f'{out_dir}/vm_error_fit.png'
    # gauss_fit_png   = f'{out_dir}/gauss_error_fit.png'
    # vm_sd_png = f'{out_dir}/vm_sd_compare.png'
    # gauss_sd_png = f'{out_dir}/gauss_sd_compare.png'
    # w_png = f'{out_dir}/uniform_w_compare.png'
    yaml_path = f'{out_dir}/fit_summary.yaml'

    # ------------------ generate test data ------------------
    torch.cuda.empty_cache()
    num_trials = 8000
    if   max_item_num == 1:  item_num = [1]
    elif max_item_num == 2:  item_num = [1, 2]
    elif max_item_num < 8:   item_num = list(range(1, max_item_num + 1, 2))
    else:                    item_num = [8, 4, 2, 1]

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

    # ---- decode ----
    u_hat         = model.readout(r_decode.reshape(-1, num_neurons))  # (steps_for_loss*trial, max_item_num * 2)
    decoded_theta = model.decode(u_hat.reshape(r_decode.shape[0], num_trials, -1))  # (trials, max_items)
    angular_diff  = (input_thetas - decoded_theta + torch.pi) % (2*torch.pi) - torch.pi  # (trials, max_items)
    del r_decode, u_hat; torch.cuda.empty_cache(); gc.collect()

    # ------------------ plot: raw histograms ------------------
    fig_raw = plt.figure(figsize=(6, 5))
    x_vals  = np.linspace(-np.pi, np.pi, 200)
    colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']

    err_sets, summary = [], {}   # keep per-condition errors for later fitting
    start = 0
    for idx, cnt in enumerate(trial_counts):
        end  = start + cnt
        mask = input_presence[start:end].bool()
        err  = angular_diff[start:end][mask].detach().cpu().numpy()
        err_sets.append(err)

        hist, bins = np.histogram(err, bins=x_vals, density=True)
        centers    = 0.5*(bins[:-1] + bins[1:])
        plt.plot(centers, hist, label=f'{item_num[idx]} item(s)', color=colors[idx])
        start = end

    plt.xlim(-np.pi, np.pi)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.xlabel('Angular Error (rad)')
    plt.ylabel('Probability Density')
    plt.title('Distribution of Decoding Errors')
    plt.legend()
    plt.ylim(bottom=0)
    fig_raw.tight_layout()
    fig_raw.savefig(hist_png, dpi=300)
    plt.close(fig_raw)
    # print(f"Raw histograms saved to: {hist_png}")

    def circular_variance(e):
        m1 = np.mean(np.exp(1j * e))
        return -2 * np.log(np.abs(m1))

    def circular_kurtosis(e):
        m1 = np.mean(np.exp(1j * e))
        m2 = np.mean(np.exp(2j * e))
        term1 = np.abs(m2) * np.cos(np.angle(m2) - 2 * np.angle(m1))
        term2 = np.abs(m1)**4
        denom = (1 - np.abs(m1))**2
        return (term1 - term2) / denom

    # Compute stats for each condition
    for i, err in enumerate(err_sets):
        n = item_num[i]
        summary[n] = {
            "circular_variance": float(circular_variance(err)),
            "circular_kurtosis": float(circular_kurtosis(err))
        }

    # ----------------- Load human data from YAML -----------------
    human_yaml_path = "/homes/jd976/working/vwm_rnn/Analysis/behavior_summary.yaml"
    with open(human_yaml_path, "r") as f:
        human_data = yaml.safe_load(f)

    # Extract data
    human_item_num = sorted(human_data.keys(), key=lambda x: int(x))
    human_variances = [human_data[int(k)]["circular_variance_mean"] for k in human_item_num]
    human_kurtoses = [human_data[int(k)]["circular_kurtosis_mean"] for k in human_item_num]
    human_var_se = [human_data[int(k)]["circular_variance_se"] for k in human_item_num]
    human_kurt_se = [human_data[int(k)]["circular_kurtosis_se"] for k in human_item_num]

    # ----------------- Mock model output (replace with your model-derived stats) -----------------
    model_variances = [summary[int(k)]["circular_variance"] for k in item_num]
    model_kurtoses = [summary[int(k)]["circular_kurtosis"] for k in item_num]

    # ----------------- Plot comparison -----------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6))

    # ------------------ Plot 2: Circular Variance ------------------
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.errorbar(human_item_num, human_variances, yerr=human_var_se, fmt='ko', capsize=3, label="Human")
    ax1.plot(item_num, model_variances, 'r', label="Model")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks(item_num)
    ax1.set_xticklabels([str(s) for s in item_num])
    ax1.set_ylabel('circular variance ($\\sigma^2$)')
    ax1.set_xlabel('items')
    ax1.set_title('width, $\\omega$')
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(var_png, dpi=300)

    # ------------------ Plot 3: Circular Kurtosis ------------------
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.errorbar(human_item_num, human_kurtoses, yerr=human_kurt_se, fmt='ko', capsize=3, label="Human")
    ax2.plot(item_num, model_kurtoses, 'r', label="Model")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xticks(item_num)
    ax2.set_xticklabels([str(s) for s in item_num])
    ax2.set_ylabel('circular kurtosis')
    ax2.set_xlabel('items')
    ax2.set_title('kurtosis')
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(kurt_png, dpi=300)

    # ---------- save YAML summary ----------
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)
    # tidy GPU
    del angular_diff, input_presence, input_thetas
    torch.cuda.empty_cache(); gc.collect()

"""
    # ==================================================================
    # =========================  mixture fitting  ======================
    # ==================================================================
    if not fit_mixture_bool:
        return

    # ---------- perform vm+uniform fits & create overlay plot ----------
    vm_fig_fit = plt.figure(figsize=(6, 5))
    x_dense = torch.linspace(-np.pi, np.pi, 512, device=device)

    for idx, err_np in enumerate(err_sets):
        errs = torch.from_numpy(err_np).to(device).float()

        w, kappa = _fit_vm_uniform(errs)
        summary[item_num[idx]].update({"vm_w": w, "vm_kappa": kappa})

        # empirical hist for overlay
        hist, bins = np.histogram(err_np, bins=x_vals, density=True)
        centers    = 0.5*(bins[:-1] + bins[1:])
        plt.plot(centers, hist, label=f'{item_num[idx]} item(s)', color=colors[idx])

        # fitted curve
        vm  = (torch.exp(kappa*torch.cos(x_dense)) /
               (2*torch.pi*torch.special.i0(torch.tensor(kappa, device=device))))
        unif = 1/(2*np.pi)
        pdf  = w*vm + (1-w)*unif
        plt.plot(x_dense.cpu().numpy(), pdf.cpu().numpy(),
                 '--', color=colors[idx], alpha=0.8,
                 label=f'{item_num[idx]} item(s) • fit')

    plt.xlim(-np.pi, np.pi)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.xlabel('Angular Error (rad)')
    plt.ylabel('Probability Density')
    plt.title('Error Histograms with VM + Uniform Fits')
    plt.legend(ncol=2, fontsize=8)
    plt.ylim(bottom=0)
    vm_fig_fit.tight_layout()
    vm_fig_fit.savefig(vm_fit_png, dpi=300)
    plt.close(vm_fig_fit)

    # ---------- perform gauss+uniform fits & create overlay plot ----------
    gauss_fig_fit = plt.figure(figsize=(6, 5))
    x_dense = torch.linspace(-np.pi, np.pi, 512, device=device)

    for idx, err_np in enumerate(err_sets):
        errs = torch.from_numpy(err_np).to(device).float()

        w, sigma = _fit_gauss_uniform(errs)
        summary[item_num[idx]].update({"gauss_w": w, "gauss_sigma": sigma})

        # empirical hist for overlay
        hist, bins = np.histogram(err_np, bins=x_vals, density=True)
        centers    = 0.5*(bins[:-1] + bins[1:])
        plt.plot(centers, hist, label=f'{item_num[idx]} item(s)', color=colors[idx])

        # fitted curve
        gauss = torch.exp(-0.5 * (x_dense / sigma) ** 2) / ( sigma * math.sqrt(2 * math.pi) )
        unif = 1/(2*np.pi)
        pdf  = w*gauss + (1-w)*unif
        plt.plot(x_dense.cpu().numpy(), pdf.cpu().numpy(),
                 '--', color=colors[idx], alpha=0.8,
                 label=f'{item_num[idx]} item(s) • fit')

    plt.xlim(-np.pi, np.pi)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.xlabel('Angular Error (rad)')
    plt.ylabel('Probability Density')
    plt.title('Error Histograms with Gauss + Uniform Fits')
    plt.legend(ncol=2, fontsize=8)
    plt.ylim(bottom=0)
    gauss_fig_fit.tight_layout()
    gauss_fig_fit.savefig(gauss_fit_png, dpi=300)
    plt.close(gauss_fig_fit)
    # print(f"Fit overlay figure saved to: {vm_fit_png} and {gauss_fit_png}")


    # ----------------------------------------------------------
    # ----------  raw-vs-mixture circular standard dev  --------
    # ----------------------------------------------------------

    raw_line_sds    = [summary[n]['raw_line_std'] for n in item_num]
    raw_circ_sds    = [summary[n]['raw_circ_std'] for n in item_num]
    uni_w_vm    = [1-summary[n]['vm_w'] for n in item_num]
    uni_w_gauss    = [1-summary[n]['gauss_w'] for n in item_num]
    mix_vm_sds    = []

    for n in item_num:
        w      = summary[n]['vm_w']
        kappa  = summary[n]['vm_kappa']

        # mean resultant length of the von-Mises part: R = I1/I0
        R      = (torch.special.i1(torch.tensor(kappa)) /
                torch.special.i0(torch.tensor(kappa))).item()
        # Rbar   = w * R                               # uniform part contributes 0
        sd_cm  = _circ_sd(R)                       # circular SD of mixture

        summary[n]['mix_vm_circ_std'] = sd_cm
        mix_vm_sds.append(sd_cm)

    # ---------- VM SD comparison figure  ----------
    fig_sd = plt.figure(figsize=(5, 4))
    plt.plot(item_num, raw_circ_sds, 'o-',  label='Empirical circ SD')
    plt.plot(item_num, mix_vm_sds, 's--', label='VM circ SD')
    plt.xlabel('Number of items')
    plt.ylabel('Circular SD (rad)')
    plt.title('Raw vs. VM SD')
    plt.legend()
    plt.tight_layout()

    fig_sd.savefig(vm_sd_png, dpi=300)
    plt.close(fig_sd)

    # ---------- Gauss SD comparison figure  ----------
    fig_sd = plt.figure(figsize=(5, 4))
    plt.plot(item_num, raw_line_sds, 'o-',  label='Empirical SD')
    plt.plot(item_num, mix_vm_sds, 's--', label='Gauss SD')
    plt.xlabel('Number of items')
    plt.ylabel('Linear SD (rad)')
    plt.title('Raw vs. Gauss SD')
    plt.legend()
    plt.tight_layout()

    fig_sd.savefig(gauss_sd_png, dpi=300)
    plt.close(fig_sd)
    # print(f"SD comparison figure saved to: {vm_sd_png} and {gauss_sd_png}")

    # ---------- VM vs Gauss 1-w figure  ----------
    fig_w = plt.figure(figsize=(5, 4))
    plt.plot(item_num, uni_w_vm, 'o-',  label='uniform weight (vm)')
    plt.plot(item_num, uni_w_gauss, 's--', label='uniform weight (gauss)')
    plt.xlabel('Number of items')
    plt.ylabel('weight')
    plt.title('Uniform Weight')
    plt.ylim([0,1])
    plt.legend()
    plt.tight_layout()

    fig_w.savefig(w_png, dpi=300)
    plt.close(fig_w)
    # print(f"Uniform weight comparison figure saved to: {w_png}")
"""

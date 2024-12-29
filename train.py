import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from rnn import *
from config import *
from utils import save_model_and_history, generate_target, generate_input


def memory_loss_integral(F, r_stack, u_0, presence, lambda_err=1.0, lambda_reg=0.1):
    """
    Calculates the total loss, activation penalty, and variance of group error.
    * r_stack = (steps, trials, neurons)
    * u_0 = (trials, max_items*2)
    * presence = (trials, max_items)
    """
    num_steps, num_trials, num_neurons = r_stack.shape
    u_hat_stack = decode(F, r_stack.reshape(-1, num_neurons)).reshape(num_steps, num_trials, -1)

    # Calculate the squared error for each trial
    # (steps,trials,items) -> (trials,)
    error_per_trial = (u_0 - u_hat_stack * presence.repeat_interleave(2, dim=1)).pow(2).sum(dim=(0, 2)) / \
                      torch.sum(presence, dim=1) / num_steps  # Average over time steps and dimensions

    # Mean error across all trials
    mean_error = lambda_err * error_per_trial.mean()

    # Variance of the error across trials (actually devided by trials-1)
    variance_error = error_per_trial.var()

    # Activation penalty
    activation_penalty = lambda_reg * r_stack.abs().mean()

    total_loss = mean_error + activation_penalty
    return total_loss, activation_penalty, mean_error, variance_error

def train(model, model_dir, history=None):
    optimizer = optim.Adam(model.parameters(), lr=eta)

    # If no history is provided, initialize empty history
    if history is None:
        history = {
            "error_per_epoch": [],  # Overall mean error per epoch
            "error_std_per_epoch": [],  # std of overall error per epoch
            "activation_per_epoch": [],
            "group_errors": [[] for _ in item_num],  # List to store errors for each 'set size' group
            "group_std": [[] for _ in item_num],  # List to store std of errors for each group
            "group_activ": [[] for _ in item_num],  # List to store std of errors for each group
        }

    # Generate input_thetas
    # input_thetas = ((torch.rand(num_trials, max_item_num) * 2 * torch.pi) - torch.pi).requires_grad_()
    # input_thetas = torch.linspace(-torch.pi, torch.pi, num_trials).unsqueeze(1).repeat(1, max_item_num).requires_grad_()

    # Split num_trials into len(num_item) groups
    trials_per_group = num_trials // len(item_num)  # Ensure equal split
    remaining_trials = num_trials % len(item_num)  # Handle leftover trials
    # Adjust trials count for each group (distribute leftovers)
    trial_counts = [trials_per_group + (1 if i < remaining_trials else 0) for i in range(len(item_num))]

    with tqdm(total=num_epochs, desc="Training Progress", unit="epoch") as pbar_epoch:
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Generate presence for each group
            input_presence = torch.zeros(num_trials, max_item_num, device=device, requires_grad=True)
            start_index = 0
            for i, count in enumerate(trial_counts):
                end_index = start_index + count
                one_hot_indices = torch.stack([
                    torch.randperm(max_item_num, device=device)[:item_num[i]] for _ in range(count)
                ])
                input_presence_temp = input_presence.clone()
                input_presence_temp[start_index:end_index] = input_presence_temp[start_index:end_index].scatter(1, one_hot_indices, 1)
                input_presence = input_presence_temp
                start_index = end_index

            # Update input_thetas every 20 epochs
            if epoch % 20 == 0:
                input_thetas = ((torch.rand(num_trials, max_item_num, device=device) * 2 * torch.pi) - torch.pi).requires_grad_()

            # Generate input tensor for all trials and time steps. (num_trials, steps, 2 * max_item_num)
            u_t = generate_input(
                presence=input_presence,
                theta=input_thetas,
                noise_level=encode_noise,
                T_init=T_init,
                T_stimi=T_stimi,
                T_delay=T_delay,
                T_decode=T_decode,
                dt=dt,
                alpha=positive_input
            )
            
            r_output, _ = model(u_t, r0=None) # (trial, steps, neuron)

            step_threshold = int((T_init + T_stimi + T_delay) / dt)
            r_loss = r_output[:, step_threshold:, :].transpose(0, 1)  # (steps_for_loss, trial, neuron)

            u_0 = generate_target(input_presence, input_thetas, stimuli_present=True, alpha=0)  # u_0 has no noise

            # Calculate total loss and group-wise errors
            total_loss, total_activ_penal, total_error, total_error_var = memory_loss_integral(
                model.F, r_loss, u_0, input_presence,
                lambda_err=lambda_err, lambda_reg=lambda_reg
            )
            history["error_per_epoch"].append(total_error.item())
            history["error_std_per_epoch"].append(total_error_var.sqrt().item())
            history["activation_per_epoch"].append((total_activ_penal / lambda_reg).item())

            total_loss.backward()
            optimizer.step()

            if model.positive_input >= 1:
                model.B.data = F.relu(model.B.data)  # Ensure B is non-negative

            if epoch%logging_period == 0:
                # Calculate group-wise mean error and variance
                start_index = 0
                for i, count in enumerate(trial_counts):
                    end_index = start_index + count
                    group_r_stack = r_loss[:, start_index:end_index]
                    group_u_0 = u_0[start_index:end_index]
                    group_presence = input_presence[start_index:end_index]

                    _, group_activ_penal, group_error, group_variance = memory_loss_integral(
                        model.F, group_r_stack, group_u_0, group_presence,
                        lambda_err=lambda_err, lambda_reg=lambda_reg
                    )

                    history["group_errors"][i].append(group_error.item())
                    history["group_activ"][i].append((group_activ_penal/lambda_reg).item())
                    history["group_std"][i].append(group_variance.sqrt().item())  # Record std (sqrt of variance)

                    start_index = end_index

                # Update progress bar
                pbar_epoch.set_postfix({
                    "Error": f"{history['error_per_epoch'][-1]:.4f}",
                    "Avg Activ": f"{history['activation_per_epoch'][-1]:.4f}"
                })
                pbar_epoch.update(logging_period)

            # Save model and history every logging_period*10 epochs
            if epoch% (logging_period*10) == 0:
                save_model_and_history(model, history, model_dir)

    return history
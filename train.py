import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from early_stopping_pytorch.early_stopping import EarlyStopping
import yaml

from rnn import *
from config import *
from utils import save_model_and_history, generate_target, generate_input


def calc_train_error(u_hat_stack, u_0, presence):
    """
    Calculate training error, variance of the error, and penalization.

    Args:
        u_hat_stack (torch.Tensor): (steps, trials, max_items*2) Decoded outputs over time.
        u_0 (torch.Tensor): (trials, max_items*2) Ground truth target.
        presence (torch.Tensor): (trials, max_items) Binary mask indicating presence of items.

    Returns:
        mean_error (torch.Tensor): scalar. Mean error across trials.
        var_error (torch.Tensor): scalar. Variance of errors across trials.
    """
    # Compute squared error for each trial
    expanded_presence = presence.repeat_interleave(2, dim=1)
    error_per_trial = ((u_0 - u_hat_stack.mean(dim=0))**2 * expanded_presence).sum(dim=1) / presence.sum(dim=1)

    # Compute mean and variance of error across trials
    mean_error = error_per_trial.mean()
    var_error = error_per_trial.var()

    return mean_error, var_error

def calc_eval_error(u_hat, target_thetas, presence):
    """
    Calculate evaluation-specific angular error.

    Args:
        u_hat (torch.Tensor): (steps, trials, max_items*2) Decoded outputs over time.
        target_thetas (torch.Tensor): (trials, max_items) Ground truth angles.
        presence (torch.Tensor): (trials, max_items) Binary mask indicating presence of items.

    Returns:
        mean_ang_error (torch.Tensor): scalar. Mean angular error across trials.
        var_ang_error (torch.Tensor): scalar. Variance of angular errors across trials.
    """
    # Reshape u_hat into (steps, trials, max_items, 2)
    u_hat_reshaped = u_hat.view(u_hat.shape[0], u_hat.shape[1], -1, 2)

    # Compute angles from decoded (cos, sin) pairs
    cos_thetas = u_hat_reshaped[..., 0]  # (steps, trials, max_items)
    sin_thetas = u_hat_reshaped[..., 1]  # (steps, trials, max_items)
    decoded_thetas = torch.atan2(sin_thetas, cos_thetas)  # (steps, trials, max_items)

    # Compute angular difference
    # (steps,trials,items) -> (trials,)
    angular_diff = (target_thetas - decoded_thetas.mean(dim=0) + torch.pi) % (2 * torch.pi) - torch.pi  # (-pi,pi)
    angular_error_per_trial = (angular_diff.abs() * presence).sum(dim=1) / presence.sum(dim=1)

    # Compute mean and variance of angular errors
    mean_ang_error = angular_error_per_trial.mean()
    var_ang_error = angular_error_per_trial.var()

    return mean_ang_error, var_ang_error

def error_calc(F, r_stack, target_thetas, presence, train_err=True):
    """
    Calculates training error, evaluation error, and activation penalty.

    Args:
        F (torch.Tensor): Decoding matrix of shape (neurons, output_dim).
        r_stack (torch.Tensor): Neural activity tensor of shape (steps, trials, neurons).
        target_thetas (torch.Tensor): Ground truth angles of shape (trials, max_items).
        presence (torch.Tensor): Binary mask of shape (trials, max_items) indicating item presence.
        train_err (bool, optional): If True, compute training error. Defaults to True.

    Returns:
        dict: Dictionary containing:
            - 'mean_train_error' (torch.Tensor): Mean training error.
            - 'variance_train_error' (torch.Tensor): Variance of training error.
            - 'mean_eval_error' (torch.Tensor): Mean evaluation error.
            - 'variance_eval_error' (torch.Tensor): Variance of evaluation error.
            - 'activation_penalty' (torch.Tensor): Activation penalty term.
    """
    # slice r_stack and reshape for decoding and recover its original shape
    step_threshold = int((T_init + T_stimi + T_delay) / dt)
    r_decode = r_stack[step_threshold:, :, :]
    num_steps, num_trials, num_neurons = r_decode.shape
    u_hat = decode(F, r_decode.reshape(-1, num_neurons)).reshape(num_steps, num_trials, -1)

    # Generate target outputs (ground truth for training loss)
    u_0 = generate_target(presence, target_thetas, stimuli_present=True)

    # Calculate training error if enabled
    if train_err:
        mean_train_error, var_train_error = calc_train_error(u_hat, u_0, presence)
    else:
        mean_train_error, var_train_error = torch.nan, torch.nan

    # Calculate evaluation error (angular)
    mean_eval_error, var_eval_error = calc_eval_error(u_hat, target_thetas, presence)

    # Calculate activation penalty (using all time steps)
    activ_penalty =  r_stack.abs().mean()

    return mean_train_error, var_train_error, mean_eval_error, var_eval_error, activ_penalty

def train(model, model_dir, history=None):
    # If no history is provided, initialize empty history
    if history is None:
        history = {
            "error_per_epoch": [],  # Overall mean error per epoch
            "error_std_per_epoch": [],  # std of overall error per epoch
            "activation_per_epoch": [],
            "group_errors": [[] for _ in item_num],  # List to store errors for each 'set size' group
            "group_std": [[] for _ in item_num],  # List to store std of errors for each group
            "group_activ": [[] for _ in item_num],  # List to store std of errors for each group
            "epochs": [],
            "lr": [],
        }
        start_epoch = 0
        start_lr = eta
    else:
        start_epoch = history['epochs'][-1]
        start_lr = history["lr"][-1]
        
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=adaptive_lr_patience)
    early_stopping = EarlyStopping(patience=early_stop_patience, verbose=False)

    # Initialize buffers to store recent history
    error_buffer = []
    error_std_buffer = []
    activation_buffer = []
    group_error_buffers = [[] for _ in item_num]  # Buffers for each group
    group_std_buffers = [[] for _ in item_num]
    group_activ_buffers = [[] for _ in item_num]

    # Split num_trials into len(num_item) groups
    trials_per_group = num_trials // len(item_num)  # Ensure equal split
    remaining_trials = num_trials % len(item_num)  # Handle leftover trials
    # Adjust trials count for each group (distribute leftovers)
    trial_counts = [trials_per_group + (1 if i < remaining_trials else 0) for i in range(len(item_num))]

    with tqdm(total=num_epochs, initial=start_epoch, desc="Training Progress", unit="epoch") as pbar_epoch:
        for epoch in range(start_epoch, num_epochs):
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
            if (epoch % 20 == 0) or (epoch==start_epoch):
                input_thetas = ((torch.rand(num_trials, max_item_num, device=device) * 2 * torch.pi) - torch.pi).requires_grad_()

            # Generate input tensor for all trials and time steps. (num_trials, steps, 2 * max_item_num)
            u_t = generate_input(
                presence=input_presence,
                theta=input_thetas,
                noise_level=ILC_noise,
                T_init=T_init,
                T_stimi=T_stimi,
                T_delay=T_delay,
                T_decode=T_decode,
                dt=dt,
            )
            
            r_output, _ = model(u_t, r0=None) # (trial, steps, neuron)
            r_output_T = r_output.transpose(0, 1)  # (steps, trial, neuron)

            # Calculate total loss
            mean_train_error, _, mean_eval_error, var_eval_error, activ_penalty = error_calc(model.F, r_output_T, input_thetas, input_presence, train_err=True)
            total_loss = lambda_err * mean_train_error + lambda_reg * activ_penalty

            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)
            earlystop_counter = early_stopping(total_loss.detach().cpu(), model)

            if model.positive_input:
                model.B.data = F.relu(model.B.data)  # Ensure B is non-negative

            # Append errors and activs to the history buffers
            error_buffer.append(mean_eval_error.item())
            error_std_buffer.append(var_eval_error.sqrt().item())
            activation_buffer.append((activ_penalty).item())

            start_index = 0
            for i, count in enumerate(trial_counts):
                end_index = start_index + count
                _, _, gp_mean_eval_error, gp_var_eval_error, gp_activ_penalty = error_calc(model.F, 
                                    r_output_T[:, start_index:end_index], 
                                    input_thetas[start_index:end_index], 
                                    input_presence[start_index:end_index], 
                                    train_err=False)

                group_error_buffers[i].append(gp_mean_eval_error.item())
                group_activ_buffers[i].append(gp_activ_penalty.item())
                group_std_buffers[i].append(gp_var_eval_error.sqrt().item())

                start_index = end_index

            if epoch % logging_period == 0:

                # Calculate averages for buffers and store in history
                history["error_per_epoch"].append(sum(error_buffer) / len(error_buffer))
                history["error_std_per_epoch"].append(sum(error_std_buffer) / len(error_std_buffer))
                history["activation_per_epoch"].append(sum(activation_buffer) / len(activation_buffer))
                history["epochs"].append(epoch)
                history["lr"].append(scheduler.get_last_lr()[0])

                for i in range(len(group_error_buffers)):
                    history["group_errors"][i].append(sum(group_error_buffers[i]) / len(group_error_buffers[i]))
                    history["group_std"][i].append(sum(group_std_buffers[i]) / len(group_std_buffers[i]))
                    history["group_activ"][i].append(sum(group_activ_buffers[i]) / len(group_activ_buffers[i]))

                # Clear all buffers
                error_buffer.clear()
                error_std_buffer.clear()
                activation_buffer.clear()
                for i in range(len(group_error_buffers)):
                    if len(group_error_buffers[i]) > logging_period:
                        group_error_buffers[i].clear()
                        group_std_buffers[i].clear()
                        group_activ_buffers[i].clear()

                # Update progress bar
                pbar_epoch.set_postfix({
                    "Error": f"{history['error_per_epoch'][-1]:.4f}rad",
                    "Activ": f"{history['activation_per_epoch'][-1]:.4f}Hz",
                    "lr": f"{scheduler.get_last_lr()}",
                    "PatienceCnt": earlystop_counter,
                })
                pbar_epoch.update(logging_period)

                # Save model and history every logging_period epochs
                os.makedirs(f'{model_dir}/models', exist_ok=True)
                save_model_and_history(model, history, 
                                    model_dir,
                                    model_name=f'model_epoch{epoch}.pth')

            # if fpf_bool and (fpf_period>0) and (epoch%fpf_period==0) and (total_loss>0.03):
            #     cloned_model = RNNMemoryModel(max_item_num, num_neurons, tau, dt, process_noise, device=device, positive_input=positive_input)
            #     cloned_model.load_state_dict(model.state_dict())  # Copy weights
            #     fixed_points_finder(cloned_model, epoch=epoch)

            if early_stopping.early_stop:
                print("Early stopping")
                break
    return history
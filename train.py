import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from early_stopping_pytorch.early_stopping import EarlyStopping

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
            "lr": eta,
        }
        start_epoch = 0
        start_lr = eta
    else:
        start_epoch = history['epochs'][-1]
        start_lr = history["lr"][0]
        
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

            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)
            earlystop_counter = early_stopping(total_loss.detach().cpu(), model)

            if model.positive_input >= 1:
                model.B.data = F.relu(model.B.data)  # Ensure B is non-negative

            # Append errors and activs to the history buffers
            error_buffer.append(total_error.item())
            error_std_buffer.append(total_error_var.sqrt().item())
            activation_buffer.append((total_activ_penal / lambda_reg).item())

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

                group_error_buffers[i].append(group_error.item())
                group_activ_buffers[i].append((group_activ_penal/lambda_reg).item())
                group_std_buffers[i].append(group_variance.sqrt().item())

                start_index = end_index

            if epoch % logging_period == 0:

                # Calculate averages for buffers and store in history
                history["error_per_epoch"].append(sum(error_buffer) / len(error_buffer))
                history["error_std_per_epoch"].append(sum(error_std_buffer) / len(error_std_buffer))
                history["activation_per_epoch"].append(sum(activation_buffer) / len(activation_buffer))
                history["epochs"].append(epoch)
                history["lr"] = scheduler.get_last_lr()

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
                    "Error": f"{history['error_per_epoch'][-1]:.4f}",
                    "Activ": f"{history['activation_per_epoch'][-1]:.4f}",
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
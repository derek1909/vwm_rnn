import torch
import torch.optim as optim
from tqdm import tqdm
from rnn import *
from config import *
from utils import save_model_and_history, generate_input


def memory_loss_integral(F, r_stack, u_0, presence, lambda_err=1.0, lambda_reg=0.1):
    # Vectorized memory loss over all time steps and trials
    # r_stack = (steps, trials, neurons)
    # u_0 = (trials, max_items*2)
    # presence = (trials, max_items)
    num_steps, num_trials, num_neurons = r_stack.shape
    u_hat_stack = decode(F, r_stack.reshape(-1, num_neurons)).reshape(num_steps, num_trials, -1)

    error = (u_0 - u_hat_stack * presence.repeat_interleave(2, dim=1)).pow(2) / torch.sum(presence,dim=1).unsqueeze(0).unsqueeze(-1) #average over num of presented item (might be different for each trial)
    error = lambda_err * error.sum() / (num_steps * num_trials) # average over trials and steps
    
    activation_penalty = lambda_reg * r_stack.abs().mean()
    total_loss = error + activation_penalty
    return total_loss, activation_penalty

def train(model, model_dir, history=None):
    optimizer = optim.Adam(model.parameters(), lr=eta)

    # If no history is provided, initialize empty history
    if history is None:
        history = {"error_per_epoch": [], "activation_penalty_per_epoch": []}

    # Generate input_thetas
    # input_thetas = ((torch.rand(num_trials, max_item_num) * 2 * torch.pi) - torch.pi).requires_grad_()
    input_thetas = torch.linspace(-torch.pi, torch.pi, num_trials).unsqueeze(1).repeat(1, max_item_num).requires_grad_()

    # Split num_trials into len(num_item) groups
    trials_per_group = num_trials // len(item_num)  # Ensure equal split
    remaining_trials = num_trials % len(item_num)  # Handle leftover trials
    # Adjust trials count for each group (distribute leftovers)
    trial_counts = [trials_per_group + (1 if i < remaining_trials else 0) for i in range(len(item_num))]

    with tqdm(total=num_epochs, desc="Training Progress", unit="epoch") as pbar_epoch:
        trial_counter = 0  # Initialize a counter for trials

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            total_loss = 0
            total_activ_penal = 0

            # Generate presence for each group
            input_presence = torch.zeros(num_trials, max_item_num, requires_grad=True)
            start_index = 0
            for i, count in enumerate(trial_counts):
                end_index = start_index + count
                one_hot_indices = torch.stack([torch.randperm(max_item_num)[:item_num[i]] for _ in range(count)])
                input_presence_temp = input_presence.clone()  # Create a detached clone
                input_presence_temp[start_index:end_index] = input_presence_temp[start_index:end_index].scatter(1, one_hot_indices, 1)
                input_presence = input_presence_temp
                start_index = end_index

            # input_thetas = ((torch.rand(num_trials, max_item_num) * 2 * torch.pi) - torch.pi).requires_grad_()

            # Initialize hidden states and collect activations for each time step
            r = torch.zeros(num_trials, num_neurons)
            r_list = []

            # Simulate the RNN across all trials and time steps
            for step in range(simul_steps):
                time = step * dt
                u_t = generate_input(
                    input_presence,
                    input_thetas,
                    noise_level=encode_noise,
                    stimuli_present=(T_init < time < T_stimi + T_init)
                )
                r = model(r, u_t)
                if time > (T_stimi + T_delay + T_init):
                    r_list.append(r.clone())

            # Calculate loss over all trials and time steps
            r_stack = torch.stack(r_list)
            u_0 = generate_input(input_presence, input_thetas, stimuli_present=True)  # u_0 has no noise
            total_loss, total_activ_penal = memory_loss_integral(
                model.F, r_stack, u_0, input_presence,
                lambda_err=lambda_err, lambda_reg=lambda_reg
            )
            total_loss.backward()

            # Update model parameters
            optimizer.step()

            # Track errors and penalties
            history["error_per_epoch"].append((total_loss - total_activ_penal).item())
            history["activation_penalty_per_epoch"].append((total_activ_penal / lambda_reg).item())

            # Update progress bar
            pbar_epoch.set_postfix({
                "Error": f"{history['error_per_epoch'][-1]:.4f}",
                "Activation": f"{history['activation_penalty_per_epoch'][-1]:.4f}"
            })
            pbar_epoch.update(1)

            # Increment trial counter
            trial_counter += num_trials

            # Save model and history every 50 trials
            if trial_counter >= 50:
                save_model_and_history(model, history, model_dir)
                trial_counter = 0  # Reset trial counter

    return history
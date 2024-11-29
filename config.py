# Model parameters
max_item_num = 8
item_num = [1,3,5,7]
num_neurons = 128
tau = 50
dt = 10
encode_noise = 0.01 # rad
process_noise = 0.5 # Hz
decode_noise = 0.0
T_init = 0
T_stimi = 400
T_delay = 0
T_decode = 800
T_simul = T_init + T_stimi + T_delay + T_decode
simul_steps = int(T_simul/dt)

# Training parameters
train_rnn = True  # Set to True if training is required
train_from_scratch = False
num_epochs = int(1)
eta = 1e-5 # learning_rate
lambda_reg = 5e-4  # coeff for activity penalty
lambda_err = 1.0  # coeff for error penalty
num_trials = 128  # Number of trials per epoch

# Model and logging parameters
# rnn_name = "fixed_discrete_input-no_noise"
rnn_name = "rnn4present_128neuron_newactiv"
model_dir = f"rnns/{rnn_name}"



# Save config
import json
import os

# Define the configuration dictionary
config = {
    "model_params": {
        "max_item_num": max_item_num,
        "num_neurons": num_neurons,
        "tau": tau,
        "dt": dt,
        "encode_noise": encode_noise,
        "process_noise": process_noise,
        "decode_noise": decode_noise,
        "T_init": T_init,
        "T_stimi": T_stimi,
        "T_delay": T_delay,
        "T_decode": T_decode,
        "T_simul": T_simul,
        "simul_steps": simul_steps,
        "item_num": item_num,
    },
    "training_params": {
        "train_rnn": train_rnn,
        "train_from_scratch": train_from_scratch,
        "num_epochs": num_epochs,
        "eta": eta,
        "lambda_reg": lambda_reg,
        "lambda_err": lambda_err,
        "num_trials": num_trials,
    },
    "model_and_logging_params": {
        "rnn_name": rnn_name,
        "model_dir": model_dir,
    }
}

# Save the configuration to a JSON file
os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist
config_path = os.path.join(model_dir, "config.json")

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

print(f"Configuration saved to {config_path}")
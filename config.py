import torch
import yaml
import json
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description="Specify YAML configuration file path")
parser.add_argument("--config", type=str, default="./config.yaml", help="Path to YAML configuration file")
args = parser.parse_args()

config_path = args.config

# Load config file
if config_path.endswith(".yaml") or config_path.endswith(".yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
elif config_path.endswith(".json"):
    with open(config_path, "r") as f:
        config = json.load(f)
else:
    raise ValueError("Unsupported config file format. Use YAML or JSON.")

# Extract configurations
model_params = config["model_params"]
training_params = config["training_params"]
logging_params = config["model_and_logging_params"]
fpf_params = config["fpf_params"]

# Model parameters
max_item_num = model_params["max_item_num"]
item_num = model_params["item_num"]
num_neurons = model_params["num_neurons"]
dt = model_params["dt"]
ILC_noise = model_params["ILC_noise"] # rad
process_noise = model_params["process_noise"] # Hz
decode_noise = model_params["decode_noise"]
positive_input = model_params["positive_input"] # positive input. 0 if no need to be positive.
T_init = model_params["T_init"]
T_stimi = model_params["T_stimi"]
T_delay = model_params["T_delay"]
T_decode = model_params["T_decode"]
T_simul = T_init + T_stimi + T_delay + T_decode
simul_steps = int(T_simul/dt)

# Training parameters
train_rnn = training_params["train_rnn"]  # Set to True if training is required
from_scratch = training_params["from_scratch"]
num_epochs = int(training_params["num_epochs"]) 
eta = training_params["eta"] # learning_rate
lambda_reg = training_params["lambda_reg"]  # coeff for activity penalty
lambda_err = training_params["lambda_err"]  # coeff for error penalty
num_trials = training_params["num_trials"]  # Number of trials per epoch
logging_period = training_params["logging_period"]  # record progress every 10 epoch
early_stop_patience = training_params["early_stop_patience"]
adaptive_lr_patience = training_params["adaptive_lr_patience"]

# Model and logging parameters
rnn_name = logging_params["rnn_name"]
rnn_name = f'{rnn_name}_n{num_neurons}item{max_item_num}PI{int(positive_input)}'
model_dir = f"rnns/{rnn_name}"
cuda_device = int(logging_params["cuda_device"])
error_dist_bool = logging_params["error_dist_bool"]

# Fixed Point Finder parameters
fpf_bool = fpf_params["fpf_bool"]
fpf_pca_bool = fpf_params["fpf_pca_bool"]
fpf_names = fpf_params["fpf_names"] # stimuli or decode.
fpf_N_init = fpf_params["fpf_N_init"] # Number of initial states for optimization
fpf_trials = fpf_params["fpf_trials"]  # Number of trials per epoch
fpf_noise_scale = fpf_params["fpf_noise_scale"] # Standard deviation of noise added to states
fpf_hps = fpf_params["fpf_hps"]  # Hyperparameters for fixed point finder

os.makedirs(model_dir, exist_ok=True)
destination_path = os.path.join(model_dir, os.path.basename(config_path))
if os.path.realpath(config_path) != os.path.realpath(destination_path):
    shutil.copyfile(config_path, destination_path)


if torch.cuda.is_available():
    torch.cuda.set_device(cuda_device)
    device = f'cuda:{cuda_device}'
else:
    device = 'cpu'  # Fallback to CPU if CUDA is not available

print(f"rnn_name: {rnn_name}")
print(f"using device: {device}")

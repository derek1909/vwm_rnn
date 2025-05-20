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
tau_max = model_params["tau_max"]
tau_min = model_params["tau_min"]
ILC_noise = model_params["ILC_noise"] # rad
spike_noise_factor = model_params["spike_noise_factor"] # [0,1]. k = 1/sqrt(M) where M is  #neurons in a single artificial neuron.
spike_noise_type = model_params["spike_noise_type"] # "gamma" or "normal"
positive_input = model_params["positive_input"] # positive input. 0 if no need to be positive.
input_strength = model_params["input_strength"] # mean value of Bu.
saturation_firing_rate = model_params["saturation_firing_rate"] # mean value of Bu.
dales_law = model_params["dales_law"]
T_init = model_params["T_init"]
T_stimi = model_params["T_stimi"]
T_delay = model_params["T_delay"]
T_decode = model_params["T_decode"]
T_simul = T_init + T_stimi + T_delay + T_decode
simul_steps = int(T_simul/dt)

# Training parameters
train_rnn = training_params["train_rnn"]  # Set to True if training is required
load_history = training_params["load_history"]
use_scripted_model = training_params["use_scripted_model"] # Cannot be used anymore because gamma does not work with jit_script
num_iterations = int(training_params["num_iterations"]) 
eta = training_params["eta"] # learning_rate
lambda_reg = training_params["lambda_reg"]  # coeff for activity penalty
lambda_err = training_params["lambda_err"]  # coeff for error penalty
num_trials = training_params["num_trials"]  # Number of trials per iteration
logging_period = training_params["logging_period"]  # record progress every 10 iteration
early_stop_patience = training_params["early_stop_patience"]
adaptive_lr_patience = training_params["adaptive_lr_patience"]

# Model and logging parameters
rnn_name = logging_params["rnn_name"]
rnn_name = f'{rnn_name}_n{num_neurons}item{max_item_num}PI{int(positive_input)}{spike_noise_type.lower()[:5]}{spike_noise_factor}'
model_dir = f"rnn_models/{rnn_name}"
cuda_device = int(logging_params["cuda_device"])
plot_weights_bool = logging_params["plot_weights_bool"]
error_dist_bool = logging_params["error_dist_bool"]
fit_mixture_bool = logging_params["fit_mixture_bool"]
SNR_analy_bool = logging_params["SNR_analy_bool"]

# Fixed Point Finder parameters
fpf_bool = fpf_params["fpf_bool"]
fpf_pca_bool = fpf_params["fpf_pca_bool"]
fpf_names = fpf_params["fpf_names"] # stimuli or decode.
fpf_N_init = fpf_params["fpf_N_init"] # Number of initial states for optimization
fpf_trials = fpf_params["fpf_trials"]  # Number of trials per iteration
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
print("Model directory:", model_dir)
print(f"using device: {device}")

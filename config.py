# Model parameters
max_item_num = 1
num_neurons = 64
tau = 50
dt = 5
encode_noise = 0.001
process_noise = 0.001
decode_noise = 0.0
T_init = 100
T_stimi = 400
T_delay = 200
T_decode = 800
T_simul = T_init + T_stimi + T_delay + T_decode
simul_steps = int(T_simul/dt)
item_num = 1

# Training parameters
train_rnn = True  # Set to True if training is required
train_from_scratch = False
num_epochs = 500
eta = 2e-4  # learning_rate
lambda_reg = 2e-5  # coeff for activity penalty
lambda_err = 1.0  # coeff for error penalty
num_trials = 64  # Number of trials per epoch



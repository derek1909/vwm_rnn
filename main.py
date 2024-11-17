import torch

from rnn import *
from config import *
from train import *
from utils import *


if __name__ == "__main__":
    model = RNNMemoryModel(max_item_num, num_neurons, tau, dt, process_noise)
    if train_rnn:
        if not train_from_scratch:
            model.load_state_dict(torch.load('models/model_weights.pth'))
        error_per_epoch, activation_penalty_per_epoch = train(model)
        plot_training_curves(error_per_epoch, activation_penalty_per_epoch)
    else:
        model.load_state_dict(torch.load('models/model_weights.pth'))
    
    angle_targets = [-1.9, -0.1, 0.6, 1.8]
    decoded_orientations = evaluate(model, angle_targets)
    plot_results(decoded_orientations)
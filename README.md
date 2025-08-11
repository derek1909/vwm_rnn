# Visual Working Memory RNN (vwm_rnn)

A biologically plausible recurrent neural network (RNN) for simulating human visual working memory (VWM).  
This project was developed as my MEng dissertation at the University of Cambridge (2024–2025).

## Overview

This codebase implements a biologically-inspired RNN model that simulates visual working memory tasks. The model incorporates several key biological constraints including Dale's law, temporal dynamics, and various noise sources to study how neural circuits maintain and manipulate visual information over time.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Analysis Tools](#analysis-tools)

## Installation

### Prerequisites
- CUDA-capable GPU (optional but recommended)

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/derek1909/vwm_rnn.git
cd vwm_rnn
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate rnn
```

3. Verify installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Project Structure

```
vwm_rnn/
├── config.py              # Configuration parser and parameter definitions
├── config.yaml            # Default configuration file
├── main.py                # Main execution script
├── rnn.py                 # Core RNN model implementation
├── train.py               # Training loop and evaluation functions
├── utils.py               # Utility functions for data generation and visualization
├── run_exp.py             # Experiment runner script
├── environment.yml        # Conda environment specification
├── early_stopping_pytorch/  # Early stopping utilities
├── analysis/              # Analysis modules
│   ├── __init__.py
│   ├── fixedpoint.py      # Fixed point analysis
│   ├── fixedpoint_utils.py
│   ├── dn_analysis.py     # Divisive normalization analysis
│   ├── error_dist_analysis.py  # Error distribution analysis
│   ├── mixed_selectivity.py    # Mixed selectivity analysis
│   ├── snr_analysis.py    # Signal-to-noise ratio analysis
│   └── FixedPoints/       # Fixed point finder packages
├── final_reports/         # Generated analysis plots and reports
├── other/                 # Additional analysis and utilities
└── rnn_models/           # Saved model checkpoints and analysis results
```

## Quick Start

### Basic Training and Evaluation
```bash
# Train with custom configuration stored in ./config.yaml
python main.py

# Train with configuration of an existing model
python main.py --config path_of_model/config.yaml
```

## Configuration

The model behavior is controlled through YAML configuration files with four main sections. The configuration system supports both YAML and JSON formats.

### Configuration Structure

#### Model Parameters (`model_params`)
Core neural network architecture and dynamics:

- **`max_item_num`** (int): Maximum number of items in working memory (default: 10)
- **`item_num`** (list): List of set sizes during training and testing [1,2,3,...,10]
- **`num_neurons`** (int): Number of neurons in the rnn (default: 256)
- **`dt`** (float): Integration time step in milliseconds (default: 10)
- **`tau_min/tau_max`** (float): Neural time constant bounds in ms for log-uniform sampling (default: 50/300)
- **`spike_noise_factor`** (float): Amplitude of spike noise [0,1] (default: 0.05)
- **`spike_noise_type`** (str): Noise distribution - 'gamma', 'gauss', 'puregauss', 'csnr' (default: 'gamma')
- **`positive_input`** (bool): Constrain input weights to be positive (default: true)
- **`input_strength`** (float): Mean magnitude of input Bu (default: 120)
- **`saturation_firing_rate`** (float): Maximum firing rate in Hz (default: 60)
- **`dales_law`** (bool): Enforce Dale's law - separate E/I populations (default: true)
- **`ILC_noise`** (float): Item location noise in radians (default: 0.01)

**Timing Parameters:**
- **`T_init`** (ms): Initial baseline period (default: 20)
- **`T_stimi`** (ms): Stimulus presentation duration (default: 500)  
- **`T_delay`** (ms): Memory delay period (default: 1000)
- **`T_decode`** (ms): Response/readout period (default: 500)

#### Training Parameters (`training_params`)
Optimization and learning configuration:

- **`train_rnn`** (bool): Whether to train the model (otherwise just load a trained model) (default: true)
- **`load_history`** (bool): Load previous trained model and training history (default: false)
- **`use_scripted_model`** (bool): Use TorchScript compilation. Conflict with gamma noise. (default: false)
- **`num_iterations`** (int): Maximum training iterations (default: 50000)
- **`error_def`** (str): Loss function type - 'l2', 'sqrtl2', 'rad', 'exp' (default: 'rad')
- **`eta`** (float): Learning rate for Adam optimizer (default: 0.0001)
- **`lambda_reg`** (float): L1 regularization coefficient (default: 0.00001)
- **`lambda_err`** (float): Error penalty coefficient (default: 1.0)
- **`num_trials`** (int): Batch size - trials per training iteration (default: 512)
- **`logging_period`** (int): Save progress every N iterations (default: 50)
- **`early_stop_patience`** (int): Early stopping patience (default: 600)
- **`adaptive_lr_patience`** (int): Learning rate reduction patience (default: 400)

#### Model and Logging Parameters (`model_and_logging_params`)
Output and analysis configuration:

- **`rnn_name`** (str): Base name for saved models (default: "EasyTask1")
- **`cuda_device`** (int): GPU device ID (default: 0)
- **`plot_weights_bool`** (bool): Generate weight visualizations (default: false)
- **`error_dist_bool`** (bool): Run error distribution analysis (default: false)
- **`fit_mixture_bool`** (bool): Fit mixture models to errors (default: false)
- **`snr_analy_bool`** (bool): Run signal-to-noise ratio analysis (default: false)
- **`mixed_selec_bool`** (bool): Run mixed selectivity analysis (default: false)

#### Fixed Point Finder Parameters (`fpf_params`)
Neural fixed point analysis configuration:

- **`fpf_bool`** (bool): Enable fixed point analysis (default: false)
- **`fpf_pca_bool`** (bool): Apply an extra 2D PCA after fixed point finding (default: true)
- **`fpf_names`** (list): Analysis phases ['fpf_NoFps', 'fpf_decode']
- **`fpf_trials`** (int): Number of trials for analysis (default: 20)
- **`fpf_N_init`** (int): Number of random initializations (default: 20)
- **`fpf_noise_scale`** (float): Noise added to initial states (default: 0.0)

**Fixed Point Finder Hyperparameters (`fpf_hps`):**
- **`max_iters`** (int): Maximum optimization iterations (default: 50)
- **`lr_init`** (float): Initial learning rate (default: 0.01)
- **`outlier_distance_scale`** (float): Outlier detection threshold (default: 10.0)
- **`verbose`** (bool): Print optimization progress (default: false)
- **`super_verbose`** (bool): Detailed debugging output (default: false)

### Automatic Model Naming
Models are automatically named using the pattern:
```
{rnn_name}_n{num_neurons}item{max_item_num}PI{positive_input}{spike_noise_type}{spike_noise_factor}{error_def}
```
Example: `EasyTask1_n256item10PI1gamma0.05rad`

## Analysis Tools

### Fixed Point Analysis
Identifies and analyzes neural fixed points to understand attractor dynamics:
- Location and stability of fixed points
- Basin of attraction analysis
- Relationship to behavioral performance

### Signal-to-Noise Ratio Analysis
Quantifies neural signal quality across different memory loads:
- Population-level SNR
- Relationship to memory capacity

### Error Distribution Analysis
Analyzes behavioral error patterns:
- Angular error distributions
- Set size effects

### Divisive Normalization Analysis
Studies normalization mechanisms in the neural population:
- Response gain modulation
- Population normalization strength
- Effects on memory precision

### Mixed Selectivity Analysis
Examines neural tuning properties:
- Orientation selectivity
- Mixed selectivity across stimuli
- Population coding properties

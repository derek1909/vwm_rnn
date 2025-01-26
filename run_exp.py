import json
import subprocess

# Define paths
json_file_path = "config.json"  # Update this if your file is elsewhere
main_script = "python main.py"  # Update this if you need a specific Python version

# List of neuron counts to iterate over
# num_neurons_values = [64, 128, 256, 512, 1024, 2048, 4096]
ilc_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1]

for ilc in ilc_values:
    # Load the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Modify the number of neurons
    data["model_params"]["ILC_noise"] = ilc
    data["model_and_logging_params"]["rnn_name"] = f'TMR-ilc-{ilc}'

    # Save the modified JSON
    with open(json_file_path, "w") as f:
        json.dump(data, f, indent=4)

    # Run main.py
    print(f"Running main.py with ILC_noise = {ilc}")
    subprocess.run(main_script, shell=True)

print("All runs completed!")
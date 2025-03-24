import os
import re
import imageio.v3 as iio

# Define the folder containing the PNG files
folder_path = "/homes/jd976/working/vwm_rnn/rnn_models/256n_1item_PI_theta4/fpf/decode/angle_45"

# Define the output GIF file name
output_gif_path = f"{folder_path}.gif"

# Get all PNG files in the folder
png_files = [file for file in os.listdir(folder_path) if file.endswith(".png")]

# Sort files by extracting the 'iteration_{iteration}' number
def extract_iteration(filename):
    match = re.search(r'iteration_(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')  # If no iteration number is found, place it at the end

png_files_sorted = sorted(png_files, key=extract_iteration)

# Read and collect images
images = []
for png_file in png_files_sorted:
    image_path = os.path.join(folder_path, png_file)
    images.append(iio.imread(image_path))

# Save the images as a GIF
iio.imwrite(output_gif_path, images, duration=100.0)

print(f"GIF created and saved to {output_gif_path}")
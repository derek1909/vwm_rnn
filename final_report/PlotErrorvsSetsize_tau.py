# Regenerate the plot, ensuring the legend only appears once (outside the axes)

import matplotlib.pyplot as plt
import os

# Data for each tau setup
set_sizes = list(range(1, 11))

data = {
    "Large τ": {
        "errors": [0.10033170819282532, 0.12378951616585254, 0.13125103645026684, 0.14399046331644058,
                   0.15200463011860849, 0.16305390283465385, 0.17621355935931204, 0.1881816044449806,
                   0.20491659224033357, 0.22299815520644187],
        "activs": [1.6835503160953522, 2.7744433426856996, 3.3682587218284605, 3.945373513698578,
                   4.510033836364746, 5.067752952575684, 5.625731792449951, 6.173509202003479,
                   6.713487486839295, 7.234781112670898]
    },
    "Medium τ": {
        "errors": [0.16944078512489796, 0.28378236562013626, 0.38369446009397506, 0.4670336028933525,
                   0.5475326019525528, 0.6109324884414673, 0.669268890619278, 0.7224836361408233,
                   0.7691057831048965, 0.8179628193378449],
        "activs": [7.074934206008911, 9.073729057312011, 9.85138524055481, 10.380134696960448,
                   10.811751184463501, 11.234747095108032, 11.690884408950806, 12.136845874786378,
                   12.638689527511596, 13.196494865417481]
    },
    "Small τ": {
        "errors": [1.5841513311862945, 1.5807469081878662, 1.580977302789688, 1.5736156058311463,
                   1.5666421055793762, 1.5789333379268646, 1.5664971244335175, 1.565912903547287,
                   1.5731834328174592, 1.5695109069347382],
        "activs": [0.3688726207613945, 0.9982017290592193, 1.5720760369300841, 2.151965718269348,
                   2.769003176689148, 3.3755135631561277, 3.9945758104324343, 4.620416855812072,
                   5.2830138635635375, 5.996839570999145]
    }
}

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharex=True)

lines = []

# Left plot: Error vs Set Size
for label, values in data.items():
    line, = axes[0].plot(set_sizes, values["errors"], label=label, linewidth=2)
    lines.append(line)
axes[0].set_title("Error vs Set Size")
axes[0].set_xlabel("Set Size")
axes[0].set_ylabel("Error (rad)")
axes[0].grid(True)

# Right plot: Activation vs Set Size
for _, values in data.items():
    axes[1].plot(set_sizes, values["activs"], linewidth=2)
axes[1].set_title("Activation vs Set Size")
axes[1].set_xlabel("Set Size")
axes[1].set_ylabel("Mean Activation (Hz)")
axes[1].grid(True)

# Shared legend outside both plots
fig.legend(lines, [line.get_label() for line in lines], loc='lower center', ncol=3, frameon=False)
plt.tight_layout(rect=[0, 0.1, 1, 1])

# Save
output_folder = "./final_reports/"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "error_activation_vs_setsize_tau.png")
plt.savefig(output_path, dpi=200)
plt.close()

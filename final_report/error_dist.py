import yaml
import matplotlib.pyplot as plt
import os

# Paths
human_yaml_path = "/homes/jd976/working/vwm_rnn/analysis/behavior_summary.yaml"
model_yaml_paths = [
    "/homes/jd976/working/vwm_rnn/final_reports/OptimalModel_check_n64item10PI1gamma0.2l2/error_dist/fit_summary.yaml",
    "/homes/jd976/working/vwm_rnn/final_reports/spike_noise_factor-0.30000000_n256item10PI1gamma0.3rad/error_dist/fit_summary.yaml"
]
output_dir = "/homes/jd976/working/vwm_rnn/final_reports"
var_png = os.path.join(output_dir, "compare_variance.png")
kurt_png = os.path.join(output_dir, "compare_kurtosis.png")

# Load human data
with open(human_yaml_path, "r") as f:
    human_data = yaml.safe_load(f)

human_item_num = sorted(human_data.keys(), key=lambda x: int(x))
human_variances = [human_data[int(k)]["circular_variance_mean"] for k in human_item_num]
human_kurtoses = [human_data[int(k)]["circular_kurtosis_mean"] for k in human_item_num]
human_var_se = [human_data[int(k)]["circular_variance_se"] for k in human_item_num]
human_kurt_se = [human_data[int(k)]["circular_kurtosis_se"] for k in human_item_num]
item_num = [int(k) for k in human_item_num]

# Load model data
model_summaries = []
for path in model_yaml_paths:
    with open(path, "r") as f:
        model_summaries.append(yaml.safe_load(f))

model_variances_list = [
    [summary[int(k)]["circular_variance"] for k in item_num]
    for summary in model_summaries
]

model_kurtoses_list = [
    [summary[int(k)]["circular_kurtosis"] for k in item_num]
    for summary in model_summaries
]

# Labels for models
model_labels = ["Euclidean", "Angular"]
model_colors = ["r", "b"]

# Plot: Circular Variance
fig1, ax1 = plt.subplots(figsize=(4, 3))
ax1.errorbar(item_num, human_variances, yerr=human_var_se, fmt='ko', capsize=3, label="Human")
for variances, label, color in zip(model_variances_list, model_labels, model_colors):
    ax1.plot(item_num, variances, color=color, label=label)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xticks(item_num)
ax1.set_xticklabels([str(s) for s in item_num])
ax1.set_ylabel('circular variance ($\\sigma^2$)')
ax1.set_xlabel('items')
ax1.set_title('width, $\\omega$')
ax1.legend()
fig1.tight_layout()
fig1.savefig(var_png, dpi=300)

# Plot: Circular Kurtosis
fig2, ax2 = plt.subplots(figsize=(4, 3))
ax2.errorbar(item_num, human_kurtoses, yerr=human_kurt_se, fmt='ko', capsize=3, label="Human")
for kurtoses, label, color in zip(model_kurtoses_list, model_labels, model_colors):
    ax2.plot(item_num, kurtoses, color=color, label=label)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xticks(item_num)
ax2.set_xticklabels([str(s) for s in item_num])
ax2.set_ylabel('circular kurtosis')
ax2.set_xlabel('items')
ax2.set_title('kurtosis')
ax2.legend()
fig2.tight_layout()
fig2.savefig(kurt_png, dpi=300)
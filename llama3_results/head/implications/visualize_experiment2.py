"""
Visualization for Experiment 2: Ablation Study Results
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open("experiment2_ablation_results.json", "r") as f:
    results = json.load(f)

# Extract data
configs = []
accuracies = []
drops = []

baseline_acc = results["conditions"]["baseline"]["accuracy"]

for config_name, data in results["conditions"].items():
    configs.append(config_name)
    accuracies.append(data["accuracy"])
    drops.append(baseline_acc - data["accuracy"])

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Experiment 2: Ablation Study Results', fontsize=16, fontweight='bold')

# Color scheme
colors = []
for config in configs:
    if config == "baseline":
        colors.append('#2563eb')  # Blue for baseline
    elif config == "early_L2":
        colors.append('#dc2626')  # Red for most impactful
    elif "copy" in config or config.startswith("L"):
        colors.append('#9ca3af')  # Gray for copy heads
    else:
        colors.append('#6b7280')

# Plot 1: Accuracy by configuration
ax1 = axes[0]
bars1 = ax1.bar(range(len(configs)), accuracies, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_xticks(range(len(configs)))
ax1.set_xticklabels(configs, rotation=45, ha='right', fontsize=10)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Accuracy After Ablation', fontsize=13)
ax1.axhline(y=baseline_acc, color='#2563eb', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.1f}%)')
ax1.legend(loc='upper right')
ax1.set_ylim(0, 70)

# Add value labels
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    ax1.annotate(f'{acc:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: Accuracy drop from baseline
ax2 = axes[1]
drop_colors = ['#dc2626' if d > 10 else '#f59e0b' if d > 0 else '#10b981' for d in drops]
bars2 = ax2.bar(range(len(configs)), drops, color=drop_colors, edgecolor='black', linewidth=0.5)
ax2.set_xticks(range(len(configs)))
ax2.set_xticklabels(configs, rotation=45, ha='right', fontsize=10)
ax2.set_ylabel('Accuracy Drop (%)', fontsize=12)
ax2.set_title('Impact of Ablation (Drop from Baseline)', fontsize=13)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add value labels
for bar, drop in zip(bars2, drops):
    height = bar.get_height()
    if height != 0:
        ax2.annotate(f'{drop:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top', 
                    fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('experiment2_ablation_visual.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Saved: experiment2_ablation_visual.png")

# Create a second figure highlighting the key finding
fig2, ax = plt.subplots(figsize=(10, 6))

# Simplified comparison: Copy heads vs Early heads
comparison_labels = ['Baseline', 'Top 3 Copy Heads\n(L17H24, L20H14, L24H27)', 'Early L2 Heads\n(L2H21, L2H22, L2H23)']
comparison_acc = [60.0, 56.7, 26.7]
comparison_colors = ['#2563eb', '#9ca3af', '#dc2626']

bars = ax.bar(comparison_labels, comparison_acc, color=comparison_colors, edgecolor='black', linewidth=1.5, width=0.6)

ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title('Key Finding: Early Layers are More Critical than "Copy Heads"', fontsize=14, fontweight='bold')
ax.set_ylim(0, 75)

# Add value labels and drop annotations
for i, (bar, acc) in enumerate(zip(bars, comparison_acc)):
    height = bar.get_height()
    ax.annotate(f'{acc:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    if i > 0:
        drop = 60.0 - acc
        ax.annotate(f'↓ {drop:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height - 5),
                    ha='center', va='top', fontsize=11, color='white', fontweight='bold')

# Add text box with finding
textstr = "Disabling early L2 heads causes 10x\nmore accuracy loss than copy heads!"
props = dict(boxstyle='round', facecolor='#fef3c7', edgecolor='#f59e0b', linewidth=2)
ax.text(0.5, 0.15, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='bottom', horizontalalignment='center', bbox=props)

plt.tight_layout()
plt.savefig('experiment2_key_finding.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Saved: experiment2_key_finding.png")


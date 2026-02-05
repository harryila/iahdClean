#!/usr/bin/env python3
"""
Phase 4: Figure 8 Replication - Ablation Curves

Generates three versions of Figure 8:
- Option A: Per-question (comparing methods)
- Option B: Per-method (comparing questions)  
- Option C: Grid layout (comprehensive view)

Each shows accuracy vs number of ablated heads, with top heads vs random baseline.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
PHASE3_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "phase3")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

METHODS = ["summed_attention", "retrieval_head_wu24", "qrhead"]
METHOD_LABELS = {
    "summed_attention": "Summed Attention",
    "retrieval_head_wu24": "Wu24 Retrieval Head",
    "qrhead": "QRHead"
}
METHOD_COLORS = {
    "summed_attention": "#1f77b4",  # blue
    "retrieval_head_wu24": "#ff7f0e",  # orange
    "qrhead": "#2ca02c"  # green
}

MODELS = ["llama3_instruct", "llama3_base"]
QUESTIONS = ["inc_state", "inc_year", "employee_count", "hq_state"]
QUESTION_LABELS = {
    "inc_state": "Inc. State",
    "inc_year": "Inc. Year",
    "employee_count": "Employee Count",
    "hq_state": "HQ State"
}
TOKEN_LENGTHS = [2048, 4096, 6144, 8192]
ABLATION_LEVELS = [5, 10, 20, 30, 40, 50]


def load_all_results():
    """Load all Phase 3 results into a structured dictionary."""
    results = {}
    
    for method in METHODS:
        results[method] = {}
        method_dir = os.path.join(PHASE3_DIR, method, "results")
        
        for model in MODELS:
            results[method][model] = {}
            
            for question in QUESTIONS:
                results[method][model][question] = {}
                
                for tokens in TOKEN_LENGTHS:
                    filepath = os.path.join(
                        method_dir, model, question, f"tokens_{tokens}.json"
                    )
                    
                    if os.path.exists(filepath):
                        with open(filepath, "r") as f:
                            results[method][model][question][tokens] = json.load(f)
                    else:
                        print(f"Warning: Missing {filepath}")
                        results[method][model][question][tokens] = None
    
    return results


def extract_ablation_curve(result):
    """Extract accuracy values at each ablation level from a result."""
    if result is None:
        return None, None, None
    
    baseline_acc = result["baseline"]["accuracy"]
    
    top_accs = {}
    random_accs = {}
    
    for abl in result.get("top_heads_ablations", []):
        top_accs[abl["num_heads"]] = abl["accuracy"]
    
    for abl in result.get("random_heads_ablations", []):
        random_accs[abl["num_heads"]] = abl["accuracy"]
    
    return baseline_acc, top_accs, random_accs


def aggregate_curves(results, group_by):
    """
    Aggregate curves across specified dimensions.
    Returns mean and std for top and random at each ablation level.
    """
    aggregated = defaultdict(lambda: defaultdict(list))
    
    for method in METHODS:
        for model in MODELS:
            for question in QUESTIONS:
                for tokens in TOKEN_LENGTHS:
                    result = results[method][model][question][tokens]
                    baseline, top_accs, random_accs = extract_ablation_curve(result)
                    
                    if baseline is None:
                        continue
                    
                    # Determine group key based on grouping strategy
                    if group_by == "method":
                        key = method
                    elif group_by == "question":
                        key = question
                    elif group_by == "method_question":
                        key = (method, question)
                    else:
                        key = "all"
                    
                    for level in ABLATION_LEVELS:
                        if level in top_accs:
                            aggregated[key][("top", level)].append(top_accs[level])
                        if level in random_accs:
                            aggregated[key][("random", level)].append(random_accs[level])
                        aggregated[key][("baseline", 0)].append(baseline)
    
    # Compute mean and std
    stats = {}
    for key, data in aggregated.items():
        stats[key] = {}
        for (curve_type, level), values in data.items():
            if values:
                stats[key][(curve_type, level)] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "n": len(values)
                }
    
    return stats


def plot_option_a(results):
    """Option A: Per-question figures comparing methods."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, question in enumerate(QUESTIONS):
        ax = axes[idx]
        
        for method in METHODS:
            # Collect data for this method-question combination
            top_means = []
            top_stds = []
            random_means = []
            random_stds = []
            baseline_vals = []
            
            for model in MODELS:
                for tokens in TOKEN_LENGTHS:
                    result = results[method][model][question][tokens]
                    baseline, top_accs, random_accs = extract_ablation_curve(result)
                    
                    if baseline is None:
                        continue
                    
                    baseline_vals.append(baseline)
                    
                    for level in ABLATION_LEVELS:
                        if level in top_accs:
                            if len(top_means) < len(ABLATION_LEVELS):
                                top_means.append([])
                                top_stds.append([])
                            idx_level = ABLATION_LEVELS.index(level)
                            if idx_level < len(top_means):
                                top_means[idx_level].append(top_accs[level])
                        
                        if level in random_accs:
                            if len(random_means) < len(ABLATION_LEVELS):
                                random_means.append([])
                            idx_level = ABLATION_LEVELS.index(level)
                            if idx_level < len(random_means):
                                random_means[idx_level].append(random_accs[level])
            
            # Compute means
            top_curve = [np.mean(vals) if vals else np.nan for vals in top_means]
            top_err = [np.std(vals) if vals else 0 for vals in top_means]
            random_curve = [np.mean(vals) if vals else np.nan for vals in random_means]
            random_err = [np.std(vals) if vals else 0 for vals in random_means]
            
            color = METHOD_COLORS[method]
            label = METHOD_LABELS[method]
            
            # Plot top heads (solid)
            if top_curve:
                ax.errorbar(ABLATION_LEVELS[:len(top_curve)], top_curve, 
                           yerr=top_err, label=f"{label} (top)", 
                           color=color, marker='o', linewidth=2, capsize=3)
            
            # Plot random heads (dashed)
            if random_curve:
                ax.errorbar(ABLATION_LEVELS[:len(random_curve)], random_curve,
                           yerr=random_err, label=f"{label} (random)",
                           color=color, marker='s', linestyle='--', linewidth=2, capsize=3, alpha=0.6)
        
        ax.set_xlabel("Number of Ablated Heads", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{QUESTION_LABELS[question]}", fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_xticks(ABLATION_LEVELS)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')
    
    plt.suptitle("Figure 8 Replication: Ablation by Question Type\n(Comparing Methods)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "figure8_option_a_by_question.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_option_b(results):
    """Option B: Per-method figures comparing questions."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    question_colors = {
        "inc_state": "#1f77b4",
        "inc_year": "#ff7f0e", 
        "employee_count": "#2ca02c",
        "hq_state": "#d62728"
    }
    
    for idx, method in enumerate(METHODS):
        ax = axes[idx]
        
        for question in QUESTIONS:
            # Collect data for this method-question combination
            top_means = [[] for _ in ABLATION_LEVELS]
            random_means = [[] for _ in ABLATION_LEVELS]
            
            for model in MODELS:
                for tokens in TOKEN_LENGTHS:
                    result = results[method][model][question][tokens]
                    baseline, top_accs, random_accs = extract_ablation_curve(result)
                    
                    if baseline is None:
                        continue
                    
                    for i, level in enumerate(ABLATION_LEVELS):
                        if level in top_accs:
                            top_means[i].append(top_accs[level])
                        if level in random_accs:
                            random_means[i].append(random_accs[level])
            
            # Compute means
            top_curve = [np.mean(vals) if vals else np.nan for vals in top_means]
            random_curve = [np.mean(vals) if vals else np.nan for vals in random_means]
            
            color = question_colors[question]
            label = QUESTION_LABELS[question]
            
            # Plot top heads (solid)
            valid_top = [(l, v) for l, v in zip(ABLATION_LEVELS, top_curve) if not np.isnan(v)]
            if valid_top:
                levels, values = zip(*valid_top)
                ax.plot(levels, values, label=f"{label} (top)", 
                       color=color, marker='o', linewidth=2)
            
            # Plot random heads (dashed)
            valid_rand = [(l, v) for l, v in zip(ABLATION_LEVELS, random_curve) if not np.isnan(v)]
            if valid_rand:
                levels, values = zip(*valid_rand)
                ax.plot(levels, values, label=f"{label} (random)",
                       color=color, marker='s', linestyle='--', linewidth=2, alpha=0.5)
        
        ax.set_xlabel("Number of Ablated Heads", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{METHOD_LABELS[method]}", fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_xticks(ABLATION_LEVELS)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')
    
    plt.suptitle("Figure 8 Replication: Ablation by Method\n(Comparing Questions)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "figure8_option_b_by_method.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_option_c(results):
    """Option C: Grid layout - comprehensive view."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    
    for row_idx, question in enumerate(QUESTIONS):
        for col_idx, method in enumerate(METHODS):
            ax = axes[row_idx, col_idx]
            
            # Collect data
            top_means = [[] for _ in ABLATION_LEVELS]
            random_means = [[] for _ in ABLATION_LEVELS]
            baselines = []
            
            for model in MODELS:
                for tokens in TOKEN_LENGTHS:
                    result = results[method][model][question][tokens]
                    baseline, top_accs, random_accs = extract_ablation_curve(result)
                    
                    if baseline is None:
                        continue
                    
                    baselines.append(baseline)
                    
                    for i, level in enumerate(ABLATION_LEVELS):
                        if level in top_accs:
                            top_means[i].append(top_accs[level])
                        if level in random_accs:
                            random_means[i].append(random_accs[level])
            
            # Compute means and stds
            top_curve = [np.mean(vals) if vals else np.nan for vals in top_means]
            top_err = [np.std(vals) if vals else 0 for vals in top_means]
            random_curve = [np.mean(vals) if vals else np.nan for vals in random_means]
            random_err = [np.std(vals) if vals else 0 for vals in random_means]
            baseline_mean = np.mean(baselines) if baselines else np.nan
            
            # Plot baseline
            ax.axhline(y=baseline_mean, color='gray', linestyle=':', linewidth=1, label='Baseline')
            
            # Plot top heads
            valid_idx = [i for i, v in enumerate(top_curve) if not np.isnan(v)]
            if valid_idx:
                levels = [ABLATION_LEVELS[i] for i in valid_idx]
                values = [top_curve[i] for i in valid_idx]
                errors = [top_err[i] for i in valid_idx]
                ax.errorbar(levels, values, yerr=errors, 
                           label='Top Heads', color='#d62728', marker='o', 
                           linewidth=2, capsize=3)
            
            # Plot random heads
            valid_idx = [i for i, v in enumerate(random_curve) if not np.isnan(v)]
            if valid_idx:
                levels = [ABLATION_LEVELS[i] for i in valid_idx]
                values = [random_curve[i] for i in valid_idx]
                errors = [random_err[i] for i in valid_idx]
                ax.errorbar(levels, values, yerr=errors,
                           label='Random Heads', color='#1f77b4', marker='s',
                           linestyle='--', linewidth=2, capsize=3, alpha=0.7)
            
            ax.set_ylim(0, 1.05)
            ax.set_xticks(ABLATION_LEVELS)
            ax.grid(True, alpha=0.3)
            
            # Labels
            if row_idx == 3:
                ax.set_xlabel("# Ablated Heads", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel("Accuracy", fontsize=10)
            
            # Title for top row
            if row_idx == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=12, fontweight='bold')
            
            # Row label
            if col_idx == 2:
                ax.text(1.05, 0.5, QUESTION_LABELS[question], transform=ax.transAxes,
                       fontsize=11, fontweight='bold', va='center', rotation=-90)
            
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc='lower left')
    
    plt.suptitle("Figure 8 Replication: Complete Ablation Study Grid\n(Top Heads vs Random Baseline)", 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "figure8_option_c_grid.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_drop_summary(results):
    """Summary bar chart showing accuracy drop at 50 heads for all combinations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Collect accuracy drops at max ablation level (50 heads)
    drops = defaultdict(list)
    
    for method in METHODS:
        for model in MODELS:
            for question in QUESTIONS:
                for tokens in TOKEN_LENGTHS:
                    result = results[method][model][question][tokens]
                    if result is None:
                        continue
                    
                    baseline = result["baseline"]["accuracy"]
                    
                    # Get accuracy at 50 heads
                    for abl in result.get("top_heads_ablations", []):
                        if abl["num_heads"] == 50:
                            drop = baseline - abl["accuracy"]
                            drops[(method, question)].append(drop)
    
    # Prepare data for plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(QUESTIONS))
    width = 0.25
    
    for i, method in enumerate(METHODS):
        means = []
        stds = []
        for question in QUESTIONS:
            vals = drops.get((method, question), [0])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        
        bars = ax.bar(x + i * width, means, width, label=METHOD_LABELS[method],
                     yerr=stds, capsize=3, color=METHOD_COLORS[method])
    
    ax.set_xlabel("Question Type", fontsize=12)
    ax.set_ylabel("Accuracy Drop (Baseline - Ablated)", fontsize=12)
    ax.set_title("Accuracy Drop at 50 Heads Ablated\n(Higher = More Causally Important Heads)", 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([QUESTION_LABELS[q] for q in QUESTIONS])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "accuracy_drop_summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Phase 4: Generating Figure 8 Replications")
    print("=" * 60)
    
    # Load all results
    print("\nLoading Phase 3 results...")
    results = load_all_results()
    
    # Generate figures
    print("\nGenerating Option A (per-question)...")
    plot_option_a(results)
    
    print("\nGenerating Option B (per-method)...")
    plot_option_b(results)
    
    print("\nGenerating Option C (grid)...")
    plot_option_c(results)
    
    print("\nGenerating accuracy drop summary...")
    plot_accuracy_drop_summary(results)
    
    print("\n" + "=" * 60)
    print("Figure 8 generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

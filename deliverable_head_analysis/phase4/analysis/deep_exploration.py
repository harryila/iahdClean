#!/usr/bin/env python3
"""
Deep Exploration of Phase 2 and Phase 3 Data

This script comprehensively explores the head detection and ablation data
to find interesting patterns across:
- Questions
- Token lengths  
- Models (Instruct vs Base)
- Methods (Summed Attention, Wu24, QRHead)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Configuration
PHASE2_DIR = "/home/ubuntu/iahdClean/deliverable_head_analysis/phase2"
PHASE3_DIR = "/home/ubuntu/iahdClean/deliverable_head_analysis/phase3"
OUTPUT_DIR = "/home/ubuntu/iahdClean/deliverable_head_analysis/phase4/exploration_figures"

METHODS = ["summed_attention", "retrieval_head_wu24", "qrhead"]
METHOD_LABELS = {
    "summed_attention": "Summed Attention",
    "retrieval_head_wu24": "Wu24 Retrieval",
    "qrhead": "QRHead"
}
MODELS = ["llama3_instruct", "llama3_base"]
MODEL_LABELS = {"llama3_instruct": "Instruct", "llama3_base": "Base"}
QUESTIONS = ["inc_state", "inc_year", "employee_count", "hq_state"]
QUESTION_LABELS = {
    "inc_state": "Inc. State",
    "inc_year": "Inc. Year", 
    "employee_count": "Employee Count",
    "hq_state": "HQ State"
}
TOKEN_LENGTHS = [2048, 4096, 6144, 8192]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_phase2_data(method, model, question, tokens, top_n=50):
    """Load Phase 2 head rankings."""
    filepath = os.path.join(
        PHASE2_DIR, method, "results", model, question, f"tokens_{tokens}.json"
    )
    if not os.path.exists(filepath):
        return None, set()
    
    with open(filepath) as f:
        data = json.load(f)
    
    # Extract top heads
    top_heads = set()
    head_rankings = data.get("head_rankings", [])[:top_n]
    for item in head_rankings:
        head_str = item["head"]
        layer = int(head_str.split("H")[0][1:])
        head = int(head_str.split("H")[1])
        top_heads.add((layer, head))
    
    return data, top_heads


def load_phase3_data(method, model, question, tokens):
    """Load Phase 3 ablation results."""
    filepath = os.path.join(
        PHASE3_DIR, method, "results", model, question, f"tokens_{tokens}.json"
    )
    if not os.path.exists(filepath):
        return None
    
    with open(filepath) as f:
        return json.load(f)


def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def compute_all_overlaps():
    """Compute comprehensive head overlap statistics."""
    print("Loading all Phase 2 data...")
    
    # Store all head sets indexed by (method, model, question, tokens)
    all_heads = {}
    all_data = {}
    
    for method in METHODS:
        for model in MODELS:
            for question in QUESTIONS:
                for tokens in TOKEN_LENGTHS:
                    data, heads = load_phase2_data(method, model, question, tokens)
                    key = (method, model, question, tokens)
                    all_heads[key] = heads
                    all_data[key] = data
    
    return all_heads, all_data


def plot_question_overlap_by_config(all_heads):
    """
    Plot head overlap between questions for each method-model-token combination.
    Creates a 3x2 grid (methods x models) with overlap heatmaps.
    """
    print("\nGenerating question overlap heatmaps...")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    
    for i, method in enumerate(METHODS):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            
            # Average overlap matrix across token lengths
            overlap_matrix = np.zeros((len(QUESTIONS), len(QUESTIONS)))
            count_matrix = np.zeros((len(QUESTIONS), len(QUESTIONS)))
            
            for tokens in TOKEN_LENGTHS:
                for qi, q1 in enumerate(QUESTIONS):
                    for qj, q2 in enumerate(QUESTIONS):
                        h1 = all_heads[(method, model, q1, tokens)]
                        h2 = all_heads[(method, model, q2, tokens)]
                        if h1 and h2:
                            overlap_matrix[qi, qj] += jaccard_similarity(h1, h2)
                            count_matrix[qi, qj] += 1
            
            # Average
            overlap_matrix = np.divide(overlap_matrix, count_matrix, 
                                       where=count_matrix > 0, out=overlap_matrix)
            
            sns.heatmap(overlap_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                       xticklabels=[QUESTION_LABELS[q] for q in QUESTIONS],
                       yticklabels=[QUESTION_LABELS[q] for q in QUESTIONS],
                       ax=ax, vmin=0, vmax=1, cbar=True)
            
            ax.set_title(f"{METHOD_LABELS[method]} - {MODEL_LABELS[model]}", 
                        fontsize=11, fontweight='bold')
    
    # 5. Universal heads (aggregated across ALL dimensions)
    print("5. Universal heads (all methods combined)...")
    hf = defaultdict(int)
    total = 0
    for key, heads in all_h.items():
        if heads:
            total += 1
            for h in heads: hf[h] += 1
    sorted_h = sorted(hf.items(), key=lambda x: x[1], reverse=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    top30 = sorted_h[:30]
    labels = [f"L{h[0][0]}H{h[0][1]}" for h in top30]
    freqs = [h[1] / total * 100 for h in top30]
    ax.barh(range(30), freqs, color=plt.cm.viridis(np.linspace(0.2, 0.8, 30)))
    ax.set_yticks(range(30))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Frequency (%)")
    ax.set_title("Top 30 Universal Heads", fontweight='bold')
    ax.invert_yaxis()
    
    ax = axes[1]
    lc = defaultdict(int)
    for (layer, _), freq in sorted_h[:100]: lc[layer] += freq
    layers = sorted(lc.keys())
    ax.bar(layers, [lc[l] for l in layers], color='steelblue', alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cumulative Frequency")
    ax.set_title("Layer Distribution (Top 100)", fontweight='bold')
    plt.suptitle("Universal Retrieval Heads\n(Aggregated across 3 methods × 2 models × 4 questions × 4 token lengths = 96 configs)", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/universal_heads.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5b. Universal heads PER METHOD (more meaningful given low method overlap)
    print("5b. Universal heads per method...")
    method_colors = {"summed_attention": "Oranges", "retrieval_head_wu24": "Greens", "qrhead": "Blues"}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for mi, m in enumerate(METHODS):
        ax = axes[mi]
        # Count heads for this method only (across models, questions, tokens = 2×4×4 = 32 configs)
        hf_method = defaultdict(int)
        total_method = 0
        for mo in MODELS:
            for q in QUESTIONS:
                for t in TOKENS:
                    heads = all_h[(m, mo, q, t)]
                    if heads:
                        total_method += 1
                        for h in heads: hf_method[h] += 1
        sorted_method = sorted(hf_method.items(), key=lambda x: x[1], reverse=True)
        
        top20 = sorted_method[:20]
        labels = [f"L{h[0][0]}H{h[0][1]}" for h in top20]
        freqs = [h[1] / total_method * 100 for h in top20]
        colors = plt.cm.get_cmap(method_colors[m])(np.linspace(0.3, 0.9, 20))
        ax.barh(range(20), freqs, color=colors)
        ax.set_yticks(range(20))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Frequency (%)")
        ax.set_title(f"{MLABELS[m]}\n(across 2 models × 4 questions × 4 tokens = 32 configs)", fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
    
    plt.suptitle("Universal Heads by Method (Top 20)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/universal_heads_by_method.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5c. Layer distribution comparison across methods
    print("5c. Layer distribution by method...")
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width = 0.25
    all_layers = list(range(32))  # Assuming 32 layers
    method_layer_counts = {}
    
    for m in METHODS:
        hf_method = defaultdict(int)
        for mo in MODELS:
            for q in QUESTIONS:
                for t in TOKENS:
                    heads = all_h[(m, mo, q, t)]
                    if heads:
                        for h in heads: hf_method[h] += 1
        sorted_method = sorted(hf_method.items(), key=lambda x: x[1], reverse=True)[:100]
        lc = defaultdict(int)
        for (layer, _), freq in sorted_method: lc[layer] += freq
        method_layer_counts[m] = lc
    
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    for mi, m in enumerate(METHODS):
        lc = method_layer_counts[m]
        x_pos = [l + mi * bar_width for l in all_layers]
        heights = [lc.get(l, 0) for l in all_layers]
        ax.bar(x_pos, heights, bar_width, label=MLABELS[m], color=colors[mi], alpha=0.8)
    
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cumulative Frequency (Top 100 heads)")
    ax.set_title("Layer Distribution of Top Heads by Method", fontweight='bold', fontsize=14)
    ax.set_xticks([l + bar_width for l in all_layers])
    ax.set_xticklabels(all_layers, fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/layer_distribution_by_method.png", dpi=150, bbox_inches='tight')
                            overlap_matrix[ti, tj] += jaccard_similarity(h1, h2)
                            count_matrix[ti, tj] += 1
            
            overlap_matrix = np.divide(overlap_matrix, count_matrix,
                                       where=count_matrix > 0, out=overlap_matrix)
            
            sns.heatmap(overlap_matrix, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=[f"{t//1024}K" for t in TOKEN_LENGTHS],
                       yticklabels=[f"{t//1024}K" for t in TOKEN_LENGTHS],
                       ax=ax, vmin=0, vmax=1)
            
            ax.set_title(f"{MODEL_LABELS[model]} - {METHOD_LABELS[method]}", 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel("Token Length")
            ax.set_ylabel("Token Length")
    
    plt.suptitle("Head Overlap Across Context Lengths\n(Are Important Heads Stable as Context Grows?)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "token_length_overlap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_overlap(all_heads):
    """
    Plot head overlap between Instruct and Base models.
    Are the same heads important regardless of instruction tuning?
    """
    print("\nGenerating model overlap analysis...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    model_overlaps = {method: [] for method in METHODS}
    
    for i, method in enumerate(METHODS):
        ax = axes[i]
        
        # Overlap by question
        question_overlaps = []
        
        for question in QUESTIONS:
            q_overlaps = []
            for tokens in TOKEN_LENGTHS:
                h_instruct = all_heads[(method, "llama3_instruct", question, tokens)]
                h_base = all_heads[(method, "llama3_base", question, tokens)]
                if h_instruct and h_base:
                    overlap = jaccard_similarity(h_instruct, h_base)
                    q_overlaps.append(overlap)
                    model_overlaps[method].append(overlap)
            
            if q_overlaps:
                question_overlaps.append(np.mean(q_overlaps))
        
        # Bar chart
        colors = plt.cm.Set2(np.linspace(0, 1, len(QUESTIONS)))
        bars = ax.bar([QUESTION_LABELS[q] for q in QUESTIONS], question_overlaps, color=colors)
        ax.set_ylabel("Jaccard Similarity")
        ax.set_title(f"{METHOD_LABELS[method]}\nMean: {np.mean(question_overlaps):.2f}", 
                    fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.axhline(y=np.mean(question_overlaps), color='red', linestyle='--', 
                  label=f'Mean={np.mean(question_overlaps):.2f}')
        
        for bar, val in zip(bars, question_overlaps):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.tick_params(axis='x', rotation=30)
    
    plt.suptitle("Instruct vs Base Model: Head Overlap by Question Type\n(Are Same Heads Important After Instruction Tuning?)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "model_overlap_instruct_vs_base.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return model_overlaps


def plot_method_overlap_detailed(all_heads):
    """
    Detailed method overlap: which heads do methods agree on?
    """
    print("\nGenerating detailed method overlap analysis...")
    
    # 1. Overall method overlap heatmap (averaged across all configs)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # First subplot: Overall overlap matrix
    ax = axes[0]
    overlap_matrix = np.zeros((len(METHODS), len(METHODS)))
    count_matrix = np.zeros((len(METHODS), len(METHODS)))
    
    for model in MODELS:
        for question in QUESTIONS:
            for tokens in TOKEN_LENGTHS:
                for mi, m1 in enumerate(METHODS):
                    for mj, m2 in enumerate(METHODS):
                        h1 = all_heads[(m1, model, question, tokens)]
                        h2 = all_heads[(m2, model, question, tokens)]
                        if h1 and h2:
                            overlap_matrix[mi, mj] += jaccard_similarity(h1, h2)
                            count_matrix[mi, mj] += 1
    
    overlap_matrix = np.divide(overlap_matrix, count_matrix,
                               where=count_matrix > 0, out=overlap_matrix)
    
    sns.heatmap(overlap_matrix, annot=True, fmt='.2f', cmap='Purples',
               xticklabels=[METHOD_LABELS[m] for m in METHODS],
               yticklabels=[METHOD_LABELS[m] for m in METHODS],
               ax=ax, vmin=0, vmax=1)
    ax.set_title("Overall Method Overlap\n(Across All Configs)", fontweight='bold')
    
    # Second subplot: Overlap by question type
    ax = axes[1]
    method_pairs = list(combinations(range(len(METHODS)), 2))
    pair_labels = [f"{METHOD_LABELS[METHODS[i]][:6]} vs\n{METHOD_LABELS[METHODS[j]][:6]}" 
                   for i, j in method_pairs]
    
    x = np.arange(len(QUESTIONS))
    width = 0.25
    
    for pi, (mi, mj) in enumerate(method_pairs):
        overlaps = []
        for question in QUESTIONS:
            q_overlaps = []
            for model in MODELS:
                for tokens in TOKEN_LENGTHS:
                    h1 = all_heads[(METHODS[mi], model, question, tokens)]
                    h2 = all_heads[(METHODS[mj], model, question, tokens)]
                    if h1 and h2:
                        q_overlaps.append(jaccard_similarity(h1, h2))
            overlaps.append(np.mean(q_overlaps) if q_overlaps else 0)
        
        ax.bar(x + pi * width, overlaps, width, label=pair_labels[pi])
    
    ax.set_ylabel("Jaccard Similarity")
    ax.set_xticks(x + width)
    ax.set_xticklabels([QUESTION_LABELS[q] for q in QUESTIONS])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.5)
    ax.set_title("Method Overlap by Question", fontweight='bold')
    
    # Third subplot: Overlap by token length
    ax = axes[2]
    x = np.arange(len(TOKEN_LENGTHS))
    
    for pi, (mi, mj) in enumerate(method_pairs):
        overlaps = []
        for tokens in TOKEN_LENGTHS:
            t_overlaps = []
            for model in MODELS:
                for question in QUESTIONS:
                    h1 = all_heads[(METHODS[mi], model, question, tokens)]
                    h2 = all_heads[(METHODS[mj], model, question, tokens)]
                    if h1 and h2:
                        t_overlaps.append(jaccard_similarity(h1, h2))
            overlaps.append(np.mean(t_overlaps) if t_overlaps else 0)
        
        ax.bar(x + pi * width, overlaps, width, label=pair_labels[pi])
    
    ax.set_ylabel("Jaccard Similarity")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{t//1024}K" for t in TOKEN_LENGTHS])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.5)
    ax.set_title("Method Overlap by Token Length", fontweight='bold')
    
    plt.suptitle("Method Agreement: Which Heads Do Different Detection Methods Identify?",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "method_overlap_detailed.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def find_universal_heads(all_heads, top_n=50):
    """
    Find heads that appear consistently across many configurations.
    These are the "universal retrieval heads."
    """
    print("\nFinding universal heads...")
    
    head_frequency = defaultdict(int)
    total_configs = 0
    
    for key, heads in all_heads.items():
        if heads:
            total_configs += 1
            for head in heads:
                head_frequency[head] += 1
    
    # Sort by frequency
    sorted_heads = sorted(head_frequency.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_heads, total_configs


def plot_universal_heads(all_heads):
    """
    Visualize the most universal heads across all configurations.
    """
    print("\nGenerating universal heads visualization...")
    
    sorted_heads, total_configs = find_universal_heads(all_heads)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Top 30 most frequent heads
    ax = axes[0]
    top_30 = sorted_heads[:30]
    head_labels = [f"L{h[0][0]}H{h[0][1]}" for h in top_30]
    frequencies = [h[1] / total_configs * 100 for h in top_30]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_30)))
    bars = ax.barh(range(len(top_30)), frequencies, color=colors)
    ax.set_yticks(range(len(top_30)))
    ax.set_yticklabels(head_labels, fontsize=8)
    ax.set_xlabel("Frequency Across Configs (%)")
    ax.set_title("Top 30 Most Universal Heads\n(Appear in Top 50 Across Configs)", fontweight='bold')
    ax.invert_yaxis()
    
    # Add frequency labels
    for bar, freq in zip(bars, frequencies):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{freq:.1f}%', va='center', fontsize=7)
    
    # Right: Layer distribution of top heads
    ax = axes[1]
    layer_counts = defaultdict(int)
    for (layer, head), freq in sorted_heads[:100]:  # Top 100 heads
        layer_counts[layer] += freq
    
    layers = sorted(layer_counts.keys())
    counts = [layer_counts[l] for l in layers]
    
    ax.bar(layers, counts, color='steelblue', alpha=0.7)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Total Frequency (sum across top 100 heads)")
    ax.set_title("Layer Distribution of Important Heads", fontweight='bold')
    ax.set_xticks(range(0, 32, 4))
    
    plt.suptitle("Universal Retrieval Heads: Which Heads Are Important Everywhere?",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "universal_heads.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return sorted_heads[:30]


def plot_head_rank_stability(all_heads, all_data):
    """
    Analyze how stable head rankings are across configurations.
    Does the #1 head stay #1?
    """
    print("\nAnalyzing head ranking stability...")
    
    # Track rank of specific heads across configs
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Find top heads overall and track their ranks
    sorted_heads, _ = find_universal_heads(all_heads)
    top_5_heads = [h[0] for h in sorted_heads[:5]]
    
    # Plot 1: Rank distribution for top 5 universal heads
    ax = axes[0, 0]
    rank_data = {i: [] for i in range(5)}
    
    for key, data in all_data.items():
        if data is None:
            continue
        head_rankings = data.get("head_rankings", [])
        head_to_rank = {}
        for item in head_rankings:
            head_str = item["head"]
            layer = int(head_str.split("H")[0][1:])
            head = int(head_str.split("H")[1])
            head_to_rank[(layer, head)] = item["rank"]
        
        for i, head in enumerate(top_5_heads):
            if head in head_to_rank:
                rank_data[i].append(head_to_rank[head])
    
    positions = []
    data_for_box = []
    labels = []
    for i, head in enumerate(top_5_heads):
        if rank_data[i]:
            data_for_box.append(rank_data[i])
            labels.append(f"L{head[0]}H{head[1]}")
            positions.append(i)
    
    bp = ax.boxplot(data_for_box, positions=positions, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rank Position")
    ax.set_title("Rank Distribution of Top 5 Universal Heads\n(Lower = More Important)", fontweight='bold')
    
    # Plot 2: Score distribution by method
    ax = axes[0, 1]
    method_scores = {method: [] for method in METHODS}
    
    for key, data in all_data.items():
        if data is None:
            continue
        method = key[0]
        head_rankings = data.get("head_rankings", [])[:10]
        for item in head_rankings:
            method_scores[method].append(item["score"])
    
    # Normalize scores for comparison
    box_data = []
    box_labels = []
    for method in METHODS:
        if method_scores[method]:
            scores = np.array(method_scores[method])
            # Normalize to 0-1 range
            if scores.max() > scores.min():
                normalized = (scores - scores.min()) / (scores.max() - scores.min())
                box_data.append(normalized)
                box_labels.append(METHOD_LABELS[method][:8])
    
    bp = ax.boxplot(box_data, patch_artist=True)
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(box_labels)
    ax.set_ylabel("Normalized Score (0-1)")
    ax.set_title("Score Distribution by Method\n(Top 10 Heads per Config)", fontweight='bold')
    
    # Plot 3: Head overlap as function of top-N threshold
    ax = axes[1, 0]
    top_n_values = [10, 20, 30, 40, 50, 75, 100]
    
    for method in METHODS:
        overlaps = []
        for top_n in top_n_values:
            # Compute average pairwise overlap for this method
            method_overlaps = []
            configs = [(m, mo, q, t) for m, mo, q, t in all_heads.keys() if m == method]
            for c1, c2 in combinations(configs, 2):
                _, h1 = load_phase2_data(*c1, top_n=top_n)
                _, h2 = load_phase2_data(*c2, top_n=top_n)
                if h1 and h2:
                    method_overlaps.append(jaccard_similarity(h1, h2))
            overlaps.append(np.mean(method_overlaps) if method_overlaps else 0)
        
        ax.plot(top_n_values, overlaps, 'o-', label=METHOD_LABELS[method], linewidth=2)
    
    ax.set_xlabel("Top-N Threshold")
    ax.set_ylabel("Mean Pairwise Jaccard Overlap")
    ax.set_title("Head Overlap vs Top-N Threshold\n(More Heads = More Agreement?)", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Consensus heads - in top-N of all methods?
    ax = axes[1, 1]
    
    # For each config, find heads in top-20 of ALL methods
    consensus_counts = defaultdict(int)
    
    for model in MODELS:
        for question in QUESTIONS:
            for tokens in TOKEN_LENGTHS:
                method_heads = []
                for method in METHODS:
                    _, heads = load_phase2_data(method, model, question, tokens, top_n=20)
                    if heads:
                        method_heads.append(heads)
                
                if len(method_heads) == 3:
                    # Find intersection
                    consensus = method_heads[0] & method_heads[1] & method_heads[2]
                    for head in consensus:
                        consensus_counts[head] += 1
    
    if consensus_counts:
        sorted_consensus = sorted(consensus_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        head_labels = [f"L{h[0][0]}H{h[0][1]}" for h, _ in sorted_consensus]
        counts = [c for _, c in sorted_consensus]
        
        ax.barh(range(len(sorted_consensus)), counts, color='coral')
        ax.set_yticks(range(len(sorted_consensus)))
        ax.set_yticklabels(head_labels, fontsize=9)
        ax.set_xlabel("Number of Configs")
        ax.set_title("Consensus Heads: In Top-20 of ALL Methods\n(Multi-Method Agreement)", fontweight='bold')
        ax.invert_yaxis()
    
    plt.suptitle("Head Ranking Stability Analysis",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "head_rank_stability.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ablation_effectiveness_deep(all_heads):
    """
    Deep dive into ablation effectiveness from Phase 3.
    """
    print("\nAnalyzing ablation effectiveness...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Collect all ablation data
    ablation_data = []
    for method in METHODS:
        for model in MODELS:
            for question in QUESTIONS:
                for tokens in TOKEN_LENGTHS:
                    data = load_phase3_data(method, model, question, tokens)
                    if data:
                        baseline = data.get("baseline", {}).get("accuracy", 0)
                        for ablation in data.get("top_heads_ablations", []):
                            ablation_data.append({
                                "method": method,
                                "model": model,
                                "question": question,
                                "tokens": tokens,
                                "num_heads": ablation["num_heads"],
                                "accuracy": ablation["accuracy"],
                                "drop": ablation["accuracy_drop"],
                                "baseline": baseline
                            })
    
    # Plot 1: Accuracy drop by method
    ax = axes[0, 0]
    for method in METHODS:
        method_data = [d for d in ablation_data if d["method"] == method]
        levels = sorted(set(d["num_heads"] for d in method_data))
        drops = []
        for level in levels:
            level_drops = [d["drop"] for d in method_data if d["num_heads"] == level]
            drops.append(np.mean(level_drops) if level_drops else 0)
        ax.plot(levels, drops, 'o-', label=METHOD_LABELS[method], linewidth=2, markersize=8)
    
    ax.set_xlabel("Number of Heads Ablated")
    ax.set_ylabel("Mean Accuracy Drop")
    ax.set_title("Ablation Effectiveness by Method", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy drop by question type
    ax = axes[0, 1]
    colors = plt.cm.Set1(np.linspace(0, 1, len(QUESTIONS)))
    for qi, question in enumerate(QUESTIONS):
        q_data = [d for d in ablation_data if d["question"] == question]
        levels = sorted(set(d["num_heads"] for d in q_data))
        drops = []
        for level in levels:
            level_drops = [d["drop"] for d in q_data if d["num_heads"] == level]
            drops.append(np.mean(level_drops) if level_drops else 0)
        ax.plot(levels, drops, 'o-', label=QUESTION_LABELS[question], 
               color=colors[qi], linewidth=2, markersize=8)
    
    ax.set_xlabel("Number of Heads Ablated")
    ax.set_ylabel("Mean Accuracy Drop")
    ax.set_title("Ablation Effectiveness by Question Type", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Instruct vs Base ablation
    ax = axes[1, 0]
    for model in MODELS:
        m_data = [d for d in ablation_data if d["model"] == model]
        levels = sorted(set(d["num_heads"] for d in m_data))
        drops = []
        for level in levels:
            level_drops = [d["drop"] for d in m_data if d["num_heads"] == level]
            drops.append(np.mean(level_drops) if level_drops else 0)
        ax.plot(levels, drops, 'o-', label=MODEL_LABELS[model], linewidth=2, markersize=8)
    
    ax.set_xlabel("Number of Heads Ablated")
    ax.set_ylabel("Mean Accuracy Drop")
    ax.set_title("Ablation: Instruct vs Base Model", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Token length effect on ablation
    ax = axes[1, 1]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(TOKEN_LENGTHS)))
    for ti, tokens in enumerate(TOKEN_LENGTHS):
        t_data = [d for d in ablation_data if d["tokens"] == tokens]
        levels = sorted(set(d["num_heads"] for d in t_data))
        drops = []
        for level in levels:
            level_drops = [d["drop"] for d in t_data if d["num_heads"] == level]
            drops.append(np.mean(level_drops) if level_drops else 0)
        ax.plot(levels, drops, 'o-', label=f"{tokens//1024}K tokens", 
               color=colors[ti], linewidth=2, markersize=8)
    
    ax.set_xlabel("Number of Heads Ablated")
    ax.set_ylabel("Mean Accuracy Drop")
    ax.set_title("Ablation by Context Length", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Deep Dive: Ablation Effectiveness Analysis",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "ablation_effectiveness_deep.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comprehensive_overlap_grid(all_heads):
    """
    Create a comprehensive 4-way overlap visualization:
    Question x Token Length x Model x Method
    """
    print("\nGenerating comprehensive overlap grid...")
    
    # Create a summary statistics figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Summary data collection
    summary_data = {
        "question_overlap": [],
        "token_overlap": [],
        "model_overlap": [],
        "method_overlap": []
    }
    
    # Question overlap (within same method/model/token)
    for method in METHODS:
        for model in MODELS:
            for tokens in TOKEN_LENGTHS:
                for q1, q2 in combinations(QUESTIONS, 2):
                    h1 = all_heads[(method, model, q1, tokens)]
                    h2 = all_heads[(method, model, q2, tokens)]
                    if h1 and h2:
                        summary_data["question_overlap"].append(jaccard_similarity(h1, h2))
    
    # Token overlap (within same method/model/question)
    for method in METHODS:
        for model in MODELS:
            for question in QUESTIONS:
                for t1, t2 in combinations(TOKEN_LENGTHS, 2):
                    h1 = all_heads[(method, model, question, t1)]
                    h2 = all_heads[(method, model, question, t2)]
                    if h1 and h2:
                        summary_data["token_overlap"].append(jaccard_similarity(h1, h2))
    
    # Model overlap (within same method/question/token)
    for method in METHODS:
        for question in QUESTIONS:
            for tokens in TOKEN_LENGTHS:
                h1 = all_heads[(method, "llama3_instruct", question, tokens)]
                h2 = all_heads[(method, "llama3_base", question, tokens)]
                if h1 and h2:
                    summary_data["model_overlap"].append(jaccard_similarity(h1, h2))
    
    # Method overlap (within same model/question/token)
    for model in MODELS:
        for question in QUESTIONS:
            for tokens in TOKEN_LENGTHS:
                for m1, m2 in combinations(METHODS, 2):
                    h1 = all_heads[(m1, model, question, tokens)]
                    h2 = all_heads[(m2, model, question, tokens)]
                    if h1 and h2:
                        summary_data["method_overlap"].append(jaccard_similarity(h1, h2))
    
    # Plot distributions
    categories = ["Question\nOverlap", "Token Length\nOverlap", "Model\nOverlap", "Method\nOverlap"]
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: Box plot comparison
    ax = axes[0, 0]
    bp = ax.boxplot([summary_data["question_overlap"], summary_data["token_overlap"],
                     summary_data["model_overlap"], summary_data["method_overlap"]],
                    tick_labels=categories, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Jaccard Similarity")
    ax.set_title("Head Overlap Distribution by Factor", fontweight='bold')
    
    # Add mean markers
    for i, key in enumerate(["question_overlap", "token_overlap", "model_overlap", "method_overlap"]):
        mean_val = np.mean(summary_data[key])
        ax.scatter(i+1, mean_val, marker='D', color='black', s=50, zorder=5)
        ax.annotate(f'{mean_val:.2f}', (i+1, mean_val), textcoords="offset points",
                   xytext=(10, 0), fontsize=9)
    
    # Plot 2: Violin plot
    ax = axes[0, 1]
    parts = ax.violinplot([summary_data["question_overlap"], summary_data["token_overlap"],
                           summary_data["model_overlap"], summary_data["method_overlap"]],
                          positions=[1, 2, 3, 4], showmeans=True, showmedians=True)
    
    for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(categories)
    ax.set_ylabel("Jaccard Similarity")
    ax.set_title("Head Overlap Distributions (Violin)", fontweight='bold')
    
    # Plot 3: Summary bar chart with error bars
    ax = axes[1, 0]
    means = [np.mean(summary_data[k]) for k in ["question_overlap", "token_overlap", 
                                                 "model_overlap", "method_overlap"]]
    stds = [np.std(summary_data[k]) for k in ["question_overlap", "token_overlap",
                                               "model_overlap", "method_overlap"]]
    
    bars = ax.bar(categories, means, yerr=stds, color=colors, alpha=0.7, 
                  capsize=5, edgecolor='black')
    ax.set_ylabel("Mean Jaccard Similarity")
    ax.set_title("Mean Overlap by Factor (±1 std)", fontweight='bold')
    ax.set_ylim(0, 1)
    
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{mean:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 4: Key findings text
    ax = axes[1, 1]
    ax.axis('off')
    
    findings = f"""
    KEY FINDINGS FROM OVERLAP ANALYSIS
    ==================================
    
    1. QUESTION OVERLAP: {np.mean(summary_data['question_overlap']):.1%}
       Different questions share {np.mean(summary_data['question_overlap']):.0%} of heads
       → Moderate question specificity
    
    2. TOKEN LENGTH OVERLAP: {np.mean(summary_data['token_overlap']):.1%}
       Heads are {np.mean(summary_data['token_overlap']):.0%} stable across context lengths
       → {'High' if np.mean(summary_data['token_overlap']) > 0.6 else 'Moderate'} stability
    
    3. MODEL OVERLAP: {np.mean(summary_data['model_overlap']):.1%}
       Instruct and Base share {np.mean(summary_data['model_overlap']):.0%} of heads
       → Instruction tuning {'preserves' if np.mean(summary_data['model_overlap']) > 0.5 else 'changes'} retrieval heads
    
    4. METHOD OVERLAP: {np.mean(summary_data['method_overlap']):.1%}
       Methods agree on only {np.mean(summary_data['method_overlap']):.0%} of heads
       → Methods capture {'similar' if np.mean(summary_data['method_overlap']) > 0.4 else 'different'} aspects
    
    MOST STABLE: {'Token Length' if np.argmax(means) == 1 else categories[np.argmax(means)].replace(chr(10), ' ')}
    LEAST STABLE: {'Method' if np.argmin(means) == 3 else categories[np.argmin(means)].replace(chr(10), ' ')}
    """
    
    ax.text(0.05, 0.95, findings, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle("Comprehensive Head Overlap Analysis: What Factors Affect Head Selection?",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "comprehensive_overlap_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return summary_data


def main():
    print("=" * 60)
    print("DEEP EXPLORATION OF PHASE 2 AND PHASE 3 DATA")
    print("=" * 60)
    
    # Load all data
    all_heads, all_data = compute_all_overlaps()
    
    # Generate visualizations
    plot_question_overlap_by_config(all_heads)
    plot_token_length_overlap(all_heads)
    plot_model_overlap(all_heads)
    plot_method_overlap_detailed(all_heads)
    plot_universal_heads(all_heads)
    plot_head_rank_stability(all_heads, all_data)
    plot_ablation_effectiveness_deep(all_heads)
    summary_data = plot_comprehensive_overlap_grid(all_heads)
    
    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated figures:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\nQuestion overlap:     {np.mean(summary_data['question_overlap']):.3f} ± {np.std(summary_data['question_overlap']):.3f}")
    print(f"Token length overlap: {np.mean(summary_data['token_overlap']):.3f} ± {np.std(summary_data['token_overlap']):.3f}")
    print(f"Model overlap:        {np.mean(summary_data['model_overlap']):.3f} ± {np.std(summary_data['model_overlap']):.3f}")
    print(f"Method overlap:       {np.mean(summary_data['method_overlap']):.3f} ± {np.std(summary_data['method_overlap']):.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 4: Research Questions Analysis

Answers the specific research questions from the deliverable:
1. Do same # of ablated heads affect different questions equally?
2. Similarities between numerical (year/employee) and state heads?
3. Are ablations equally effective on each category?
4. Patterns within numerical vs categorical questions?

Also generates additional interesting analyses.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from itertools import combinations

# Configuration
PHASE2_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "phase2")
PHASE3_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "phase3")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

METHODS = ["summed_attention", "retrieval_head_wu24", "qrhead"]
METHOD_LABELS = {
    "summed_attention": "Summed Attention",
    "retrieval_head_wu24": "Wu24",
    "qrhead": "QRHead"
}

MODELS = ["llama3_instruct", "llama3_base"]
QUESTIONS = ["inc_state", "inc_year", "employee_count", "hq_state"]
QUESTION_LABELS = {
    "inc_state": "Inc. State",
    "inc_year": "Inc. Year",
    "employee_count": "Employee Count",
    "hq_state": "HQ State"
}

# Question categories
CATEGORICAL_QUESTIONS = ["inc_state", "hq_state"]
NUMERICAL_QUESTIONS = ["inc_year", "employee_count"]

TOKEN_LENGTHS = [2048, 4096, 6144, 8192]
ABLATION_LEVELS = [5, 10, 20, 30, 40, 50]


def load_phase2_heads(method, model, question, tokens, top_n=50):
    """Load top N heads from Phase 2 results."""
    filepath = os.path.join(
        PHASE2_DIR, method, "results", model, question, f"tokens_{tokens}.json"
    )
    
    if not os.path.exists(filepath):
        return set()
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    heads = set()
    for h in data.get("head_rankings", [])[:top_n]:
        head_str = h["head"]  # e.g., "L13H18"
        parts = head_str.replace("L", "").split("H")
        heads.add((int(parts[0]), int(parts[1])))
    
    return heads


def load_phase3_results():
    """Load all Phase 3 ablation results."""
    results = {}
    
    for method in METHODS:
        results[method] = {}
        
        for model in MODELS:
            results[method][model] = {}
            
            for question in QUESTIONS:
                results[method][model][question] = {}
                
                for tokens in TOKEN_LENGTHS:
                    filepath = os.path.join(
                        PHASE3_DIR, method, "results", model, question, f"tokens_{tokens}.json"
                    )
                    
                    if os.path.exists(filepath):
                        with open(filepath, "r") as f:
                            results[method][model][question][tokens] = json.load(f)
    
    return results


def question1_equal_impact(results):
    """
    Q1: Do same # of ablated heads affect different questions equally?
    
    Compare accuracy drops across questions at each ablation level.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for level_idx, level in enumerate(ABLATION_LEVELS):
        ax = axes[level_idx // 3, level_idx % 3]
        
        question_drops = defaultdict(list)
        
        for method in METHODS:
            for model in MODELS:
                for question in QUESTIONS:
                    for tokens in TOKEN_LENGTHS:
                        result = results.get(method, {}).get(model, {}).get(question, {}).get(tokens)
                        if result is None:
                            continue
                        
                        baseline = result["baseline"]["accuracy"]
                        
                        for abl in result.get("top_heads_ablations", []):
                            if abl["num_heads"] == level:
                                drop = baseline - abl["accuracy"]
                                question_drops[question].append(drop)
        
        # Box plot
        data = [question_drops[q] for q in QUESTIONS]
        bp = ax.boxplot(data, labels=[QUESTION_LABELS[q] for q in QUESTIONS], patch_artist=True)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_title(f"Ablating {level} Heads", fontsize=12, fontweight='bold')
        ax.set_ylabel("Accuracy Drop")
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Q1: Do Same # of Ablated Heads Affect Different Questions Equally?", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "q1_equal_impact.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return question_drops


def question2_numerical_vs_categorical_heads():
    """
    Q2: Similarities between numerical (year/employee) and state heads?
    
    Compare head overlap between question types.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Collect heads for each question type
    all_heads = defaultdict(set)
    
    for method in METHODS:
        for model in MODELS:
            for question in QUESTIONS:
                for tokens in TOKEN_LENGTHS:
                    heads = load_phase2_heads(method, model, question, tokens, top_n=50)
                    all_heads[(method, question)].update(heads)
    
    # Compute overlap matrix between questions (aggregated across methods)
    overlap_matrix = np.zeros((len(QUESTIONS), len(QUESTIONS)))
    
    for i, q1 in enumerate(QUESTIONS):
        for j, q2 in enumerate(QUESTIONS):
            # Aggregate heads across methods
            q1_heads = set()
            q2_heads = set()
            for method in METHODS:
                q1_heads.update(all_heads[(method, q1)])
                q2_heads.update(all_heads[(method, q2)])
            
            if q1_heads and q2_heads:
                intersection = len(q1_heads & q2_heads)
                union = len(q1_heads | q2_heads)
                overlap_matrix[i, j] = intersection / union if union > 0 else 0
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(overlap_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=[QUESTION_LABELS[q] for q in QUESTIONS],
                yticklabels=[QUESTION_LABELS[q] for q in QUESTIONS],
                ax=ax, vmin=0, vmax=1)
    
    ax.set_title("Q2: Head Overlap Between Question Types\n(Jaccard Similarity of Top 50 Heads)",
                fontsize=14, fontweight='bold')
    
    # Add category annotations
    ax.axhline(y=2, color='black', linewidth=2)
    ax.axvline(x=2, color='black', linewidth=2)
    ax.text(-0.5, 0.5, 'Categorical', fontsize=10, fontweight='bold', rotation=90, va='center')
    ax.text(-0.5, 2.5, 'Numerical', fontsize=10, fontweight='bold', rotation=90, va='center')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "q2_head_overlap_questions.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Also create a numerical vs categorical comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute within-category and between-category overlaps
    within_numerical = []
    within_categorical = []
    between_category = []
    
    for method in METHODS:
        # Within numerical
        num_heads_list = [all_heads[(method, q)] for q in NUMERICAL_QUESTIONS]
        if all(num_heads_list) and len(num_heads_list) >= 2:
            inter = len(num_heads_list[0] & num_heads_list[1])
            union = len(num_heads_list[0] | num_heads_list[1])
            within_numerical.append(inter / union if union > 0 else 0)
        
        # Within categorical
        cat_heads_list = [all_heads[(method, q)] for q in CATEGORICAL_QUESTIONS]
        if all(cat_heads_list) and len(cat_heads_list) >= 2:
            inter = len(cat_heads_list[0] & cat_heads_list[1])
            union = len(cat_heads_list[0] | cat_heads_list[1])
            within_categorical.append(inter / union if union > 0 else 0)
        
        # Between categories
        for nq in NUMERICAL_QUESTIONS:
            for cq in CATEGORICAL_QUESTIONS:
                nh = all_heads[(method, nq)]
                ch = all_heads[(method, cq)]
                if nh and ch:
                    inter = len(nh & ch)
                    union = len(nh | ch)
                    between_category.append(inter / union if union > 0 else 0)
    
    # Bar plot
    categories = ['Within\nNumerical', 'Within\nCategorical', 'Between\nCategories']
    means = [np.mean(within_numerical), np.mean(within_categorical), np.mean(between_category)]
    stds = [np.std(within_numerical), np.std(within_categorical), np.std(between_category)]
    
    bars = ax.bar(categories, means, yerr=stds, capsize=5, 
                  color=['#ff7f0e', '#1f77b4', '#2ca02c'], alpha=0.7)
    
    ax.set_ylabel("Head Overlap (Jaccard Similarity)")
    ax.set_title("Q2: Are Numerical vs Categorical Heads Similar?", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "q2_numerical_vs_categorical.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def question3_ablation_effectiveness(results):
    """
    Q3: Are ablations equally effective on each category?
    
    Compare effectiveness (accuracy drop) between numerical and categorical questions.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Collect drops by category
    numerical_drops = defaultdict(list)
    categorical_drops = defaultdict(list)
    
    for method in METHODS:
        for model in MODELS:
            for tokens in TOKEN_LENGTHS:
                for question in NUMERICAL_QUESTIONS:
                    result = results.get(method, {}).get(model, {}).get(question, {}).get(tokens)
                    if result is None:
                        continue
                    baseline = result["baseline"]["accuracy"]
                    for abl in result.get("top_heads_ablations", []):
                        drop = baseline - abl["accuracy"]
                        numerical_drops[abl["num_heads"]].append(drop)
                
                for question in CATEGORICAL_QUESTIONS:
                    result = results.get(method, {}).get(model, {}).get(question, {}).get(tokens)
                    if result is None:
                        continue
                    baseline = result["baseline"]["accuracy"]
                    for abl in result.get("top_heads_ablations", []):
                        drop = baseline - abl["accuracy"]
                        categorical_drops[abl["num_heads"]].append(drop)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(ABLATION_LEVELS))
    width = 0.35
    
    num_means = [np.mean(numerical_drops[l]) for l in ABLATION_LEVELS]
    num_stds = [np.std(numerical_drops[l]) for l in ABLATION_LEVELS]
    cat_means = [np.mean(categorical_drops[l]) for l in ABLATION_LEVELS]
    cat_stds = [np.std(categorical_drops[l]) for l in ABLATION_LEVELS]
    
    ax.bar(x - width/2, num_means, width, label='Numerical (year, employee)', 
           yerr=num_stds, capsize=3, color='#ff7f0e', alpha=0.7)
    ax.bar(x + width/2, cat_means, width, label='Categorical (state)', 
           yerr=cat_stds, capsize=3, color='#1f77b4', alpha=0.7)
    
    ax.set_xlabel("Number of Ablated Heads")
    ax.set_ylabel("Accuracy Drop")
    ax.set_title("Q3: Are Ablations Equally Effective on Numerical vs Categorical Questions?",
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ABLATION_LEVELS)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "q3_ablation_effectiveness.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def question4_cross_question_patterns():
    """
    Q4: Patterns within numerical vs categorical questions?
    
    Analyze which specific heads are shared.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Collect top 20 heads for each question, aggregated across methods and configs
    top_heads_by_question = defaultdict(lambda: defaultdict(int))
    
    for method in METHODS:
        for model in MODELS:
            for question in QUESTIONS:
                for tokens in TOKEN_LENGTHS:
                    heads = load_phase2_heads(method, model, question, tokens, top_n=20)
                    for head in heads:
                        top_heads_by_question[question][head] += 1
    
    # Find most common heads for each question
    most_common = {}
    for question in QUESTIONS:
        sorted_heads = sorted(top_heads_by_question[question].items(), 
                             key=lambda x: x[1], reverse=True)[:10]
        most_common[question] = sorted_heads
    
    # Create visualization: head frequency heatmap by layer
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for idx, question in enumerate(QUESTIONS):
        ax = axes[idx // 2, idx % 2]
        
        # Create layer-head frequency matrix
        layer_head_freq = np.zeros((32, 32))  # 32 layers, 32 heads for Llama 3 8B
        
        for head, count in top_heads_by_question[question].items():
            if head[0] < 32 and head[1] < 32:
                layer_head_freq[head[0], head[1]] = count
        
        im = ax.imshow(layer_head_freq, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer Index")
        ax.set_title(f"{QUESTION_LABELS[question]}", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Frequency')
    
    plt.suptitle("Q4: Head Frequency by Layer and Position\n(Which heads are important for each question?)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "q4_head_patterns.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def method_comparison():
    """Compare which heads each method identifies - head overlap between methods."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Collect heads by method (aggregated across all questions/models/tokens)
    method_heads = defaultdict(set)
    
    for method in METHODS:
        for model in MODELS:
            for question in QUESTIONS:
                for tokens in TOKEN_LENGTHS:
                    heads = load_phase2_heads(method, model, question, tokens, top_n=50)
                    method_heads[method].update(heads)
    
    # Compute overlap matrix
    overlap_matrix = np.zeros((len(METHODS), len(METHODS)))
    
    for i, m1 in enumerate(METHODS):
        for j, m2 in enumerate(METHODS):
            h1 = method_heads[m1]
            h2 = method_heads[m2]
            if h1 and h2:
                intersection = len(h1 & h2)
                union = len(h1 | h2)
                overlap_matrix[i, j] = intersection / union if union > 0 else 0
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(overlap_matrix, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[METHOD_LABELS[m] for m in METHODS],
                yticklabels=[METHOD_LABELS[m] for m in METHODS],
                ax=ax, vmin=0, vmax=1)
    
    ax.set_title("Method Comparison: Head Overlap Between Detection Methods\n(Jaccard Similarity)",
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "method_head_overlap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Also show unique heads per method
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute unique and shared
    all_methods_intersection = method_heads[METHODS[0]]
    for m in METHODS[1:]:
        all_methods_intersection &= method_heads[m]
    
    unique_counts = []
    total_counts = []
    shared_with_all = len(all_methods_intersection)
    
    for method in METHODS:
        total_counts.append(len(method_heads[method]))
        others = set()
        for m in METHODS:
            if m != method:
                others.update(method_heads[m])
        unique = method_heads[method] - others
        unique_counts.append(len(unique))
    
    x = np.arange(len(METHODS))
    width = 0.35
    
    ax.bar(x - width/2, total_counts, width, label='Total Heads', color='#1f77b4', alpha=0.7)
    ax.bar(x + width/2, unique_counts, width, label='Unique to Method', color='#ff7f0e', alpha=0.7)
    ax.axhline(y=shared_with_all, color='green', linestyle='--', 
               label=f'Shared by All ({shared_with_all})', linewidth=2)
    
    ax.set_xlabel("Method")
    ax.set_ylabel("Number of Heads")
    ax.set_title("Method Comparison: Total vs Unique Heads Identified", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "method_unique_heads.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def model_comparison(results):
    """Compare Instruct vs Base model ablation effects."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Collect drops by model
    instruct_drops = defaultdict(list)
    base_drops = defaultdict(list)
    
    for method in METHODS:
        for question in QUESTIONS:
            for tokens in TOKEN_LENGTHS:
                # Instruct
                result = results.get(method, {}).get("llama3_instruct", {}).get(question, {}).get(tokens)
                if result:
                    baseline = result["baseline"]["accuracy"]
                    for abl in result.get("top_heads_ablations", []):
                        instruct_drops[abl["num_heads"]].append(baseline - abl["accuracy"])
                
                # Base
                result = results.get(method, {}).get("llama3_base", {}).get(question, {}).get(tokens)
                if result:
                    baseline = result["baseline"]["accuracy"]
                    for abl in result.get("top_heads_ablations", []):
                        base_drops[abl["num_heads"]].append(baseline - abl["accuracy"])
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(ABLATION_LEVELS))
    width = 0.35
    
    instruct_means = [np.mean(instruct_drops[l]) for l in ABLATION_LEVELS]
    instruct_stds = [np.std(instruct_drops[l]) for l in ABLATION_LEVELS]
    base_means = [np.mean(base_drops[l]) for l in ABLATION_LEVELS]
    base_stds = [np.std(base_drops[l]) for l in ABLATION_LEVELS]
    
    ax.bar(x - width/2, instruct_means, width, label='Llama-3-8B-Instruct',
           yerr=instruct_stds, capsize=3, color='#1f77b4', alpha=0.7)
    ax.bar(x + width/2, base_means, width, label='Llama-3-8B-Base',
           yerr=base_stds, capsize=3, color='#ff7f0e', alpha=0.7)
    
    ax.set_xlabel("Number of Ablated Heads")
    ax.set_ylabel("Accuracy Drop")
    ax.set_title("Model Comparison: Instruct vs Base Ablation Effects", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ABLATION_LEVELS)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def token_length_effect(results):
    """Analyze how token length affects ablation effectiveness."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    for idx, question in enumerate(QUESTIONS):
        ax = axes[idx]
        
        for tokens in TOKEN_LENGTHS:
            drops = []
            levels_with_data = []
            
            for method in METHODS:
                for model in MODELS:
                    result = results.get(method, {}).get(model, {}).get(question, {}).get(tokens)
                    if result is None:
                        continue
                    baseline = result["baseline"]["accuracy"]
                    for abl in result.get("top_heads_ablations", []):
                        if len(drops) < len(ABLATION_LEVELS):
                            drops.append([])
                        idx_level = ABLATION_LEVELS.index(abl["num_heads"])
                        if idx_level < len(drops):
                            drops[idx_level].append(baseline - abl["accuracy"])
            
            if drops:
                means = [np.mean(d) if d else 0 for d in drops]
                ax.plot(ABLATION_LEVELS[:len(means)], means, marker='o', 
                       label=f"{tokens} tokens", linewidth=2)
        
        ax.set_xlabel("# Ablated Heads")
        ax.set_ylabel("Accuracy Drop")
        ax.set_title(QUESTION_LABELS[question], fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Token Length Effect on Ablation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "token_length_effect.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Phase 4: Research Questions Analysis")
    print("=" * 60)
    
    # Load results
    print("\nLoading Phase 3 results...")
    results = load_phase3_results()
    
    # Generate analyses
    print("\nQ1: Equal impact analysis...")
    question1_equal_impact(results)
    
    print("\nQ2: Numerical vs categorical heads...")
    question2_numerical_vs_categorical_heads()
    
    print("\nQ3: Ablation effectiveness by category...")
    question3_ablation_effectiveness(results)
    
    print("\nQ4: Cross-question patterns...")
    question4_cross_question_patterns()
    
    print("\nMethod comparison...")
    method_comparison()
    
    print("\nModel comparison...")
    model_comparison(results)
    
    print("\nToken length effect...")
    token_length_effect(results)
    
    print("\n" + "=" * 60)
    print("Research questions analysis complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

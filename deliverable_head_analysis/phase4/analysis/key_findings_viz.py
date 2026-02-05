#!/usr/bin/env python3
"""Visualizations for the most shocking findings"""

import json, os, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

PHASE2 = "/home/ubuntu/iahdClean/deliverable_head_analysis/phase2"
OUTPUT = "/home/ubuntu/iahdClean/deliverable_head_analysis/phase4/exploration_figures"

METHODS = ["summed_attention", "retrieval_head_wu24", "qrhead"]
MLABELS = {"summed_attention": "Summed Attn", "retrieval_head_wu24": "Wu24", "qrhead": "QRHead"}
MODELS = ["llama3_instruct", "llama3_base"]
QUESTIONS = ["inc_state", "inc_year", "employee_count", "hq_state"]
QLABELS = {"inc_state": "Inc.State", "inc_year": "Inc.Year", "employee_count": "Emp.Count", "hq_state": "HQ.State"}
TOKENS = [2048, 4096, 6144, 8192]

def load_full(method, model, q, tok):
    fp = f"{PHASE2}/{method}/results/{model}/{q}/tokens_{tok}.json"
    if not os.path.exists(fp): return None
    with open(fp) as f: return json.load(f)

def get_head_ranks(data):
    if not data: return {}
    return {item['head']: item['rank'] for item in data.get('head_rankings', [])}

# Load all data
all_data = {}
for m in METHODS:
    for mo in MODELS:
        for q in QUESTIONS:
            for t in TOKENS:
                all_data[(m, mo, q, t)] = load_full(m, mo, q, t)

# ============================================================
# VIZ 1: Method Agreement - Only 2 heads in top-100 for ALL methods!
# ============================================================
print("Creating method agreement visualization...")

head_method_ranks = defaultdict(lambda: {m: [] for m in METHODS})
for m in METHODS:
    for mo in MODELS:
        for q in QUESTIONS:
            for t in TOKENS:
                data = all_data[(m, mo, q, t)]
                if data:
                    ranks = get_head_ranks(data)
                    for h, r in ranks.items():
                        head_method_ranks[h][m].append(r)

# Compute average rank per method
head_avg_ranks = {}
for h, method_ranks in head_method_ranks.items():
    avgs = {m: np.mean(method_ranks[m]) for m in METHODS if method_ranks[m]}
    if len(avgs) == 3:
        head_avg_ranks[h] = avgs

# Find heads with biggest disagreements
disagreements = []
for h, avgs in head_avg_ranks.items():
    min_rank = min(avgs.values())
    max_rank = max(avgs.values())
    disagreements.append((h, avgs, max_rank - min_rank))

disagreements.sort(key=lambda x: x[2], reverse=True)
top_disagree = disagreements[:15]

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(top_disagree))
width = 0.25
colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

for i, m in enumerate(METHODS):
    ranks = [d[1][m] for d in top_disagree]
    bars = ax.bar(x + i*width, ranks, width, label=MLABELS[m], color=colors[i], alpha=0.8)

ax.set_xlabel("Head", fontsize=12)
ax.set_ylabel("Average Rank (lower = more important)", fontsize=12)
ax.set_title("Method Disagreement on Head Importance\nSame heads ranked completely differently by different methods", 
             fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([d[0] for d in top_disagree], rotation=45, ha='right')
ax.legend()
ax.set_yscale('log')
ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Top-100 threshold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/method_disagreement.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# VIZ 2: Question Specialists - heads that only work for specific questions
# ============================================================
print("Creating question specialist visualization...")

# Find top specialists for Summed Attention
head_q_ranks = defaultdict(lambda: defaultdict(list))
for mo in MODELS:
    for q in QUESTIONS:
        for t in TOKENS:
            data = all_data[("summed_attention", mo, q, t)]
            if data:
                ranks = get_head_ranks(data)
                for h, r in ranks.items():
                    head_q_ranks[h][q].append(r)

specialists = [
    ("L16H9", "hq_state"),
    ("L17H23", "hq_state"),
    ("L16H11", "hq_state"),
    ("L23H25", "inc_year"),
    ("L22H13", "employee_count"),
]

fig, axes = plt.subplots(1, len(specialists), figsize=(18, 5))
for idx, (head, best_q) in enumerate(specialists):
    ax = axes[idx]
    q_avgs = {q: np.mean(head_q_ranks[head][q]) for q in QUESTIONS}
    colors = ['green' if q == best_q else 'gray' for q in QUESTIONS]
    bars = ax.bar([QLABELS[q] for q in QUESTIONS], [q_avgs[q] for q in QUESTIONS], color=colors)
    ax.set_ylabel("Avg Rank")
    ax.set_title(f"{head}\nSpecialist for {QLABELS[best_q]}", fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    ax.invert_yaxis()  # Lower rank = better, so invert

plt.suptitle("Question Specialists: Heads Important for ONE Question Only (Summed Attention)", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/question_specialists.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# VIZ 3: Instruct vs Base - dramatic differences
# ============================================================
print("Creating Instruct vs Base visualization...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

dramatic_heads = {
    "summed_attention": [("L10H19", 814, 176), ("L10H26", 719, 278), ("L8H27", 612, 232)],
    "retrieval_head_wu24": [("L15H14", 198, 84), ("L14H28", 187, 94), ("L10H2", 174, 84)],
    "qrhead": [("L15H19", 727, 226), ("L13H31", 683, 240), ("L9H4", 733, 370)]
}

for mi, m in enumerate(METHODS):
    ax = axes[mi]
    heads = dramatic_heads[m]
    x = np.arange(len(heads))
    width = 0.35
    
    instruct_ranks = [h[1] for h in heads]
    base_ranks = [h[2] for h in heads]
    
    bars1 = ax.bar(x - width/2, instruct_ranks, width, label='Instruct', color='coral')
    bars2 = ax.bar(x + width/2, base_ranks, width, label='Base', color='steelblue')
    
    ax.set_ylabel("Average Rank")
    ax.set_title(f"{MLABELS[m]}", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([h[0] for h in heads])
    ax.legend()
    
    # Add difference annotations
    for i, (h, ir, br) in enumerate(heads):
        diff = ir - br
        y_pos = max(ir, br) + 30
        ax.annotate(f'Δ={diff:+d}', xy=(i, y_pos), ha='center', fontsize=9, fontweight='bold',
                   color='red' if diff > 0 else 'green')

plt.suptitle("Instruct vs Base: Heads with Dramatic Rank Differences\n(Same head, different importance depending on instruction tuning)", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/instruct_vs_base_differences.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# VIZ 4: Only 2 heads agree across ALL methods!
# ============================================================
print("Creating consensus heads visualization...")

# Find heads in top-100 for all methods
consensus = []
for h, avgs in head_avg_ranks.items():
    if all(v < 100 for v in avgs.values()):
        consensus.append((h, avgs))

fig, ax = plt.subplots(figsize=(12, 7))

# Show consensus heads with their ranks for ALL 3 methods
if consensus:
    heads = [c[0] for c in consensus]
    x = np.arange(len(heads))
    width = 0.25
    
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    for i, m in enumerate(METHODS):
        ranks = [c[1][m] for c in consensus]
        bars = ax.bar(x + i*width, ranks, width, label=MLABELS[m], color=colors[i], alpha=0.8)
        # Add value labels on bars
        for bar, rank in zip(bars, ranks):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                   f'{rank:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(heads, fontsize=14, fontweight='bold')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Top-100 threshold')
    ax.set_ylabel("Average Rank (lower = more important)", fontsize=12)
    ax.set_xlabel("Head", fontsize=12)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 120)

# Add text annotation
ax.text(0.5, 0.95, f"Out of 1024 heads, only {len(consensus)} are in top-100 for ALL 3 methods",
        transform=ax.transAxes, ha='center', va='top', fontsize=12, 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.set_title(f"Consensus Retrieval Heads: Top-100 Across All Methods\n(Summed Attention, Wu24, and QRHead)", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/consensus_heads.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# VIZ 5: Head Volatility - some heads swing wildly
# ============================================================
print("Creating volatility visualization...")

# Calculate volatility for each method
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for mi, m in enumerate(METHODS):
    ax = axes[mi]
    head_ranks_list = defaultdict(list)
    for mo in MODELS:
        for q in QUESTIONS:
            for t in TOKENS:
                data = all_data[(m, mo, q, t)]
                if data:
                    ranks = get_head_ranks(data)
                    for h, r in ranks.items():
                        head_ranks_list[h].append(r)
    
    # Top 5 most volatile
    volatility = [(h, np.std(ranks), np.mean(ranks), min(ranks), max(ranks)) 
                  for h, ranks in head_ranks_list.items() if len(ranks) >= 16]
    volatility.sort(key=lambda x: x[1], reverse=True)
    top5 = volatility[:5]
    
    for i, (h, std, mean, min_r, max_r) in enumerate(top5):
        ax.barh(i, max_r - min_r, left=min_r, height=0.6, color=plt.cm.Reds(0.3 + i*0.14))
        ax.scatter([mean], [i], color='black', s=100, zorder=5, marker='|')
        ax.annotate(f'{h}\nstd={std:.0f}', xy=(max_r + 20, i), va='center', fontsize=9)
    
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"#{i+1}" for i in range(5)])
    ax.set_xlabel("Rank Range (min to max)")
    ax.set_title(f"{MLABELS[m]}", fontweight='bold')
    ax.set_xlim(0, 1100)

plt.suptitle("Most Volatile Heads: Rank Swings from Best to Worst Across Configs", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/head_volatility.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# VIZ 6: Wu24 has EXTREME dominance of top heads
# ============================================================
print("Creating score dominance visualization...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for mi, m in enumerate(METHODS):
    ax = axes[mi]
    gaps = []
    for mo in MODELS:
        for q in QUESTIONS:
            for t in TOKENS:
                data = all_data[(m, mo, q, t)]
                if data and len(data.get('head_rankings', [])) >= 2:
                    r = data['head_rankings']
                    gap_pct = (r[0]['score'] - r[1]['score']) / r[0]['score'] * 100
                    gaps.append(gap_pct)
    
    ax.hist(gaps, bins=20, color=plt.cm.tab10(mi), alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(gaps), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel("Gap between #1 and #2 (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{MLABELS[m]}\nMean gap: {np.mean(gaps):.1f}%", fontweight='bold')

plt.suptitle("Score Gap Between #1 and #2 Ranked Heads\n(Wu24 has MUCH more dominant top heads!)", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/score_dominance.png", dpi=150, bbox_inches='tight')
plt.close()

print("\nDone! Created 6 finding visualizations.")

#!/usr/bin/env python3
"""Deep dive into Phase 2 data - finding surprising patterns"""

import json, os, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy import stats
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

def get_head_scores(data):
    """Return dict of head -> score"""
    if not data: return {}
    return {item['head']: item['score'] for item in data.get('head_rankings', [])}

def get_head_ranks(data):
    """Return dict of head -> rank"""
    if not data: return {}
    return {item['head']: item['rank'] for item in data.get('head_rankings', [])}

print("="*70)
print("DEEP DIVE: Phase 2 Surprising Findings")
print("="*70)

# Load all data
all_data = {}
for m in METHODS:
    for mo in MODELS:
        for q in QUESTIONS:
            for t in TOKENS:
                all_data[(m, mo, q, t)] = load_full(m, mo, q, t)

# ============================================================
# FINDING 1: Score magnitude differences across methods
# ============================================================
print("\n" + "="*70)
print("FINDING 1: Score Magnitude Differences")
print("="*70)

method_scores = {m: [] for m in METHODS}
for m in METHODS:
    for mo in MODELS:
        for q in QUESTIONS:
            for t in TOKENS:
                data = all_data[(m, mo, q, t)]
                if data:
                    scores = get_head_scores(data)
                    method_scores[m].extend(scores.values())

for m in METHODS:
    scores = method_scores[m]
    print(f"{MLABELS[m]:15s}: min={min(scores):.2f}, max={max(scores):.2f}, mean={np.mean(scores):.2f}, std={np.std(scores):.2f}")

# ============================================================
# FINDING 2: Head Volatility - which heads change ranks most?
# ============================================================
print("\n" + "="*70)
print("FINDING 2: Head Volatility (rank std across configs)")
print("="*70)

for m in METHODS:
    head_ranks_list = defaultdict(list)
    for mo in MODELS:
        for q in QUESTIONS:
            for t in TOKENS:
                data = all_data[(m, mo, q, t)]
                if data:
                    ranks = get_head_ranks(data)
                    for h, r in ranks.items():
                        head_ranks_list[h].append(r)
    
    # Calculate volatility (std of ranks)
    volatility = {h: np.std(ranks) for h, ranks in head_ranks_list.items() if len(ranks) >= 16}
    sorted_vol = sorted(volatility.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{MLABELS[m]} - Most Volatile Heads (high rank std):")
    for h, v in sorted_vol[:5]:
        mean_rank = np.mean(head_ranks_list[h])
        print(f"  {h}: std={v:.1f}, mean_rank={mean_rank:.0f}, range={min(head_ranks_list[h])}-{max(head_ranks_list[h])}")
    
    print(f"\n{MLABELS[m]} - Most Stable Heads (low rank std):")
    for h, v in sorted_vol[-5:]:
        mean_rank = np.mean(head_ranks_list[h])
        print(f"  {h}: std={v:.1f}, mean_rank={mean_rank:.0f}")

# ============================================================
# FINDING 3: Question Specialists - heads that spike for specific questions
# ============================================================
print("\n" + "="*70)
print("FINDING 3: Question Specialists")
print("="*70)

for m in METHODS:
    # For each head, get average rank per question
    head_q_ranks = defaultdict(lambda: defaultdict(list))
    for mo in MODELS:
        for q in QUESTIONS:
            for t in TOKENS:
                data = all_data[(m, mo, q, t)]
                if data:
                    ranks = get_head_ranks(data)
                    for h, r in ranks.items():
                        head_q_ranks[h][q].append(r)
    
    # Find specialists: low rank for one question, high for others
    specialists = []
    for h, q_ranks in head_q_ranks.items():
        if len(q_ranks) < 4: continue
        avg_ranks = {q: np.mean(r) for q, r in q_ranks.items()}
        best_q = min(avg_ranks, key=avg_ranks.get)
        best_rank = avg_ranks[best_q]
        other_ranks = [r for q, r in avg_ranks.items() if q != best_q]
        avg_other = np.mean(other_ranks)
        
        if best_rank < 50 and avg_other > 200:  # Top-50 for one, >200 for others
            specialists.append((h, best_q, best_rank, avg_other))
    
    specialists.sort(key=lambda x: x[2])
    print(f"\n{MLABELS[m]} - Question Specialists (top-50 for one, rank>200 for others):")
    for h, best_q, best_rank, avg_other in specialists[:5]:
        print(f"  {h}: best for {QLABELS[best_q]} (rank {best_rank:.0f}), others avg rank {avg_other:.0f}")

# ============================================================
# FINDING 4: Instruct vs Base - biggest differences
# ============================================================
print("\n" + "="*70)
print("FINDING 4: Instruct vs Base - Heads with Biggest Rank Differences")
print("="*70)

for m in METHODS:
    head_model_ranks = defaultdict(lambda: {"instruct": [], "base": []})
    for q in QUESTIONS:
        for t in TOKENS:
            data_i = all_data[(m, "llama3_instruct", q, t)]
            data_b = all_data[(m, "llama3_base", q, t)]
            if data_i and data_b:
                ranks_i = get_head_ranks(data_i)
                ranks_b = get_head_ranks(data_b)
                for h in ranks_i:
                    if h in ranks_b:
                        head_model_ranks[h]["instruct"].append(ranks_i[h])
                        head_model_ranks[h]["base"].append(ranks_b[h])
    
    # Find heads with biggest differences
    diffs = []
    for h, ranks in head_model_ranks.items():
        if len(ranks["instruct"]) >= 8:
            avg_i = np.mean(ranks["instruct"])
            avg_b = np.mean(ranks["base"])
            diff = avg_i - avg_b  # negative = better in instruct
            diffs.append((h, avg_i, avg_b, diff))
    
    diffs.sort(key=lambda x: x[3])
    print(f"\n{MLABELS[m]} - Heads MUCH better in Instruct:")
    for h, avg_i, avg_b, diff in diffs[:3]:
        print(f"  {h}: Instruct rank {avg_i:.0f}, Base rank {avg_b:.0f} (diff={diff:.0f})")
    
    print(f"\n{MLABELS[m]} - Heads MUCH better in Base:")
    for h, avg_i, avg_b, diff in diffs[-3:]:
        print(f"  {h}: Instruct rank {avg_i:.0f}, Base rank {avg_b:.0f} (diff={diff:.0f})")

# ============================================================
# FINDING 5: Token length sensitivity
# ============================================================
print("\n" + "="*70)
print("FINDING 5: Token Length Sensitivity - Heads that change with context")
print("="*70)

for m in METHODS:
    head_tok_ranks = defaultdict(lambda: {t: [] for t in TOKENS})
    for mo in MODELS:
        for q in QUESTIONS:
            for t in TOKENS:
                data = all_data[(m, mo, q, t)]
                if data:
                    ranks = get_head_ranks(data)
                    for h, r in ranks.items():
                        head_tok_ranks[h][t].append(r)
    
    # Find heads that improve or degrade with longer context
    trends = []
    for h, tok_ranks in head_tok_ranks.items():
        avgs = [np.mean(tok_ranks[t]) if tok_ranks[t] else None for t in TOKENS]
        if None in avgs: continue
        # Check correlation with token length
        corr, _ = stats.pearsonr(TOKENS, avgs)
        trends.append((h, corr, avgs))
    
    trends.sort(key=lambda x: x[1])
    print(f"\n{MLABELS[m]} - Heads that get WORSE with longer context (rank increases):")
    for h, corr, avgs in trends[-3:]:
        print(f"  {h}: corr={corr:.2f}, ranks @ 2K/4K/6K/8K = {avgs[0]:.0f}/{avgs[1]:.0f}/{avgs[2]:.0f}/{avgs[3]:.0f}")
    
    print(f"\n{MLABELS[m]} - Heads that get BETTER with longer context (rank decreases):")
    for h, corr, avgs in trends[:3]:
        print(f"  {h}: corr={corr:.2f}, ranks @ 2K/4K/6K/8K = {avgs[0]:.0f}/{avgs[1]:.0f}/{avgs[2]:.0f}/{avgs[3]:.0f}")

# ============================================================
# FINDING 6: Cross-method agreement on specific heads
# ============================================================
print("\n" + "="*70)
print("FINDING 6: Cross-Method Agreement - Heads ranked similarly by all methods")
print("="*70)

# For each head, get average rank per method
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

# Find heads where all methods agree it's important
agreements = []
for h, method_ranks in head_method_ranks.items():
    avgs = {m: np.mean(method_ranks[m]) for m in METHODS if method_ranks[m]}
    if len(avgs) == 3:
        all_top100 = all(v < 100 for v in avgs.values())
        if all_top100:
            max_rank = max(avgs.values())
            agreements.append((h, avgs, max_rank))

agreements.sort(key=lambda x: x[2])
print("\nHeads in TOP-100 for ALL three methods:")
for h, avgs, _ in agreements[:10]:
    print(f"  {h}: SummedAttn={avgs['summed_attention']:.0f}, Wu24={avgs['retrieval_head_wu24']:.0f}, QRHead={avgs['qrhead']:.0f}")

# Find heads with huge disagreement
disagreements = []
for h, method_ranks in head_method_ranks.items():
    avgs = {m: np.mean(method_ranks[m]) for m in METHODS if method_ranks[m]}
    if len(avgs) == 3:
        min_rank = min(avgs.values())
        max_rank = max(avgs.values())
        if min_rank < 20 and max_rank > 500:  # Top-20 in one, >500 in another
            disagreements.append((h, avgs, max_rank - min_rank))

disagreements.sort(key=lambda x: x[2], reverse=True)
print("\nHeads with HUGE disagreement (top-20 in one method, >500 in another):")
for h, avgs, diff in disagreements[:10]:
    print(f"  {h}: SummedAttn={avgs['summed_attention']:.0f}, Wu24={avgs['retrieval_head_wu24']:.0f}, QRHead={avgs['qrhead']:.0f}")

# ============================================================
# FINDING 7: Score distribution shape
# ============================================================
print("\n" + "="*70)
print("FINDING 7: Top head dominance (score gap between #1 and #2)")
print("="*70)

for m in METHODS:
    gaps = []
    for mo in MODELS:
        for q in QUESTIONS:
            for t in TOKENS:
                data = all_data[(m, mo, q, t)]
                if data and len(data.get('head_rankings', [])) >= 2:
                    r = data['head_rankings']
                    gap = r[0]['score'] - r[1]['score']
                    pct = gap / r[0]['score'] * 100
                    gaps.append((mo, q, t, r[0]['head'], gap, pct))
    
    gaps.sort(key=lambda x: x[5], reverse=True)
    print(f"\n{MLABELS[m]} - Largest #1 vs #2 gaps:")
    for mo, q, t, h1, gap, pct in gaps[:3]:
        print(f"  {mo.split('_')[1]}/{QLABELS[q]}/{t}: {h1} leads by {pct:.1f}%")

# ============================================================
# FINDING 8: Layer patterns - which layers dominate?
# ============================================================
print("\n" + "="*70)
print("FINDING 8: Layer Dominance in Top-10")
print("="*70)

for m in METHODS:
    layer_counts = defaultdict(int)
    total = 0
    for mo in MODELS:
        for q in QUESTIONS:
            for t in TOKENS:
                data = all_data[(m, mo, q, t)]
                if data:
                    for item in data.get('head_rankings', [])[:10]:
                        layer = int(item['head'].split('H')[0][1:])
                        layer_counts[layer] += 1
                        total += 1
    
    # Top layers
    sorted_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{MLABELS[m]} - Most common layers in top-10:")
    for layer, cnt in sorted_layers[:5]:
        print(f"  Layer {layer}: {cnt} appearances ({cnt/total*100:.1f}%)")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

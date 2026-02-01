"""
Clean visualization: Impact of Shuffling on Accuracy
"""
import json
import matplotlib.pyplot as plt
import numpy as np

with open('llama3_results/head/context_attention_sweep.json', 'r') as f:
    data = json.load(f)

# Extract and sort data
results = []
for ctx_len, res in data['by_context_length'].items():
    unshuf = res['unshuffled']['accuracy'] * 100
    shuf = res['shuffled']['accuracy'] * 100
    results.append({
        'ctx': int(ctx_len),
        'drop': unshuf - shuf
    })

results.sort(key=lambda x: x['ctx'])
ctx = [r['ctx'] for r in results]
drop = [r['drop'] for r in results]

# Only show contexts where we have meaningful accuracy (exclude 10K+)
valid_idx = [i for i, c in enumerate(ctx) if c <= 8000]
ctx_valid = [ctx[i] for i in valid_idx]
drop_valid = [drop[i] for i in valid_idx]

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#3b82f6' if d > 0 else '#94a3b8' for d in drop_valid]
bars = ax.bar(range(len(ctx_valid)), drop_valid, color=colors, edgecolor='white', linewidth=1.5)

ax.set_xticks(range(len(ctx_valid)))
ax.set_xticklabels([f'{x//1000}K' if x >= 1000 else str(x) for x in ctx_valid])
ax.set_xlabel('Context Length (tokens)', fontsize=12)
ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
ax.set_title('How Much Does Shuffling Hurt Accuracy?', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, d in zip(bars, drop_valid):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'+{d:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('llama3_results/head/shuffle_impact.png', dpi=150, facecolor='white')
print("Saved: llama3_results/head/shuffle_impact.png")


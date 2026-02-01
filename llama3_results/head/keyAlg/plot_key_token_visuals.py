"""
Visualizations for Key Token Attention Sweep Results
Same style as context attention visualizations
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('../key_token_attention_sweep.json', 'r') as f:
    data = json.load(f)

# ============================================================
# 1. Accuracy vs Context Length
# ============================================================
results = []
for ctx_len, res in data['by_context_length'].items():
    results.append({
        'ctx': int(ctx_len),
        'unshuf': res['unshuffled']['accuracy'] * 100,
        'shuf': res['shuffled']['accuracy'] * 100
    })

results.sort(key=lambda x: x['ctx'])
ctx = [r['ctx'] for r in results]
unshuf = [r['unshuf'] for r in results]
shuf = [r['shuf'] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(ctx, unshuf, 'o-', color='#2563eb', linewidth=2.5, markersize=8, label='Unshuffled')
plt.plot(ctx, shuf, 's--', color='#dc2626', linewidth=2.5, markersize=8, label='Shuffled')

plt.xlabel('Context Length (tokens)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Llama 3 Retrieval Accuracy vs Context Length\n(Key Token Attention Experiment)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(-5, 75)
plt.xscale('log')
plt.xticks(ctx, [f'{x//1000}K' if x >= 1000 else str(x) for x in ctx], rotation=45)

plt.tight_layout()
plt.savefig('accuracy_vs_context.png', dpi=150, facecolor='white')
print("Saved: accuracy_vs_context.png")
plt.close()

# ============================================================
# 2. Shuffle Impact
# ============================================================
context_lengths = [200, 500, 1000, 2000, 4000, 8000]
drop = []
for c in context_lengths:
    res = data['by_context_length'][str(c)]
    unshuf_acc = res['unshuffled']['accuracy'] * 100
    shuf_acc = res['shuffled']['accuracy'] * 100
    drop.append(unshuf_acc - shuf_acc)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#3b82f6' if d > 0 else '#94a3b8' for d in drop]
bars = ax.bar(range(len(context_lengths)), drop, color=colors, edgecolor='white', linewidth=1.5)

ax.set_xticks(range(len(context_lengths)))
ax.set_xticklabels([f'{x//1000}K' if x >= 1000 else str(x) for x in context_lengths])
ax.set_xlabel('Context Length (tokens)', fontsize=12)
ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
ax.set_title('How Much Does Shuffling Hurt Accuracy?\n(Key Token Attention Experiment)', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

for bar, d in zip(bars, drop):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'+{d:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('shuffle_impact.png', dpi=150, facecolor='white')
print("Saved: shuffle_impact.png")
plt.close()

# ============================================================
# 3. Top Heads - Unshuffled
# ============================================================
fig1, ax1 = plt.subplots(figsize=(12, 6))

x_positions = np.arange(len(context_lengths))
width = 0.15
colors = ['#1e40af', '#3b82f6', '#60a5fa', '#93c5fd', '#dbeafe']

for rank in range(5):
    scores = []
    labels = []
    for ctx in context_lengths:
        res = data['by_context_length'][str(ctx)]['unshuffled']
        if res.get('top_heads') and len(res['top_heads']) > rank:
            head_name, score = res['top_heads'][rank]
            scores.append(score)
            labels.append(head_name)
        else:
            scores.append(0)
            labels.append('')
    
    bars = ax1.bar(x_positions + rank * width, scores, width, 
                   label=f'Rank {rank+1}', color=colors[rank], edgecolor='white')
    
    for bar, label in zip(bars, labels):
        if label:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    label, ha='center', va='bottom', fontsize=7, rotation=45)

ax1.set_xticks(x_positions + 2*width)
ax1.set_xticklabels([f'{c//1000}K' if c >= 1000 else str(c) for c in context_lengths])
ax1.set_xlabel('Context Length (tokens)', fontsize=12)
ax1.set_ylabel('Attention Score (to answer tokens)', fontsize=12)
ax1.set_title('UNSHUFFLED: Top 5 Retrieval Heads by Context Length\n(Key Token Attention)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 0.5)

plt.tight_layout()
plt.savefig('top_heads_unshuffled.png', dpi=150, facecolor='white')
print("Saved: top_heads_unshuffled.png")
plt.close()

# ============================================================
# 4. Top Heads - Shuffled
# ============================================================
fig2, ax2 = plt.subplots(figsize=(12, 6))

colors_shuf = ['#991b1b', '#dc2626', '#f87171', '#fca5a5', '#fecaca']

for rank in range(5):
    scores = []
    labels = []
    for ctx in context_lengths:
        res = data['by_context_length'][str(ctx)]['shuffled']
        if res.get('top_heads') and len(res['top_heads']) > rank:
            head_name, score = res['top_heads'][rank]
            scores.append(score)
            labels.append(head_name)
        else:
            scores.append(0)
            labels.append('')
    
    bars = ax2.bar(x_positions + rank * width, scores, width, 
                   label=f'Rank {rank+1}', color=colors_shuf[rank], edgecolor='white')
    
    for bar, label in zip(bars, labels):
        if label:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    label, ha='center', va='bottom', fontsize=7, rotation=45)

ax2.set_xticks(x_positions + 2*width)
ax2.set_xticklabels([f'{c//1000}K' if c >= 1000 else str(c) for c in context_lengths])
ax2.set_xlabel('Context Length (tokens)', fontsize=12)
ax2.set_ylabel('Attention Score (to answer tokens)', fontsize=12)
ax2.set_title('SHUFFLED: Top 5 Retrieval Heads by Context Length\n(Key Token Attention)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 0.5)

plt.tight_layout()
plt.savefig('top_heads_shuffled.png', dpi=150, facecolor='white')
print("Saved: top_heads_shuffled.png")
plt.close()

print("\nAll visualizations created!")


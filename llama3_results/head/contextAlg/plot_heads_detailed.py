"""
Detailed visualization: Top Retrieval Heads - Shuffled vs Unshuffled
"""
import json
import matplotlib.pyplot as plt
import numpy as np

with open('llama3_results/head/context_attention_sweep.json', 'r') as f:
    data = json.load(f)

# Only use contexts where we have attention data (<=8000)
context_lengths = [200, 500, 1000, 2000, 4000, 8000]

# ============================================================
# Figure 1: UNSHUFFLED - Top 5 heads at each context length
# ============================================================
fig1, ax1 = plt.subplots(figsize=(12, 6))

print("UNSHUFFLED TOP 5 HEADS")
print("=" * 70)

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
    
    # Add head labels on ALL bars
    for bar, label in zip(bars, labels):
        if label:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    label, ha='center', va='bottom', fontsize=7, rotation=45)

# Print table
for ctx in context_lengths:
    res = data['by_context_length'][str(ctx)]['unshuffled']
    if res.get('top_heads'):
        heads = [f"{h[0]}({h[1]:.2f})" for h in res['top_heads'][:5]]
        print(f"{ctx:>5}: {', '.join(heads)}")

ax1.set_xticks(x_positions + 2*width)
ax1.set_xticklabels([f'{c//1000}K' if c >= 1000 else str(c) for c in context_lengths])
ax1.set_xlabel('Context Length (tokens)', fontsize=12)
ax1.set_ylabel('Attention Score (sum over Section 1)', fontsize=12)
ax1.set_title('UNSHUFFLED: Top 5 Retrieval Heads by Context Length', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('llama3_results/head/top_heads_unshuffled.png', dpi=150, facecolor='white')
print("\nSaved: llama3_results/head/top_heads_unshuffled.png")

# ============================================================
# Figure 2: SHUFFLED - Top 5 heads at each context length
# ============================================================
fig2, ax2 = plt.subplots(figsize=(12, 6))

print("\n" + "=" * 70)
print("SHUFFLED TOP 5 HEADS")
print("=" * 70)

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
    
    # Add head labels on ALL bars
    for bar, label in zip(bars, labels):
        if label:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    label, ha='center', va='bottom', fontsize=7, rotation=45)

# Print table
for ctx in context_lengths:
    res = data['by_context_length'][str(ctx)]['shuffled']
    if res.get('top_heads'):
        heads = [f"{h[0]}({h[1]:.2f})" for h in res['top_heads'][:5]]
        print(f"{ctx:>5}: {', '.join(heads)}")

ax2.set_xticks(x_positions + 2*width)
ax2.set_xticklabels([f'{c//1000}K' if c >= 1000 else str(c) for c in context_lengths])
ax2.set_xlabel('Context Length (tokens)', fontsize=12)
ax2.set_ylabel('Attention Score (sum over Section 1)', fontsize=12)
ax2.set_title('SHUFFLED: Top 5 Retrieval Heads by Context Length', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('llama3_results/head/top_heads_shuffled.png', dpi=150, facecolor='white')
print("\nSaved: llama3_results/head/top_heads_shuffled.png")

# ============================================================
# Figure 3: Side-by-side comparison table visualization
# ============================================================
fig3, ax3 = plt.subplots(figsize=(14, 8))
ax3.axis('off')

# Build comparison table
table_data = []
headers = ['Context', 'Unshuffled Rank 1', 'Unshuffled Rank 2', 'Unshuffled Rank 3',
           'Shuffled Rank 1', 'Shuffled Rank 2', 'Shuffled Rank 3']

for ctx in context_lengths:
    row = [f'{ctx//1000}K' if ctx >= 1000 else str(ctx)]
    
    # Unshuffled top 3
    unshuf = data['by_context_length'][str(ctx)]['unshuffled']
    if unshuf.get('top_heads'):
        for i in range(3):
            if i < len(unshuf['top_heads']):
                row.append(unshuf['top_heads'][i][0])
            else:
                row.append('-')
    else:
        row.extend(['-', '-', '-'])
    
    # Shuffled top 3
    shuf = data['by_context_length'][str(ctx)]['shuffled']
    if shuf.get('top_heads'):
        for i in range(3):
            if i < len(shuf['top_heads']):
                row.append(shuf['top_heads'][i][0])
            else:
                row.append('-')
    else:
        row.extend(['-', '-', '-'])
    
    table_data.append(row)

table = ax3.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)

# Color the header
for i, key in enumerate(headers):
    table[(0, i)].set_facecolor('#e5e7eb')
    table[(0, i)].set_text_props(fontweight='bold')

# Color unshuffled columns blue-ish
for row in range(1, len(table_data) + 1):
    for col in [1, 2, 3]:
        table[(row, col)].set_facecolor('#dbeafe')

# Color shuffled columns red-ish
for row in range(1, len(table_data) + 1):
    for col in [4, 5, 6]:
        table[(row, col)].set_facecolor('#fee2e2')

ax3.set_title('Top 3 Retrieval Heads: Unshuffled vs Shuffled\n', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('llama3_results/head/heads_comparison_table.png', dpi=150, facecolor='white')
print("\nSaved: llama3_results/head/heads_comparison_table.png")

plt.close('all')


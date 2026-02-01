"""
Compare Context Attention vs Key Token Attention metrics
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load both results
with open('llama3_results/head/context_attention_sweep.json', 'r') as f:
    context = json.load(f)

with open('llama3_results/head/key_token_attention_sweep.json', 'r') as f:
    key_token = json.load(f)

contexts = [200, 500, 1000, 2000, 4000, 8000]
ctx_labels = ['200', '500', '1K', '2K', '4K', '8K']

# ============================================================
# Figure 1: Average Layer by Metric and Condition
# ============================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))

def get_avg_layer(data, ctx, condition):
    heads = data['by_context_length'][str(ctx)][condition].get('top_heads', [])[:5]
    if heads:
        layers = [int(h[0].split('H')[0][1:]) for h in heads]
        return sum(layers) / len(layers)
    return 0

# Get data
ctx_unshuf = [get_avg_layer(context, c, 'unshuffled') for c in contexts]
ctx_shuf = [get_avg_layer(context, c, 'shuffled') for c in contexts]
key_unshuf = [get_avg_layer(key_token, c, 'unshuffled') for c in contexts]
key_shuf = [get_avg_layer(key_token, c, 'shuffled') for c in contexts]

x = np.arange(len(contexts))
width = 0.2

ax1.bar(x - 1.5*width, ctx_unshuf, width, label='Context Attn - Unshuffled', color='#2563eb')
ax1.bar(x - 0.5*width, ctx_shuf, width, label='Context Attn - Shuffled', color='#93c5fd')
ax1.bar(x + 0.5*width, key_unshuf, width, label='Key Token Attn - Unshuffled', color='#dc2626')
ax1.bar(x + 1.5*width, key_shuf, width, label='Key Token Attn - Shuffled', color='#fca5a5')

ax1.set_xticks(x)
ax1.set_xticklabels(ctx_labels)
ax1.set_xlabel('Context Length (tokens)', fontsize=12)
ax1.set_ylabel('Average Layer of Top 5 Heads', fontsize=12)
ax1.set_title('Which Layers Do Retrieval? Context vs Key Token Attention', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=16, color='gray', linestyle='--', alpha=0.5, label='Late layer threshold')
ax1.set_ylim(0, 25)

# Add annotations
ax1.annotate('Context Attn:\nEarly layers (L0)', xy=(0, 5), fontsize=9, color='#2563eb')
ax1.annotate('Key Token Attn:\nAlways late layers', xy=(0, 20), fontsize=9, color='#dc2626')

plt.tight_layout()
plt.savefig('llama3_results/head/metric_layer_comparison.png', dpi=150, facecolor='white')
print('Saved: llama3_results/head/metric_layer_comparison.png')

# ============================================================
# Figure 2: Key Token Attention Scores - Shuffled vs Unshuffled
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 5))

def get_top_score(data, ctx, condition):
    heads = data['by_context_length'][str(ctx)][condition].get('top_heads', [])
    return heads[0][1] if heads else 0

unshuf_scores = [get_top_score(key_token, c, 'unshuffled') for c in contexts]
shuf_scores = [get_top_score(key_token, c, 'shuffled') for c in contexts]

x = np.arange(len(contexts))
width = 0.35

bars1 = ax2.bar(x - width/2, unshuf_scores, width, label='Unshuffled', color='#2563eb')
bars2 = ax2.bar(x + width/2, shuf_scores, width, label='Shuffled', color='#dc2626')

ax2.set_xticks(x)
ax2.set_xticklabels(ctx_labels)
ax2.set_xlabel('Context Length (tokens)', fontsize=12)
ax2.set_ylabel('Attention Score to Answer Tokens', fontsize=12)
ax2.set_title('Key Token Attention: How Much Do Heads Focus on Answer?', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

# Add ratio annotations
for i, (u, s) in enumerate(zip(unshuf_scores, shuf_scores)):
    if s > 0:
        ratio = u / s
        ax2.annotate(f'{ratio:.1f}x', xy=(i, max(u, s) + 0.02), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('llama3_results/head/key_token_scores.png', dpi=150, facecolor='white')
print('Saved: llama3_results/head/key_token_scores.png')

# ============================================================
# Figure 3: Summary Table
# ============================================================
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.axis('off')

summary = '''
METRIC COMPARISON SUMMARY
=========================

                        CONTEXT ATTENTION              KEY TOKEN ATTENTION
                        (sum over Section 1)           (sum over answer tokens)
--------------------------------------------------------------------------------
Window size:            2000-18000 tokens              1-3 tokens (e.g. "Delaware")

UNSHUFFLED:
  - Short contexts:     Layer 0 heads (L0H22, etc)     Late layers (L17-L25)
  - Long contexts:      Late layers (L16-L20)          Late layers (L17-L25)
  
SHUFFLED:
  - Short contexts:     Layer 0 heads + L17H24         Late layers (L17-L25)
  - Long contexts:      Late layers (L16-L27)          Late layers (L17-L27)

KEY INSIGHT:
  - Context Attention favors EARLY layers at short contexts (broad attention)
  - Key Token Attention ALWAYS favors LATE layers (precise retrieval)
  - Shuffling reduces attention STRENGTH by ~50%, but same heads identified

INTERPRETATION:
  - Late layers (L17-L27) are the "copy heads" that extract the answer
  - They work for both shuffled and unshuffled, but less effectively when scrambled
  - This explains accuracy drop: same mechanism, weaker signal
'''

ax3.text(0.05, 0.95, summary, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

plt.tight_layout()
plt.savefig('llama3_results/head/metric_comparison_summary.png', dpi=150, facecolor='white')
print('Saved: llama3_results/head/metric_comparison_summary.png')

plt.close('all')


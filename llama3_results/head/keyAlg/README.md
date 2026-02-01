# Key Token Attention Analysis (Copy Head Style)

## Metric Definition

**Key Token Attention** = Sum of attention from the last token to SPECIFIC ANSWER TOKENS only

```python
# Find where "Delaware" (or other state name) appears in the input
key_attn = 0
for start, end in answer_token_positions:
    key_attn += last_pos_attn[start:end].sum()
```

This measures: **"Which heads are COPYING the specific answer?"**

## Key Results

### Accuracy vs Context Length

Same accuracy as context attention (same task, different metric for head identification):

| Context | Unshuffled | Shuffled | Drop |
|---------|------------|----------|------|
| 200 | 63.3% | 43.3% | +20% |
| 500 | 63.3% | 50.0% | +13% |
| 1000 | 63.3% | 50.0% | +13% |
| 2000 | 60.0% | 40.0% | +20% |
| 4000 | 53.3% | 40.0% | +13% |
| 8000 | 66.7% | 33.3% | +33% |
| 10K+ | 0% | ~3% | — |

### Top Retrieval Heads

**Unshuffled:**
- ALL contexts: **L17H24, L24H27, L20H1, L20H14, L25H15** (Late layers L17-L25)

**Shuffled:**
- ALL contexts: **L20H14, L17H24, L25H15, L20H1** (Same late layers L17-L27!)

### Attention Scores (Critical Finding!)

| Context | Unshuffled Score | Shuffled Score | Ratio |
|---------|-----------------|----------------|-------|
| 200 | 0.42 | 0.21 | 2.0x |
| 500 | 0.42 | 0.22 | 1.9x |
| 1000 | 0.41 | 0.20 | 2.1x |
| 2000 | 0.39 | 0.23 | 1.7x |
| 4000 | 0.32 | 0.16 | 2.0x |
| 8000 | 0.24 | 0.12 | 2.0x |

**Shuffling cuts attention strength in half!**

### Interpretation

- **Late layers (L17-L27) are "copy heads"** - they extract the specific answer
- **Same heads identified for both shuffled and unshuffled**
- The difference is **attention strength**, not which heads
- Shuffling makes the model attend ~50% less strongly to answer tokens
- This explains the accuracy drop: same mechanism, weaker signal

## Files

| File | Description |
|------|-------------|
| `accuracy_vs_context.png` | Line plot: accuracy vs context length |
| `shuffle_impact.png` | Bar chart: accuracy drop from shuffling |
| `top_heads_unshuffled.png` | Top 5 heads for unshuffled condition |
| `top_heads_shuffled.png` | Top 5 heads for shuffled condition |
| `key_token_attention_sweep.json` | Raw results data |
| `llama3_key_token_attention_sweep.py` | Main experiment script |
| `plot_key_token_visuals.py` | Visualization script |


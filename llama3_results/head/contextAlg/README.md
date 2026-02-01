# Context Attention Analysis

## Metric Definition

**Context Attention** = Sum of attention from the last token to the ENTIRE Section 1 region

```python
context_attn = last_pos_attn[section_start:section_end].sum()
```

This measures: **"Which heads are LOCATING/ATTENDING to the relevant region?"**

## Key Results

### Accuracy vs Context Length

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
- Short contexts (200-2K): **L0H22, L0H28, L0H21, L0H30** (Layer 0 dominates)
- Long contexts (4K-8K): **L17H24, L16H1, L18H16** (Late layers take over)

**Shuffled:**
- **L17H24** jumps to #1 even at short contexts
- Still uses L0 heads but less prominently

### Interpretation

- **Early layers (L0)** handle broad positional attention over large regions
- **Late layers (L16-20)** handle focused retrieval at longer contexts
- Shuffling disrupts sentence structure, affecting how the model finds information

## Files

| File | Description |
|------|-------------|
| `accuracy_vs_context.png` | Line plot: accuracy vs context length |
| `shuffle_impact.png` | Bar chart: accuracy drop from shuffling |
| `top_heads_unshuffled.png` | Top 5 heads for unshuffled condition |
| `top_heads_shuffled.png` | Top 5 heads for shuffled condition |
| `context_attention_sweep.json` | Raw results data |
| `llama3_context_attention_sweep.py` | Main experiment script |
| `plot_*.py` | Visualization scripts |


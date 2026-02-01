# Context Attention vs Key Token Attention Comparison

## Overview

This folder contains comparisons between two different metrics for identifying retrieval heads in Llama 3:

1. **Context Attention** (Harry's approach): Sum of attention over the ENTIRE Section 1 region (~2000-18000 tokens)
2. **Key Token Attention** (Paper's copy head approach): Sum of attention to SPECIFIC ANSWER TOKENS only (1-3 tokens, e.g., "Delaware")

## Key Findings

### Different Metrics Identify Different Layer Patterns

| Metric | Window Size | Unshuffled (Short) | Unshuffled (Long) | Shuffled |
|--------|-------------|-------------------|-------------------|----------|
| **Context Attention** | 2000-18000 tokens | Early layers (L0) | Late layers (L16-20) | Mixed (L0 + L17) |
| **Key Token Attention** | 1-3 tokens | **Late layers (L17-25)** | **Late layers (L17-25)** | **Late layers (L17-27)** |

### The Big Insight

**Key Token Attention shows the SAME late-layer heads for both shuffled and unshuffled!**

The difference is in **attention strength**, not which heads:
- Unshuffled: ~0.42 attention to answer tokens
- Shuffled: ~0.21 attention to answer tokens (~50% reduction)

This means:
1. The "copy heads" (L17-L27) are the same mechanism for both conditions
2. They just work WORSE when text is scrambled (weaker signal)
3. This explains the accuracy drop - same mechanism, less effective

### Why Context Attention Shows Different Results

Context Attention favors **early layers (L0)** at short contexts because:
- Layer 0 heads do broad, diffuse attention across many positions
- When you sum over a huge region (2000-18000 tokens), broad attention = high score
- Late layers do focused attention to specific tokens = lower total sum

Key Token Attention favors **late layers** because:
- It only looks at attention to the actual answer (1-3 tokens)
- Late layers are the ones that pinpoint and copy the specific answer
- Early layers' broad attention doesn't help when measuring attention to specific tokens

## Files in This Folder

| File | Description |
|------|-------------|
| `metric_layer_comparison.png` | Bar chart showing average layer of top 5 heads by metric and condition |
| `key_token_scores.png` | Attention strength comparison: unshuffled vs shuffled |
| `metric_comparison_summary.png` | Text summary of all findings |
| `heads_comparison_table.png` | Side-by-side table of top heads for both metrics |
| `plot_metric_comparison.py` | Script that generated these visualizations |

## Interpretation

### What This Tells Us About the Model

1. **Late layers (L17-L27) are the "copy heads"** - they extract and copy the answer
2. **Early layers (L0) do positional/structural attention** - they find the relevant region
3. **Shuffling breaks comprehension but not basic retrieval** - the model can still attend to the answer, just less effectively

### Original Hypothesis vs Current Results

**Original finding (with 500-token window):**
- Unshuffled → Late layers (L16-24) = comprehension heads
- Shuffled → Early layers (L2-5) = keyword matching heads

**Current finding (with full Section 1 / answer tokens):**
- Both metrics show late layers for both conditions
- The difference is in attention STRENGTH, not which heads

The discrepancy is likely due to:
1. Different window sizes (500 tokens vs 2000-18000 tokens)
2. OOM errors affecting which samples were analyzed
3. Different sample selection

## Experiment Configuration

- **Model**: Llama 3 8B Instruct
- **Samples**: 30 per condition
- **Context lengths**: 200, 500, 1000, 2000, 4000, 8000, 10000, 12000, 15000 tokens
- **Attention analysis**: Only for contexts ≤8000 tokens (memory constraint)
- **Needle position**: Middle (0.5)
- **Haystack**: Alice in Wonderland text


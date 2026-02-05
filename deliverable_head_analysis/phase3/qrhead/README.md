# Phase 3 Method 3: QRHead Ablation Study

## Status: ✅ COMPLETE (32/32 experiments)

## Overview

This phase tests the **causal importance** of attention heads identified by the QRHead method in Phase 2 by ablating them and measuring the impact on accuracy.

**Key Goal:** Replicate Figure 8 format—showing accuracy degradation as more heads are ablated, with random baseline for comparison.

## QRHead vs Wu24: Different Use Cases, Same Ablation

### QRHead's Original Approach (NOT ablation)

QRHead identifies "query-relevant" heads and uses them for **retrieval scoring**, not ablation. From `QRHead/src/qrretriever/attn_retriever.py`:

```python
# QRHead SELECTS heads for scoring (lines 229-236)
head_set = self.attn_head_set.split(',')  # e.g., "13-18,13-21,8-11"
head_set = [tuple(map(int, h.split('-'))) for h in head_set]
indices = torch.tensor(head_set).to(self.device)
per_token_scores_CAL = per_token_scores_CAL[layers, heads]  # SELECT heads' scores
```

The model still runs with ALL heads active; they just use a subset for retrieval scoring.

### Our Ablation Approach

For Phase 3, we use the same **hook-based ablation** as Methods 1 and 2 to test if QRHead's identified heads are **causally important** (not just correlated):

```python
class HeadAblator:
    """Zeros out attention head outputs BEFORE o_proj via forward pre-hooks."""
    
    def _make_ablation_pre_hook(self, layer_idx, heads_in_layer):
        def hook(module, args):
            hidden_states = args[0].clone()
            reshaped = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
            for head_idx in heads_in_layer:
                reshaped[:, :, head_idx, :] = 0  # Zero the head's OUTPUT
            return (reshaped.view(batch_size, seq_len, -1),)
        return hook
```

---

## Method

### Incremental Ablation Approach

Same as other Phase 3 methods:

```
Ablation Levels: [5, 10, 20, 30, 40, 50] heads
```

For each level N:
1. **Top N heads:** Ablate the top N heads from Phase 2 QRHead rankings
2. **Random N heads:** Ablate N random heads (excluding top heads) as baseline

### Head Loading (Dynamic)

Heads are loaded from Phase 2 QRHead results:

```python
def load_top_heads_from_phase2(model_key, question_key, total_tokens, top_n=50):
    results_path = f"phase2/qrhead/results/{model_dir}/{question_key}/tokens_{total_tokens}.json"
    # Parse "L15H30" -> (15, 30) tuples
    return [(layer, head) for h in results["head_rankings"][:top_n]]
```

### Random Baseline Selection

Random heads are selected **excluding** the top heads:

```python
def get_random_heads(all_heads, top_heads, n, seed=42):
    available = [h for h in all_heads if h not in top_heads]
    return random.sample(available, n)
```

---

## Key Results

### Validation: QRHead Heads Are Causally Important

| Ablation Level | Top Heads Drop | Random Drop | Gap |
|----------------|---------------|-------------|-----|
| 5 heads | 6.8% | 1.3% | 5.5% |
| 10 heads | 11.9% | 2.7% | 9.2% |
| 20 heads | 17.8% | 4.1% | 13.7% |
| 30 heads | 23.5% | 5.5% | 18.0% |
| 40 heads | 27.9% | 6.7% | 21.2% |
| 50 heads | 30.8% | 7.2% | 23.6% |

**Clear gap between top heads and random baseline** confirms QRHead identifies causally important heads, despite using a fundamentally different detection approach (calibrated query-relevant attention).

---

## Ablation Method Comparison

### Wu24's Original Ablation Approaches

| Method | What's Zeroed | Effect |
|--------|--------------|--------|
| Flash Attention | Query states (Q) | Head output ≈ 0 |
| Normal Attention | Attention logits (before softmax) | Head outputs **mean of V** (not zero!) |

### Our Approach (Used for All Phase 3 Methods)

| Method | What's Zeroed | Effect |
|--------|--------------|--------|
| **Output zeroing** | Attention output (before o_proj) | Head contribution = exactly 0 |

Our method is most similar to **Wu24's Flash Attention approach**—both effectively remove the head's contribution. This same mechanism is used consistently across all three Phase 3 methods for fair comparison.

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Models | Llama-3-8B-Instruct, Llama-3-8B-Base |
| Questions | inc_state, inc_year, employee_count, hq_state |
| Token Lengths | 2048, 4096, 6144, 8192 |
| Ablation Levels | 5, 10, 20, 30, 40, 50 heads |
| Comparison | Top heads vs Random heads |
| Test Set | 20% of GT-verified samples (from Phase 1) |
| max_new_tokens | 10 (one-word answers) |

**Total experiments:** 2 models × 4 questions × 4 token lengths = **32 experiments**

Each experiment tests 6 ablation levels × 2 conditions = 12 ablation conditions

---

## Files

| File | Description |
|------|-------------|
| `run_ablation.py` | Main ablation script |
| `run_all.py` | Batch runner for all 32 experiments |
| `results/` | Output JSON files |

---

## Usage

```bash
cd phase3/qrhead/

# Single experiment
python run_ablation.py --model instruct --question inc_state --tokens 2048

# All 32 experiments
python run_all.py

# Skip random baseline (faster, for debugging)
python run_ablation.py --model instruct --question inc_state --tokens 2048 --skip-random
```

---

## Output JSON Format

```json
{
  "method": "qrhead_ablation_incremental",
  "model_key": "instruct",
  "question": "inc_state",
  "total_tokens": 2048,
  "ablation_levels": [5, 10, 20, 30, 40, 50],
  "baseline": {
    "accuracy": 0.85,
    "correct": 28,
    "total": 33
  },
  "top_heads_ablations": [
    {"num_heads": 5, "accuracy": 0.82, "accuracy_drop": 0.03},
    {"num_heads": 10, "accuracy": 0.76, "accuracy_drop": 0.09},
    // ...
  ],
  "random_heads_ablations": [
    {"num_heads": 5, "accuracy": 0.84, "accuracy_drop": 0.01},
    // ...
  ]
}
```

---

## Expected Results (Figure 8 Replication)

If QRHead correctly identifies retrieval heads:

1. **Top heads ablation** should show significant accuracy drop
2. **Random heads ablation** should show minimal accuracy drop
3. The **gap between curves** indicates the causal importance of identified heads

```
Accuracy
  |
  |  ****  Random baseline (minimal drop)
  |    ****
  |      ****
  |  
  |  oooo  Top heads (significant drop)
  |      oooo
  |          oooo
  |               oooo
  +-----------------------> Number of ablated heads
     5  10  20  30  40  50
```

**Result:** ✅ The gap is clear—QRHead identifies causally important heads.

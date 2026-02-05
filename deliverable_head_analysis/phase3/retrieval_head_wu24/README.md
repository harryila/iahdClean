# Phase 3 Method 2: Wu24 Retrieval Head Ablation Study

## Overview

This phase tests the **causal importance** of attention heads identified by the Wu24 Retrieval Head method in Phase 2 by ablating them and measuring the impact on accuracy.

**Key Goal**: Replicate Figure 8 from the Wu24 paper - showing accuracy degradation as more heads are ablated, with random baseline for comparison.

## Method

### Incremental Ablation Approach

Unlike single-point ablation, we test **multiple ablation levels** to create a curve:

```
Ablation Levels: [5, 10, 20, 30, 40, 50] heads
```

For each level N:
1. **Top N heads**: Ablate the top N heads from Phase 2 Wu24 rankings
2. **Random N heads**: Ablate N random heads (excluding top heads) as baseline

This allows direct comparison: if top heads are causally important, ablating them should cause a larger accuracy drop than ablating random heads.

### Ablation Mechanism

Since Wu24's modified LlamaForCausalLM has compatibility issues with newer transformers versions, we use the same hook-based approach as Phase 3 Method 1:

```python
class HeadAblator:
    """Zeros out attention head outputs BEFORE o_proj via forward pre-hooks."""
    
    def _make_ablation_pre_hook(self, layer_idx, heads_in_layer):
        def hook(module, args):
            hidden_states = args[0].clone()
            # Reshape to [batch, seq, num_heads, head_dim]
            reshaped = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
            # Zero out specified heads
            for head_idx in heads_in_layer:
                reshaped[:, :, head_idx, :] = 0
            return (reshaped.view(batch_size, seq_len, -1),)
        return hook
```

The key difference from Method 1 is the **source of heads**: Phase 2 Wu24 results vs Phase 2 Summed Attention results.

### Head Loading (Dynamic)

Heads are loaded dynamically from Phase 2 Wu24 results:

```python
def load_top_heads_from_phase2(model_key, question_key, total_tokens, top_n=50):
    results_path = f"phase2/retrieval_head_wu24/results/{model_dir}/{question_key}/tokens_{total_tokens}.json"
    # Parse "L15H30" -> (15, 30) tuples
    return [(layer, head) for h in results["head_rankings"][:top_n]]
```

### Random Baseline Selection

Random heads are selected **excluding** the top heads to ensure fair comparison:

```python
def get_random_heads(all_heads, top_heads, n, seed=42):
    available = [h for h in all_heads if h not in top_heads]
    return random.sample(available, n)
```

## Prompt Format

```
[Haystack padding - Alice in Wonderland text]
[Section 1 content - the "needle"]
[Haystack padding - Alice in Wonderland text]

Question: {question}
Answer in one word:
```

**Important:** The prompt explicitly asks for a **one-word answer** to ensure consistent evaluation. `max_new_tokens=10` limits generation to at most 10 tokens.

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Models | Llama-3-8B-Instruct, Llama-3-8B-Base |
| Questions | inc_state, inc_year, employee_count, hq_state |
| Token Lengths | 2048, 4096, 6144, 8192 |
| Ablation Levels | 5, 10, 20, 30, 40, 50 heads |
| Comparison | Top heads vs Random heads |
| Test Set | 20% of GT-verified samples |
| max_new_tokens | 10 (one-word answers) |

**Total experiments**: 2 models × 4 questions × 4 token lengths = **32 experiments**

Each experiment tests 6 ablation levels × 2 conditions (top + random) = 12 ablation conditions

## Scripts

### `run_ablation.py`

Main ablation script with arguments:
- `--model`: instruct or base
- `--question`: inc_state, inc_year, employee_count, hq_state
- `--tokens`: 2048, 4096, 6144, 8192
- `--skip-random`: Skip random baseline (faster, for debugging)

### `run_all.py`

Batch runner for all 32 experiments.

## Results Format

```json
{
  "method": "wu24_ablation_incremental",
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
    {"num_heads": 20, "accuracy": 0.64, "accuracy_drop": 0.21},
    ...
  ],
  "random_heads_ablations": [
    {"num_heads": 5, "accuracy": 0.84, "accuracy_drop": 0.01},
    {"num_heads": 10, "accuracy": 0.82, "accuracy_drop": 0.03},
    {"num_heads": 20, "accuracy": 0.79, "accuracy_drop": 0.06},
    ...
  ]
}
```

## Expected Results (Figure 8 Replication)

If Wu24's method correctly identifies retrieval heads:

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

## Ablation Method Comparison

### Wu24's Original Ablation Approaches

Wu24's `faiss_attn/source/modeling_llama.py` implements TWO different ablation strategies via `block_list`:

**1. Flash Attention (lines 554-558)** - Zero the **query states**:
```python
if 'block_list' in kwargs:
    for h in kwargs['block_list']:
        if self.layer_idx==h[0]:
            query_states[:,h[1], :, :] = 0  # Zero QUERY before attention
```
Effect: Q·K^T = 0 → attention output ≈ 0 for that head

**2. Normal Attention (lines 686-689)** - Zero the **attention weights** before softmax:
```python
if 'block_list' in kwargs:
    for h in kwargs['block_list']:
        if self.layer_idx==h[0]:                   
            attn_weights[:, h[1], :, :] = 0  # Zero attention scores BEFORE softmax
```
Effect: softmax([0,0,...]) = uniform distribution → head outputs **mean of all values** (NOT zero!)

### Our Approach

We use **output zeroing** via pre-hooks on `o_proj`:

```python
def hook(module, args):
    reshaped = hidden_states.view(batch, seq, num_heads, head_dim)
    reshaped[:, :, head_idx, :] = 0  # Zero the head's OUTPUT directly
```
Effect: Head's contribution to residual stream = exactly 0

### Comparison

| Method | What's Zeroed | Effect |
|--------|--------------|--------|
| Wu24 Flash Attention | Query states (Q) | Head output ≈ 0 |
| Wu24 Normal Attention | Attention logits | Head outputs **mean of V** (not zero!) |
| **Ours** | Attention output | Head contribution = exactly 0 |

Our method is most similar to **Wu24's Flash Attention approach** - both effectively remove the head's contribution. Wu24's Normal Attention approach is actually a different intervention (uniform attention rather than removal).

### Why We Use Our Method

1. **Consistency**: Same ablation mechanism across all three Phase 3 methods
2. **Independence**: Works with any attention implementation (SDPA, Flash, eager)
3. **True ablation**: Completely removes head contribution (unlike Wu24 Normal Attention)
4. **Compatibility**: No need for Wu24's modified model classes

## Code Attribution

This implementation is inspired by:
- `WU_Retrieval_Head/needle_in_haystack_with_mask.py` (ablation logic)
- `WU_Retrieval_Head/faiss_attn/source/modeling_llama.py` (original ablation mechanism)

Due to compatibility issues with newer transformers, we use hook-based ablation instead of Wu24's modified model.

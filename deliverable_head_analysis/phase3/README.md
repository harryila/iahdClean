# Phase 3: Ablation Study

## Status: ✅ COMPLETE (96/96 experiments)

| Method | Status |
|--------|--------|
| Summed Attention | ✅ 32/32 complete |
| Wu24 Retrieval Head | ✅ 32/32 complete |
| QRHead | ✅ 32/32 complete |

## Purpose

Test the **causal importance** of heads identified in Phase 2 by ablating them and measuring accuracy drop.

Phase 2 identified heads that correlate with retrieval (high attention, copy behavior, query-relevance). Phase 3 answers: **If we disable these heads, does retrieval accuracy actually drop?**

## Goal: Replicate Figure 8 from Wu24 Paper

The key visualization:
- **X-axis:** Number of heads ablated (0, 5, 10, 20, 30, 40, 50)
- **Y-axis:** Accuracy (%)
- **Two lines:**
  - **Top-k heads:** Ablate top k retrieval heads → big drop = heads are important
  - **Random-k heads:** Ablate k random heads → small drop = baseline

If the methods correctly identify retrieval heads, ablating top heads should cause **much larger** accuracy drops than ablating random heads.

## Ablation Method

All three Phase 3 methods use the same ablation mechanism for fair comparison:

### Hook-Based Output Zeroing

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

### Comparison with Wu24's Original Ablation

Wu24's `faiss_attn/source/modeling_llama.py` has two approaches:

| Method | What's Zeroed | Effect |
|--------|--------------|--------|
| Wu24 Flash Attention | Query states (Q) | Head output ≈ 0 |
| Wu24 Normal Attention | Attention logits (before softmax) | Head outputs **mean of V** (not zero!) |
| **Our approach** | Attention output (before o_proj) | Head contribution = exactly 0 |

Our method is most similar to Wu24's Flash Attention approach—both effectively remove the head's contribution.

---

## Experiment Design

### Incremental Ablation Levels

```
Ablation Levels: [5, 10, 20, 30, 40, 50] heads
```

For each level N:
1. **Top N heads:** Ablate the top N heads from Phase 2 rankings
2. **Random N heads:** Ablate N random heads (excluding top heads) as baseline

### Test Set

Uses `phase1/test_samples.json` (20% held-out data, 50 samples total).

### Accuracy Measurement

```python
# One-word answer extraction and comparison
generated = model.generate(input_ids, max_new_tokens=10)
answer = extract_first_word(tokenizer.decode(generated))
correct = normalize(answer) == normalize(ground_truth)
accuracy = sum(correct) / total
```

---

## Experiments Per Method

For each of the 3 methods:
- 2 models × 4 questions × 4 token lengths = 32 experiments
- Each experiment tests 6 ablation levels × 2 conditions = 12 ablation conditions
- Total: 32 × 12 = 384 ablation runs per method

**Total across all methods: 96 experiments, 1,152 ablation runs**

---

## Key Results

### All Methods Work: Top Heads Cause Significant Drops

| Method | Mean Drop @ 50 Heads | Random Baseline |
|--------|---------------------|-----------------|
| Summed Attention | 30.8% | ~5% |
| Wu24 Retrieval Head | 30.9% | ~5% |
| QRHead | 30.8% | ~5% |

All three methods identify heads that, when ablated, cause ~30% accuracy drops—validating they find causally important heads.

### Model Comparison

| Model | Mean Drop @ 50 Heads |
|-------|---------------------|
| Llama-3-8B-Instruct | 24.9% |
| Llama-3-8B-Base | 36.8% |

Base model shows larger drops, suggesting more concentrated retrieval mechanisms.

### Question Type Matters

| Question | Avg Drop |
|----------|----------|
| Inc Year (numerical) | 30.9% |
| HQ State (categorical) | 23.6% |
| Employee Count (numerical) | 15.3% |
| Inc State (categorical) | 7.6% |

---

## Output Format

Each method produces JSON files in `{method}/results/`:

```json
{
  "method": "summed_attention_ablation_incremental",
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
    {"num_heads": 30, "accuracy": 0.55, "accuracy_drop": 0.30},
    {"num_heads": 40, "accuracy": 0.48, "accuracy_drop": 0.37},
    {"num_heads": 50, "accuracy": 0.42, "accuracy_drop": 0.43}
  ],
  "random_heads_ablations": [
    {"num_heads": 5, "accuracy": 0.84, "accuracy_drop": 0.01},
    {"num_heads": 10, "accuracy": 0.82, "accuracy_drop": 0.03},
    {"num_heads": 20, "accuracy": 0.79, "accuracy_drop": 0.06},
    // ...
  ]
}
```

---

## Usage

### Run All Experiments

```bash
# Summed Attention ablation
cd phase3/summed_attention/
python run_all.py

# Wu24 ablation
cd phase3/retrieval_head_wu24/
python run_all.py

# QRHead ablation
cd phase3/qrhead/
python run_all.py
```

### Run Single Experiment

```bash
python run_ablation.py --model instruct --question inc_state --tokens 2048
```

---

## Figures Generated

The ablation results are visualized in Phase 4:

| Figure | Description |
|--------|-------------|
| `figure8_option_a_by_question.png` | Ablation curves by question (comparing methods) |
| `figure8_option_b_by_method.png` | Ablation curves by method (comparing questions) |
| `figure8_option_c_grid.png` | Complete 4×3 grid |
| `accuracy_drop_summary.png` | Bar chart of drops at 50 heads |

**See:** `phase4/figures/`

# Phase 3: Ablation Study

## Purpose
Test the **causal importance** of heads identified in Phase 2 by ablating them and measuring accuracy drop.

## Goal: Replicate Figure 8
- X-axis: Number of heads ablated (0, 10, 20, 30, 40, 50)
- Y-axis: Accuracy (%)
- Two lines:
  - **Top-k heads:** Ablate top k retrieval heads → big drop = heads are important
  - **Random-k heads:** Ablate k random heads → small drop = baseline

## Ablation Implementation
Uses `WU_Retrieval_Head/needle_in_haystack_with_mask.py` with `block_list` parameter:
```python
outputs = model(input_ids=inp, past_key_values=past_kv, 
                block_list=block_list)  # Zeros out specified heads
```

## Experiments per Method
For each of the 3 methods (using heads identified in Phase 2):
- 2 models × 4 questions × 6 ablation levels × 2 conditions = 96 runs per method
- Total: 288 runs

## Output Format
Each method produces JSON files in `{method}/results/`:
```json
{
    "config": {...},
    "ablation_results": {
        "top_heads": {
            "0": {"accuracy": 0.66},
            "10": {"accuracy": 0.58},
            ...
        },
        "random_heads": {
            "0": {"accuracy": 0.66},
            "10": {"accuracy": 0.62},
            ...
        }
    }
}
```

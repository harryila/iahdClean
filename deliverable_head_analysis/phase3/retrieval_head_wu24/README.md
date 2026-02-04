# Phase 3: Ablation - Retrieval Head (Wu24)

## Purpose
Test if heads identified by **Retrieval Head** method (Phase 2) are causally important.

## Input
Head rankings from `../phase2/retrieval_head_wu24/results/`

## Process
1. Load top 50 heads from Phase 2
2. For k = 0, 10, 20, 30, 40, 50:
   - Ablate top-k heads → measure accuracy
   - Ablate random-k heads → measure accuracy (baseline)
3. Compare curves

## Ablation Code
Uses `WU_Retrieval_Head/needle_in_haystack_with_mask.py` pattern with `block_list`.

## Usage
```bash
python run_ablation.py --model llama_instruct --question inc_state
```

## Output
`results/{model}_{question}_ablation.json`

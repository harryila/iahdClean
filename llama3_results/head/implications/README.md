# Implications Experiments

This folder contains experiments to test the practical implications of our retrieval head analysis findings.

## Experiments

### Experiment 1: Attention Boosting (`experiment1_boost_attention.py`)

**Hypothesis:** If we artificially boost attention in L17-L27 heads when running on shuffled text, accuracy should improve.

**Method:**
- Run shuffled text through model normally → baseline accuracy
- Run shuffled text with 2x attention boost on copy heads → measure improvement

**Expected Result:** Accuracy improvement would prove that:
1. The copy heads (L17-L27) ARE the bottleneck
2. Stronger attention signal → better retrieval
3. Retrieval can be "forced" even on scrambled text

### Experiment 2: Ablation Study (`experiment2_ablation.py`)

**Hypothesis:** Disabling L17H24, L20H14, L24H27 should hurt accuracy the most.

**Method:**
- Baseline accuracy on unshuffled text
- Disable individual heads → measure accuracy drop
- Compare which heads are most important

**Configurations tested:**
- `baseline`: No ablation
- `L17H24`: Single head ablation
- `L20H14`: Single head ablation  
- `L24H27`: Single head ablation
- `top3_copy`: Ablate top 3 copy heads together
- `top5_copy`: Ablate top 5 copy heads together
- `L0_heads`: Ablate early layer heads (for comparison)

### Experiment 3: Replicate Original Finding (`experiment3_replicate_original.py`)

**Goal:** Reproduce the original finding:
- UNSHUFFLED: Later layers (L16-24) do retrieval
- SHUFFLED: Early layers (L2-5) do retrieval

**Key Difference:** Uses 500-token window (like original analysis) instead of full Section 1.

This should show whether the layer distribution difference depends on the window size.

## How to Run

```bash
# Make sure GPU is clear
nvidia-smi

# Run each experiment
cd /home/ubuntu/iahd/retrieval_heads/llama3_results/head/implications

python3 experiment1_boost_attention.py
python3 experiment2_ablation.py
python3 experiment3_replicate_original.py
```

## Files

| File | Description |
|------|-------------|
| `experiment1_boost_attention.py` | Attention boosting experiment |
| `experiment2_ablation.py` | Head ablation experiment |
| `experiment3_replicate_original.py` | 500-token window replication |
| `experiment*_results.json` | Results (generated after running) |
| `experiment3_replication_visual.png` | Visualization (generated after running) |


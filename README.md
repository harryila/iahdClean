# Retrieval Head Analysis in LLMs

Investigation of retrieval heads in Llama 3/3.1, studying how models retrieve information from needle-in-haystack tasks using SEC 10-K filings.

## Project Overview

**Task**: State of incorporation retrieval from SEC 10-K filings  
**Needle**: Entire Section 1 text (~600 tokens)  
**Haystack**: Alice in Wonderland text (Project Gutenberg)  
**Models**: Llama 3 8B, Llama 3.1 8B

## Key Findings

### 1. Two Attention Metrics Reveal Different Patterns

| Metric | Description | Key Result |
|--------|-------------|------------|
| **contextAlg** | Sum attention over entire Section 1 | Early layers (L0) dominate at short context, L17H24 at long context |
| **keyAlg** | Sum attention to answer tokens only | Late layers (L17-25) always dominate, ~50% reduction when shuffled |

### 2. Critical Discovery: High Attention ≠ Causal Importance

From ablation experiments:
- Disabling late "copy heads" (L17-24) → only **-3%** accuracy drop
- Disabling early L2 heads → **-33%** accuracy drop

**Early layers do the critical retrieval work, late layers just refine.**

### 3. Context Length Effects

| Model | Max Context | 8K Accuracy | 50K Accuracy | 130K Accuracy |
|-------|-------------|-------------|--------------|---------------|
| Llama 3 | 8,192 | 66.7% | N/A (collapse) | N/A |
| Llama 3.1 | 128,000 | 80.0% | 83.3% | 86.7% |

Shuffling hurts accuracy by 20-50% depending on context length.

## Repository Structure

```
├── needle_haystack_sweep.py          # Core experiment infrastructure
├── edgar_gt_verified_slim.csv        # Ground truth data (250 samples)
├── llama3_context_attention_sweep.py # contextAlg experiment
├── llama3_key_token_attention_sweep.py # keyAlg experiment
├── HarryUpdates.md                   # Experiment log
│
├── llama3_results/
│   ├── llama3_context_sweep_0.json   # Baseline unshuffled results
│   ├── llama_context_sweep_shuffled_0.json
│   └── head/
│       ├── contextAlg/               # Context attention results
│       ├── keyAlg/                   # Key token attention results
│       ├── comparison_contextVkey/   # Metric comparison
│       └── implications/             # Ablation & boosting experiments
│           └── FINDINGS.md           # Summary of causal analysis
│
└── llama31_results/                  # Llama 3.1 (up to 130K context)
```

## Running Experiments

### Prerequisites

```bash
pip install torch transformers datasets pandas matplotlib tqdm
```

### Basic Context Sweep

```bash
python needle_haystack_sweep.py --model llama --samples 30 --sweep length
python needle_haystack_sweep.py --model llama --samples 30 --sweep length --shuffle
```

### Attention Analysis

```bash
python llama3_context_attention_sweep.py  # contextAlg
python llama3_key_token_attention_sweep.py # keyAlg
```

### Ablation Study

```bash
cd llama3_results/head/implications
python experiment2_ablation.py
```

## Key Files

| File | Purpose |
|------|---------|
| `needle_haystack_sweep.py` | Core infrastructure - data loading, context creation, inference |
| `edgar_gt_verified_slim.csv` | Verified ground truth for 250 SEC filings |
| `llama3_results/head/implications/FINDINGS.md` | Summary of causal analysis findings |

## Citation

Based on retrieval head analysis methodology. Uses EDGAR corpus for SEC filings.

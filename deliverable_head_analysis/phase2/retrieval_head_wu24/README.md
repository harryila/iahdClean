# Retrieval Head Method (Wu24)

## Status: ✅ COMPLETE (32/32 experiments)

## Overview

This implements the attention head detection method from:
> **"Retrieval Head Mechanistically Explains Long-Context Factuality"**  
> Wu et al., 2024 ([arXiv:2404.15574](https://arxiv.org/abs/2404.15574))

The key insight: During **decoding**, identify heads whose **top attention** (argmax) points to the needle region AND the attended token **matches** the generated token. This captures "copy-like" retrieval behavior.

## Algorithm

### Core Logic (from `WU_Retrieval_Head/retrieval_head_detection.py` lines 221-229)

```python
def retrieval_calculate(attention_matrix, retrieval_score, inp, step_token, topk=1):
    """
    For each head, check if its top attention points to needle AND copies.
    """
    for layer_idx in range(layer_num):
        for head_idx in range(head_num):
            # Get the position with highest attention from last token
            values, idx = attention_matrix[layer_idx][0][head_idx][-1].topk(topk)
            
            for v, i in zip(values, idx):
                # Two conditions must BOTH be true:
                # 1. Position is within needle region
                # 2. Generated token matches the token at that position
                if needle_start <= i < needle_end and inp.item() == prompt_ids[i].item():
                    # Increment score (normalized by needle length)
                    retrieval_score[layer_idx][head_idx][0] += 1 / (needle_end - needle_start)
                    break  # Only count once per head per step
```

### Success Filtering

Following the original paper, we only count head scores when **retrieval is successful**:

```python
# ROUGE-1 recall > 50% indicates successful retrieval
rouge_score = scorer.score(ground_truth, generated_text)['rouge1'].recall * 100

if rouge_score > 50:
    # Only then accumulate head scores
    for head, score in retrieval_scores.items():
        all_head_scores[head].append(score)
```

---

## Key Differences from Summed Attention

| Aspect | Summed Attention | Wu24 Retrieval Head |
|--------|------------------|---------------------|
| **When computed** | Encoding (before generation) | Decoding (each generation step) |
| **Aggregation** | Sum of all attention weights | argmax (top-1 only) |
| **Token matching** | Not required | Required (must match generated token) |
| **Success filter** | None (always counts) | ROUGE > 50% required |
| **What it captures** | General attention magnitude | Specific copy/retrieval behavior |
| **Speed** | Fast (~0.4s/sample) | Slower (~8-9s/sample) |

### Why Token Matching Matters

The Wu24 method requires that:
1. The head's top attention points to a token in the needle region
2. That token **matches** the token being generated

This captures "copy-like" behavior where the model is literally retrieving and copying from context. A head might attend strongly to the needle but not be responsible for copying—this method specifically identifies heads that DO copy.

---

## Results Summary

### Top Heads (Example: instruct/inc_state/2048)

| Rank | Head | Score | Samples |
|------|------|-------|---------|
| 1 | L15H30 | 0.0107 | 121 |
| 2 | **L16H1** | 0.0073 | 121 |
| 3 | L24H27 | 0.0070 | 121 |
| 4 | L15H1 | 0.0068 | 119 |
| 5 | **L20H14** | 0.0063 | 122 |

### Key Finding: Different Top Heads Than Summed Attention

Wu24 identifies **L24H27** as a top head, which ranks much lower in Summed Attention. This head likely has strong "copy" behavior but lower overall attention magnitude.

### Cross-Method Validation

Heads appearing in top rankings for BOTH Summed Attention and Wu24:
- **L16H1**: Summed Attention rank 1, Wu24 rank 2
- **L20H14**: Summed Attention rank 5, Wu24 rank 5

These heads are validated as important by two independent methods.

---

## Key Characteristics

| Aspect | Description |
|--------|-------------|
| **When computed** | During decoding (each generation step) |
| **Aggregation** | argmax (top-1 attention only) |
| **Token matching** | Required - must be "copying" from needle |
| **Success filter** | ROUGE > 50% required |
| **Speed** | Slower (~8-9s per sample at 2K tokens) |

---

## Files

| File | Description |
|------|-------------|
| `run_detection.py` | Main detection script |
| `run_all.py` | Batch runner for all 32 experiments |
| `results/` | Output JSON files (32 complete) |

---

## Usage

```bash
cd phase2/retrieval_head_wu24/

# Single experiment
python run_detection.py --model instruct --question inc_state --tokens 2048

# All 32 experiments (takes several hours)
python run_all.py
```

---

## Output JSON Format

```json
{
  "method": "wu24_retrieval_head",
  "model_key": "instruct",
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "question": "inc_state",
  "question_prompt": "What state was the company incorporated in?",
  "total_tokens": 2048,
  "needle_position": 0.5,
  "samples_processed": 131,
  "successful_retrievals": 123,
  "success_rate": 0.969,
  "timestamp": "2026-02-04T...",
  "head_rankings": [
    {"head": "L15H30", "score": 0.0107, "num_samples": 121, "rank": 1},
    {"head": "L16H1", "score": 0.0073, "num_samples": 121, "rank": 2},
    // ...
  ],
  "top_50_heads": ["L15H30", "L16H1", ...]
}
```

---

## Key Finding: Wu24 Has Extreme #1 Dominance

Analysis shows Wu24's top head leads by **24.3%** on average over the #2 head (compared to ~3.5% for other methods). This suggests Wu24 identifies a few "super-heads" with very strong copy behavior.

**See:** `phase4/exploration_figures/score_dominance.png`

---

## Dependencies

```python
# Required for success filtering
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
```

---

## Based On

- **Paper:** Wu et al., "Retrieval Head Mechanistically Explains Long-Context Factuality"
- **Original code:** `WU_Retrieval_Head/retrieval_head_detection.py`
- **Key functions:** `retrieval_calculate()` (lines 221-229), `decode()` (lines 235-247)

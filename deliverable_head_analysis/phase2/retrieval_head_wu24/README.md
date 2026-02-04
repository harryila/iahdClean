# Retrieval Head Method (Wu24)

## Overview

This implements the attention head detection method from:
> **"Retrieval Head Mechanistically Explains Long-Context Factuality"**  
> Wu et al., 2024 ([arXiv:2404.15574](https://arxiv.org/abs/2404.15574))

The key insight: During **decoding**, identify heads whose **top attention** (argmax) points to the needle region AND the attended token **matches** the generated token. This captures "copy-like" retrieval behavior.

## Needle-in-Haystack Setup

Same structure as all Phase 2 methods:

```
[Padding - Alice in Wonderland text]
[Needle - Entire Section 1 containing the GT answer]
[Padding - More Alice in Wonderland text]

Question: {question}
Answer:
```

### Key Parameters

| Parameter | Value |
|-----------|-------|
| **Needle** | Entire Section 1 from SEC filing |
| **Haystack** | Alice in Wonderland text |
| **Needle Position** | 0.5 (fixed in middle) |
| **Total Tokens** | 2K, 4K, 6K, 8K |

---

## Algorithm

### Core Logic (from `WU_Retrieval_Head/retrieval_head_detection.py` lines 221-229)

```python
def retrieval_calculate(attention_matrix, retrieval_score, inp, step_token, topk=1):
    """
    For each head, check if its top attention points to needle AND copies.
    
    Args:
        attention_matrix: Attention weights from all layers/heads
        retrieval_score: Dict accumulating scores per head
        inp: The token ID that was just generated
        step_token: String representation of generated token
        needle_start, needle_end: Token indices of needle region
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

This ensures we're measuring heads that contribute to **correct** retrieval, not just any attention.

---

## Key Differences from Summed Attention

| Aspect | Summed Attention (Method 1) | Wu24 Retrieval Head (Method 2) |
|--------|----------------------------|-------------------------------|
| **When computed** | Encoding (before generation) | Decoding (each generation step) |
| **Aggregation** | Sum of all attention weights | argmax (top-1 only) |
| **Token matching** | Not required | Required (must match generated token) |
| **Success filter** | None (always counts) | ROUGE > 50% required |
| **What it captures** | General attention magnitude to needle | Specific copy/retrieval behavior |
| **Speed** | Fast (~0.4s/sample at 2k) | Slower (~8-9s/sample at 2k) |

### Why Token Matching Matters

The Wu24 method requires that:
1. The head's top attention points to a token in the needle region
2. That token **matches** the token being generated

This captures "copy-like" behavior where the model is literally retrieving and copying from context. A head might attend strongly to the needle but not be responsible for copying—this method specifically identifies heads that DO copy.

---

## Implementation Details

### Files

| File | Description |
|------|-------------|
| `run_detection.py` | Main detection script |
| `run_all.py` | Batch runner for all 32 experiments |
| `results/` | Output JSON files |

### Memory Management

For longer sequences (6K/8K tokens), we use an efficient generation approach:
- KV caching during generation
- Attention computed only for current token
- GPU memory cleared between samples

### Dependencies

```python
# Required for success filtering
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
```

---

## Experiment Matrix

**2 models × 4 questions × 4 token lengths = 32 experiments**

### Models

| Model | HuggingFace ID |
|-------|----------------|
| Llama 3 8B Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Llama 3 8B Base | `meta-llama/Meta-Llama-3-8B` |

### Questions

| Key | Question | Type | Samples |
|-----|----------|------|---------|
| `inc_state` | What state was the company incorporated in? | Categorical | 127 |
| `inc_year` | What year was the company incorporated? | Numerical | 126 |
| `employee_count` | How many employees does the company have? | Numerical | 152 |
| `hq_state` | What state is the company headquarters located in? | Categorical | 106 |

---

## Usage

```bash
cd phase2/retrieval_head_wu24/

# Single experiment
python run_detection.py --model instruct --question inc_state --tokens 2048

# All token lengths for one config
python run_detection.py --model instruct --question inc_state --tokens 2048
python run_detection.py --model instruct --question inc_state --tokens 4096
# ... etc

# All 32 experiments (takes several hours)
python run_all.py
```

---

## Output Structure

```
results/
├── llama3_instruct/
│   ├── inc_state/
│   │   ├── tokens_2048.json
│   │   ├── tokens_4096.json
│   │   ├── tokens_6144.json
│   │   └── tokens_8192.json
│   ├── inc_year/
│   │   └── ...
│   └── ...
└── llama3_base/
    └── (same structure)
```

### Output JSON Format

```json
{
  "method": "wu24_retrieval_head",
  "model_key": "instruct",
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "question": "inc_state",
  "question_prompt": "What state was the company incorporated in?",
  "total_tokens": 2048,
  "needle_position": 0.5,
  "samples_processed": 127,
  "successful_retrievals": 123,
  "success_rate": 0.969,
  "timestamp": "2026-02-04T08:30:02.698044",
  "head_rankings": [
    {"head": "L15H30", "score": 0.0107, "num_samples": 121, "rank": 1},
    {"head": "L16H1", "score": 0.0073, "num_samples": 121, "rank": 2},
    ...
  ],
  "top_50_heads": ["L15H30", "L16H1", ...]
}
```

---

## Results

### Completed: instruct/inc_state/2048

| Metric | Value |
|--------|-------|
| Samples processed | 127 |
| Successful retrievals | 123 |
| Success rate | 96.9% |

**Top 10 Retrieval Heads:**

| Rank | Head | Score | Samples |
|------|------|-------|---------|
| 1 | L15H30 | 0.0107 | 121 |
| 2 | **L16H1** | 0.0073 | 121 |
| 3 | L24H27 | 0.0070 | 121 |
| 4 | L15H1 | 0.0068 | 119 |
| 5 | **L20H14** | 0.0063 | 122 |
| 6 | L16H20 | 0.0059 | 120 |
| 7 | L8H6 | 0.0057 | 118 |
| 8 | L31H5 | 0.0056 | 119 |
| 9 | L14H31 | 0.0055 | 118 |
| 10 | L20H1 | 0.0054 | 120 |

### Cross-Method Validation

Heads appearing in top rankings for BOTH Summed Attention and Wu24:
- **L16H1**: Summed Attention rank 1, Wu24 rank 2
- **L20H14**: Summed Attention rank 5 (implicit), Wu24 rank 5

This cross-method agreement suggests these heads are genuinely important for retrieval.

---

## Based On

- **Paper:** Wu et al., "Retrieval Head Mechanistically Explains Long-Context Factuality"
- **Original code:** `WU_Retrieval_Head/retrieval_head_detection.py`
- **Key functions:** `retrieval_calculate()` (lines 221-229), `decode()` (lines 235-247)

---

## Progress

| Model | Question | 2K | 4K | 6K | 8K |
|-------|----------|----|----|----|----|
| Instruct | inc_state | ✅ | ⏳ | ⏳ | ⏳ |
| | inc_year | ⏳ | ⏳ | ⏳ | ⏳ |
| | employee_count | ⏳ | ⏳ | ⏳ | ⏳ |
| | hq_state | ⏳ | ⏳ | ⏳ | ⏳ |
| Base | inc_state | ⏳ | ⏳ | ⏳ | ⏳ |
| | inc_year | ⏳ | ⏳ | ⏳ | ⏳ |
| | employee_count | ⏳ | ⏳ | ⏳ | ⏳ |
| | hq_state | ⏳ | ⏳ | ⏳ | ⏳ |

**Completed: 1/32**

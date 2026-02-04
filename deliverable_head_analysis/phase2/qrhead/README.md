# QRHead Method

## Overview

Query-focused attention method from the QRHead paper. The key insight: compute attention **FROM query tokens TO document tokens**, then **calibrate** by subtracting attention with a null query to isolate query-relevant heads.

This is fundamentally different from Summed Attention (last token → needle) and Wu24 (generated token → needle).

## Needle-in-Haystack Setup

Same structure as all Phase 2 methods (for fair comparison):

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

Adapted from `QRHead/exp_scripts/detection/detect_qrhead_lme.py` and `QRHead/src/qrretriever/attn_retriever.py`.

### Step 1: Compute Query-to-Document Attention

```python
# Get attention FROM query tokens TO all tokens
# Shape: (num_layers, num_heads, num_query_tokens, seq_len)
actual_attn = get_attention_from_query(model, prompt_with_actual_query)
```

### Step 2: Calibration (Key QRHead Feature)

```python
# Compute attention with null query ("N/A") 
null_attn = get_attention_from_query(model, prompt_with_null_query)

# Subtract to remove query-independent attention
# From QRHead/src/qrretriever/attn_retriever.py lines 291-292
calibrated_attn = actual_attn - null_attn
```

**Why calibration?** Some heads attend to certain regions regardless of the query. Subtracting the null-query attention isolates heads that attend **because of the specific question**.

### Step 3: Extract Needle Region & Average Over Query Tokens

```python
# Extract attention to needle region only
needle_attn = calibrated_attn[:, :, :, needle_start:needle_end]

# Average over query tokens (following QRHead pattern)
# Shape: (num_layers, num_heads, needle_length)
needle_attn = needle_attn.mean(dim=2)
```

### Step 4: Outlier Removal (Key QRHead Feature)

```python
# From QRHead/src/qrretriever/attn_retriever.py lines 304-312
for layer in range(num_layers):
    for head in range(num_heads):
        scores = needle_attn[layer, head]
        
        # Threshold: mean - 2*std (per-head)
        threshold = scores.mean() - 2 * scores.std()
        
        # Mask out abnormally low scores
        mask = scores > threshold
        
        # Sum only "normal" tokens
        head_scores[f"L{layer}H{head}"] = (scores * mask).sum()
```

**Why outlier removal?** Removes tokens with abnormally low (possibly negative after calibration) attention that could skew the sum.

### Step 5: Aggregate Across Samples

```python
# From QRHead/exp_scripts/detection/detect_qrhead_lme.py lines 9-32
def lme_eval(retrieval_results, data_instances):
    all_score_over_gold = []
    for data in data_instances:
        gt_docs = data["gt_docs"]  # For us: the needle
        score_over_gold = sum(doc_scores[doc_id] for doc_id in gt_docs)
        all_score_over_gold.append(score_over_gold)
    return np.mean(all_score_over_gold)  # QRScore for this head
```

---

## Key Differences from Other Methods

| Aspect | Summed Attention | Wu24 | **QRHead** |
|--------|------------------|------|------------|
| **Attention from** | Last token | Generated tokens | **Query tokens** |
| **Attention to** | Needle | Needle (if matches) | **Needle** |
| **Calibration** | No | No | **Yes (null query)** |
| **Outlier removal** | No | No | **Yes (mean-2σ)** |
| **When computed** | Encoding | Decoding | **Encoding** |
| **What it captures** | General attention | Copy behavior | **Query-relevant attention** |

---

## Implementation Details

### Files

| File | Description |
|------|-------------|
| `run_detection.py` | Main detection script |
| `run_all.py` | Batch runner for all 32 experiments |
| `results/` | Output JSON files |

### Code Reuse from QRHead

| Component | Source |
|-----------|--------|
| Calibration logic | `attn_retriever.py` lines 291-292 |
| Outlier removal | `attn_retriever.py` lines 304-312 |
| Head scoring aggregation | `detect_qrhead_lme.py` lines 66-101 |
| LME evaluation | `detect_qrhead_lme.py` lines 9-32 |

### Null Query

Following QRHead, we use `"N/A"` as the null query for calibration.

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
cd phase2/qrhead/

# Single experiment
python run_detection.py --model instruct --question inc_state --tokens 2048

# All 32 experiments
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
  "method": "qrhead",
  "model_key": "instruct",
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "question": "inc_state",
  "total_tokens": 2048,
  "null_query": "N/A",
  "samples_processed": 127,
  "head_rankings": [
    {"head": "L16H1", "score": 45.23, "rank": 1},
    ...
  ]
}
```

---

## Based On

- **Paper:** QRHead - Query-Relevant Head Detection
- **Detection script:** `QRHead/exp_scripts/detection/detect_qrhead_lme.py`
- **Core retriever:** `QRHead/src/qrretriever/attn_retriever.py`
- **Key functions:**
  - `score_docs_per_head_for_detection()` (lines 258-320)
  - `lme_eval()` (lines 9-32)
  - `score_heads()` (lines 66-101)

---

## Progress

| Model | inc_state | inc_year | employee_count | hq_state |
|-------|-----------|----------|----------------|----------|
| Instruct | ⏳ | ⏳ | ⏳ | ⏳ |
| Base | ⏳ | ⏳ | ⏳ | ⏳ |

**Completed: 0/32**

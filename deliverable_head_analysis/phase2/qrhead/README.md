# QRHead Method

## Status: ✅ COMPLETE (32/32 experiments)

## Overview

Query-focused attention method from the QRHead paper. The key insight: compute attention **FROM query tokens TO document tokens**, then **calibrate** by subtracting attention with a null query to isolate query-relevant heads.

This is fundamentally different from:
- **Summed Attention:** last token → needle
- **Wu24:** generated token → needle (with token matching)
- **QRHead:** query tokens → needle (with calibration)

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
# Average head scores across all training samples
final_scores[head] = mean(sample_scores[head] for all samples)
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

## Results Summary

### Top Heads (Typical Configuration)

QRHead tends to identify different heads than the other methods:

| Rank | Head | Note |
|------|------|------|
| 1 | **L14H31** | Also top for Summed Attention |
| 2 | L8H8 | Unique to QRHead |
| 3 | L15H3 | Unique to QRHead |

### Key Finding: Different Layer Focus

QRHead focuses heavily on **mid-layers (9-10, 14-16)**, while Wu24 is more distributed and Summed Attention focuses on layers 13-14 and 20-21.

**See:** `phase4/exploration_figures/layer_distribution_by_method.png`

---

## Key Characteristics

| Aspect | Description |
|--------|-------------|
| **When computed** | During encoding |
| **Calibration** | Yes - subtracts null-query attention |
| **Outlier removal** | Yes - mean-2σ threshold |
| **Speed** | Medium (2 forward passes per sample) |

---

## Files

| File | Description |
|------|-------------|
| `run_detection.py` | Main detection script |
| `run_all.py` | Batch runner for all 32 experiments |
| `results/` | Output JSON files (32 complete) |

### Code Reuse from QRHead Repository

| Component | Source |
|-----------|--------|
| Calibration logic | `attn_retriever.py` lines 291-292 |
| Outlier removal | `attn_retriever.py` lines 304-312 |
| Head scoring aggregation | `detect_qrhead_lme.py` lines 66-101 |

---

## Null Query

Following QRHead, we use `"N/A"` as the null query for calibration:

```python
# Actual query
prompt_actual = f"...Question: What state was the company incorporated in?\nAnswer in one word:"

# Null query  
prompt_null = f"...Question: N/A\nAnswer in one word:"

# Calibration
calibrated = actual_attention - null_attention
```

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

## Output JSON Format

```json
{
  "method": "qrhead",
  "model_key": "instruct",
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "question": "inc_state",
  "question_prompt": "What state was the company incorporated in?",
  "total_tokens": 2048,
  "null_query": "N/A",
  "samples_processed": 131,
  "timestamp": "2026-02-04T...",
  "head_rankings": [
    {"head": "L14H31", "score": 45.23, "rank": 1},
    {"head": "L8H8", "score": 42.15, "rank": 2},
    // ...
  ],
  "top_50_heads": ["L14H31", "L8H8", ...]
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

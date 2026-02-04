# Phase 2: Head Identification

## Purpose

Identify the top 50 attention heads for each question type using 3 different methods, across multiple context lengths.

## Needle-in-Haystack Setup (All Methods)

Every experiment uses the same needle-in-haystack structure:

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

### Token Length Sweep

Each method sweeps across 4 total token lengths:

| Total Tokens | Description |
|--------------|-------------|
| 2,048 | Short context |
| 4,096 | Medium context |
| 6,144 | Medium-long context |
| 8,192 | Max context for Llama 3 8B |

This lets us analyze how head importance changes with context length.

---

## Methods

### 1. Summed Attention (`summed_attention/`) - COMPLETED ✓

Our original method: Sum attention from last token to Section 1 region.

#### Algorithm

```python
# For each head, sum attention weights from last token to needle region
score = sum(attention[last_token, needle_start:needle_end])
```

#### Key Characteristics

| Aspect | Description |
|--------|-------------|
| **When computed** | During encoding (before generation) |
| **Aggregation** | Sum of all attention weights to needle |
| **Success criterion** | Always counts (no ROUGE filter) |
| **Speed** | Fast (~0.4s per sample at 2k tokens) |

#### Implementation

- **Script:** `summed_attention/run_detection.py`
- **Batch runner:** `summed_attention/run_all.py`
- **Memory optimization:** Uses forward hooks for 6k/8k tokens to avoid OOM

#### Results Summary (32/32 experiments complete)

| Model | Question | 2K Top Head | 4K Top Head | 6K Top Head | 8K Top Head |
|-------|----------|-------------|-------------|-------------|-------------|
| **Instruct** | inc_state | L16H1 | L16H1 | L14H31 | L20H14 |
| | inc_year | L20H25 | L16H1 | L20H25 | L20H25 |
| | employee_count | L13H18 | L16H1 | L31H14 | L23H14 |
| | hq_state | L16H9 | L16H9 | L16H9 | L16H9 |
| **Base** | inc_state | L16H1 | L16H1 | L31H14 | L31H14 |
| | inc_year | L16H1 | L17H24 | L9H1 | L9H1 |
| | employee_count | L13H3 | L16H1 | L31H14 | L9H1 |
| | hq_state | L16H9 | L16H9 | L31H14 | L31H14 |

**Key Findings:**
- **L16H1** dominates at short contexts (2K/4K) for most questions
- **L16H9** is remarkably consistent for `hq_state` in Instruct model (all contexts)
- **L31H14** emerges at longer contexts (6K/8K) in Base model
- Different context lengths reveal different important heads

---

### 2. Retrieval Head Wu24 (`retrieval_head_wu24/`) - IN PROGRESS

Paper's method from "Retrieval Head Mechanistically Explains Long-Context Factuality" (Wu et al., 2024).

#### Algorithm

```python
# From WU_Retrieval_Head/retrieval_head_detection.py lines 221-229
def retrieval_calculate(attention_matrix, retrieval_score, inp, step_token, topk=1):
    for layer_idx in range(layer_num):
        for head_idx in range(head_num):
            # Get top-1 attention position from the last generated token
            values, idx = attention_matrix[layer_idx][0][head_idx][-1].topk(topk)
            
            for v, i in zip(values, idx):
                # TWO conditions must be met:
                # 1. Attention points to needle region
                # 2. Generated token MATCHES token at that position (copy behavior)
                if needle_start <= i < needle_end and inp.item() == prompt_ids[i].item():
                    retrieval_score[layer_idx][head_idx][0] += 1/(needle_end - needle_start)
                    break
```

#### Key Characteristics

| Aspect | Description |
|--------|-------------|
| **When computed** | During decoding (each generation step) |
| **Aggregation** | argmax (top-1 attention only) |
| **Token matching** | Required - must be "copying" from needle |
| **Success criterion** | Only counts if ROUGE > 50% (successful retrieval) |
| **Speed** | Slower (~8-9s per sample at 2k tokens due to autoregressive generation) |

#### Key Differences from Summed Attention

| Aspect | Summed Attention | Wu24 Retrieval Head |
|--------|------------------|---------------------|
| **When** | Encoding (before generation) | Decoding (during generation) |
| **What** | Sum of attention to needle | Whether argmax points to needle AND token matches |
| **Score type** | Continuous (attention sum) | Discrete (copy event count) |
| **Success filter** | None | ROUGE > 50% required |
| **Captures** | General attention to needle | Specific "copy-like" retrieval behavior |

#### Implementation

- **Script:** `retrieval_head_wu24/run_detection.py`
- **Batch runner:** `retrieval_head_wu24/run_all.py`
- **Dependencies:** `rouge-score` for success filtering

#### Results Summary (1/32 experiments complete)

**instruct/inc_state/2048:**
- Samples processed: 127
- Successful retrievals: 123 (96.9% success rate)

| Rank | Head | Score | Samples |
|------|------|-------|---------|
| 1 | L15H30 | 0.0107 | 121 |
| 2 | **L16H1** | 0.0073 | 121 |
| 3 | L24H27 | 0.0070 | 121 |
| 4 | L15H1 | 0.0068 | 119 |
| 5 | **L20H14** | 0.0063 | 122 |
| 6 | L16H20 | 0.0059 | 120 |

---

### 3. QRHead (`qrhead/`) - NOT YET IMPLEMENTED

Query-focused method from the QRHead paper.

- **Code basis:** `QRHead/src/qrretriever/attn_retriever.py`
- **Metric:** Calibrated query→document attention
- **See:** `qrhead/README.md` for planned details

---

## Cross-Method Comparison

### inc_state, Instruct, 2K tokens

| Summed Attention | Wu24 Retrieval Head | Notes |
|------------------|---------------------|-------|
| **L16H1** (rank 1, score 117.8) | **L16H1** (rank 2, score 0.0073) | **Appears in both!** |
| L14H31 (rank 2) | L15H30 (rank 1) | Different |
| L17H24 (rank 3) | L24H27 (rank 3) | Different |
| L18H20 (rank 4) | L15H1 (rank 4) | Different |
| **L20H14** (rank 5, implicit) | **L20H14** (rank 5, score 0.0063) | **Appears in both!** |

**Key Insight:** L16H1 and L20H14 appear in the top 5 for BOTH methods, providing cross-method validation that these heads are genuinely important for retrieval.

---

## Experiment Matrix

**2 models × 4 questions × 4 token lengths = 32 experiments per method**

**3 methods × 32 experiments = 96 total head identification runs**

### Models

| Model | HuggingFace ID |
|-------|----------------|
| Llama 3 8B Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Llama 3 8B Base | `meta-llama/Meta-Llama-3-8B` |

### Questions

| Key | Question | Type | Train Samples |
|-----|----------|------|---------------|
| `inc_state` | Incorporation state | Categorical | 127 |
| `inc_year` | Incorporation year | Numerical | 126 |
| `employee_count` | Total employees | Numerical | 152 |
| `hq_state` | Headquarters state | Categorical | 106 |

---

## Output Structure

Each method stores results in:

```
{method}/results/
├── llama3_instruct/
│   ├── inc_state/
│   │   ├── tokens_2048.json
│   │   ├── tokens_4096.json
│   │   ├── tokens_6144.json
│   │   └── tokens_8192.json
│   ├── inc_year/
│   │   └── ...
│   ├── employee_count/
│   │   └── ...
│   └── hq_state/
│       └── ...
└── llama3_base/
    └── (same structure)
```

### JSON Output Format

**Summed Attention:**
```json
{
  "method": "summed_attention",
  "model_key": "instruct",
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "question": "inc_state",
  "total_tokens": 2048,
  "samples_processed": 127,
  "head_rankings": [
    {"head": "L16H1", "score": 117.79, "rank": 1},
    ...
  ]
}
```

**Wu24 Retrieval Head:**
```json
{
  "method": "wu24_retrieval_head",
  "model_key": "instruct",
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "question": "inc_state",
  "total_tokens": 2048,
  "samples_processed": 127,
  "successful_retrievals": 123,
  "success_rate": 0.969,
  "head_rankings": [
    {"head": "L15H30", "score": 0.0107, "num_samples": 121, "rank": 1},
    ...
  ]
}
```

---

## Data Sources

- **Training samples:** `phase1/train_samples.json` (80% of GT data)
- **Ground truth:** `edgar_gt_verified_slim.csv`
- **SEC filings:** `phase1/section1_cache.json` (cached from `c3po-ai/edgar-corpus`)
- **Haystack:** Alice in Wonderland from `needle_haystack_sweep.py`

---

## Usage

### Summed Attention

```bash
cd summed_attention/

# Single experiment
python run_detection.py --model instruct --question inc_state --tokens 4096

# All 32 experiments
python run_all.py
```

### Wu24 Retrieval Head

```bash
cd retrieval_head_wu24/

# Single experiment
python run_detection.py --model instruct --question inc_state --tokens 2048

# All 32 experiments  
python run_all.py
```

---

## Progress Tracking

| Method | Status | Completed |
|--------|--------|-----------|
| Summed Attention | ✅ Complete | 32/32 |
| Wu24 Retrieval Head | 🔄 In Progress | 1/32 |
| QRHead | ⏳ Not Started | 0/32 |

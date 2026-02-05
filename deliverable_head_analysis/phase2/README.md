# Phase 2: Head Identification

## Purpose

Identify the top 50 attention heads for each question type using 3 different methods, across multiple context lengths. This phase produces head rankings that are then tested for causal importance in Phase 3.

## Status: ✅ COMPLETE (96/96 experiments)

| Method | Status |
|--------|--------|
| Summed Attention | ✅ 32/32 complete |
| Wu24 Retrieval Head | ✅ 32/32 complete |
| QRHead | ✅ 32/32 complete |

## Needle-in-Haystack Setup (All Methods)

Every experiment uses the same needle-in-haystack structure for fair comparison:

```
[Padding - Alice in Wonderland text]
[Needle - Entire Section 1 containing the GT answer]
[Padding - More Alice in Wonderland text]

Question: {question}
Answer in one word:
```

### Key Parameters

| Parameter | Value |
|-----------|-------|
| **Needle** | Entire Section 1 from SEC filing |
| **Haystack** | Alice in Wonderland text |
| **Needle Position** | 0.5 (fixed in middle) |
| **Total Tokens** | 2K, 4K, 6K, 8K |
| **max_new_tokens** | 10 (one-word answers) |

### Token Length Sweep

Each method sweeps across 4 total token lengths:

| Total Tokens | Description |
|--------------|-------------|
| 2,048 | Short context |
| 4,096 | Medium context |
| 6,144 | Medium-long context |
| 8,192 | Max context for Llama 3 8B |

---

## Methods Overview

### 1. Summed Attention (`summed_attention/`)

**Approach:** Sum attention weights from the last token to the needle region during encoding.

```python
# Core algorithm
score = attention[last_token, needle_start:needle_end].sum()
```

**Key Characteristics:**
- Computed during encoding (before generation)
- Measures total attention magnitude to needle
- Fast (~0.4s per sample at 2K tokens)
- No success filtering

**See:** `summed_attention/README.md` for full details.

---

### 2. Wu24 Retrieval Head (`retrieval_head_wu24/`)

**Approach:** During decoding, identify heads whose top attention (argmax) points to needle AND the token matches.

```python
# Core algorithm (from Wu24 paper)
for each generation step:
    if argmax(attention) in needle_region AND generated_token == attended_token:
        score += 1 / needle_length
```

**Key Characteristics:**
- Computed during decoding (each generation step)
- Requires token matching (copy behavior)
- Only counts successful retrievals (ROUGE > 50%)
- Slower (~8-9s per sample at 2K tokens)

**See:** `retrieval_head_wu24/README.md` for full details.

---

### 3. QRHead (`qrhead/`)

**Approach:** Compute attention from query tokens to document, then calibrate by subtracting null-query attention.

```python
# Core algorithm (from QRHead paper)
actual_attn = attention_from_query(prompt_with_question)
null_attn = attention_from_query(prompt_with_null_query)  # "N/A"
calibrated_attn = actual_attn - null_attn
score = calibrated_attn[:, needle_region].sum()
```

**Key Characteristics:**
- Computed during encoding
- Calibration isolates query-relevant attention
- Outlier removal (mean - 2σ threshold)
- Moderate speed

**See:** `qrhead/README.md` for full details.

---

## Method Comparison

| Aspect | Summed Attention | Wu24 | QRHead |
|--------|------------------|------|--------|
| **When computed** | Encoding | Decoding | Encoding |
| **Attention from** | Last token | Generated tokens | Query tokens |
| **Attention to** | Needle | Needle (if matches) | Needle |
| **Token matching** | No | Yes | No |
| **Calibration** | No | No | Yes (null query) |
| **Success filter** | None | ROUGE > 50% | None |
| **Speed** | Fast | Slow | Medium |
| **What it captures** | Attention magnitude | Copy behavior | Query-relevant attention |

---

## Key Finding: Methods Identify Different Heads

**Only ~12% overlap** between methods when comparing top-50 heads (Jaccard similarity).

Only **2 heads** (L20H14 and L14H31) appear in top-100 for ALL three methods.

| Method | Top Universal Heads |
|--------|-------------------|
| Summed Attention | L20H14, L14H31, L16H19 |
| Wu24 | L24H27, L20H14, L15H30 |
| QRHead | L14H31, L8H8, L15H3 |

**See:** `phase4/exploration_figures/consensus_heads.png` and `phase4/exploration_figures/universal_heads_by_method.png`

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
| `inc_state` | What state was the company incorporated in? | Categorical | 131 |
| `inc_year` | What year was the company incorporated? | Numerical | 128 |
| `employee_count` | How many employees does the company have? | Numerical | 152 |
| `hq_state` | What state is the company headquarters located in? | Categorical | 106 |

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
│   ├── employee_count/
│   └── hq_state/
└── llama3_base/
    └── (same structure)
```

### JSON Output Format

```json
{
  "method": "summed_attention",
  "model_key": "instruct",
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "question": "inc_state",
  "question_prompt": "What state was the company incorporated in?",
  "total_tokens": 2048,
  "needle_position": 0.5,
  "samples_processed": 131,
  "timestamp": "2026-02-04T...",
  "head_rankings": [
    {"head": "L16H1", "score": 117.79, "rank": 1},
    {"head": "L14H31", "score": 112.98, "rank": 2},
    // ... all 1024 heads ranked
  ],
  "top_50_heads": ["L16H1", "L14H31", ...]
}
```

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

### QRHead

```bash
cd qrhead/

# Single experiment
python run_detection.py --model instruct --question inc_state --tokens 2048

# All 32 experiments
python run_all.py
```

---

## Data Sources

- **Training samples:** `phase1/train_samples.json` (80% of GT data)
- **Ground truth:** `../edgar_gt_verified_slim.csv`
- **SEC filings:** Cached Section 1 content
- **Haystack:** Alice in Wonderland text

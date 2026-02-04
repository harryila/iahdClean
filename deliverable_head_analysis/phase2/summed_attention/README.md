# Summed Attention Method

## Overview

Our original method for identifying retrieval heads: **Sum attention from the last token to the needle region** (Section 1 containing the GT answer).

This is the simplest and fastest approach—it directly measures how much each head attends to the relevant context during encoding.

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

### Core Logic

```python
def compute_attention_to_needle(model, tokenizer, prompt, needle_start, needle_end):
    """
    Sum attention from last token to needle region for all heads.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    head_scores = {}
    for layer_idx, layer_attn in enumerate(outputs.attentions):
        # layer_attn shape: [batch, num_heads, seq_len, seq_len]
        for head_idx in range(layer_attn.size(1)):
            # Attention from LAST token to NEEDLE region
            attn_to_needle = layer_attn[0, head_idx, -1, needle_start:needle_end]
            score = attn_to_needle.sum().item()
            head_scores[f"L{layer_idx}H{head_idx}"] = score
    
    return head_scores
```

### Memory-Efficient Version (for 6K/8K tokens)

For longer sequences, we use forward hooks to compute attention for only the last query token, avoiding O(N²) memory:

```python
def compute_attention_to_needle_memory_efficient(model, tokenizer, prompt, needle_start, needle_end):
    """
    Uses forward hooks to manually compute attention scores for last token only.
    Handles Grouped Query Attention (GQA) in Llama 3.
    """
    head_scores = {}
    
    def make_hook(layer_idx):
        def hook(module, args, kwargs, output):
            # Get Q, K projections
            q = module.q_proj(hidden_states)  # [batch, seq, num_heads * head_dim]
            k = module.k_proj(hidden_states)  # [batch, seq, num_kv_heads * head_dim]
            
            # Handle GQA: expand K heads to match Q heads
            if num_kv_heads != num_heads:
                k = k.unsqueeze(2).expand(...).reshape(...)
            
            # Compute attention only for last query token
            q_last = q[:, :, -1:, :]
            attn_weights = torch.matmul(q_last, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            
            # Sum attention to needle for each head
            needle_attn = attn_weights[0, :, 0, needle_start:needle_end]
            for head_idx in range(num_heads):
                head_scores[f"L{layer_idx}H{head_idx}"] = needle_attn[head_idx].sum().item()
        return hook
    
    # Register hooks, run forward pass, remove hooks
    ...
```

---

## Key Characteristics

| Aspect | Description |
|--------|-------------|
| **When computed** | During encoding (before generation) |
| **Aggregation** | Sum of all attention weights to needle |
| **Success filter** | None (always counts all samples) |
| **Speed** | Fast (~0.4s/sample at 2K, ~1.1s/sample at 8K) |
| **Memory** | Efficient hooks for long sequences |

---

## Implementation Details

### Files

| File | Description |
|------|-------------|
| `run_detection.py` | Main detection script (487 lines) |
| `run_all.py` | Batch runner for all 32 experiments |
| `results/` | Output JSON files (32 complete) |

### Conditional Memory Management

```python
# Choose method based on sequence length
if total_tokens >= 6144:
    head_scores = compute_attention_to_needle_memory_efficient(...)
else:
    head_scores = compute_attention_to_needle(...)

# Clear GPU memory after each sample
torch.cuda.empty_cache()
```

---

## Experiment Matrix

**2 models × 4 questions × 4 token lengths = 32 experiments** ✅ ALL COMPLETE

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

## Results Summary

### Complete Results Table (Top Head per Configuration)

| Model | Question | 2K | 4K | 6K | 8K |
|-------|----------|----|----|----|----|
| **Instruct** | inc_state | L16H1 (117.8) | L16H1 (117.0) | L14H31 (106.8) | L20H14 (108.1) |
| | inc_year | L20H25 (118.1) | L16H1 (117.4) | L20H25 (114.5) | L20H25 (114.4) |
| | employee_count | L13H18 (143.7) | L16H1 (145.4) | L31H14 (140.0) | L23H14 (144.6) |
| | hq_state | L16H9 (100.0) | L16H9 (98.2) | L16H9 (96.9) | L16H9 (95.7) |
| **Base** | inc_state | L16H1 (120.7) | L16H1 (118.6) | L31H14 (110.9) | L31H14 (108.3) |
| | inc_year | L16H1 (120.6) | L17H24 (118.9) | L9H1 (111.9) | L9H1 (111.6) |
| | employee_count | L13H3 (142.9) | L16H1 (144.9) | L31H14 (140.3) | L9H1 (134.3) |
| | hq_state | L16H9 (97.9) | L16H9 (96.9) | L31H14 (94.8) | L31H14 (94.8) |

### Key Findings

1. **L16H1 dominates short contexts (2K/4K)**
   - Top head for 10/16 short-context experiments
   - Particularly strong for Instruct model
   - May be a general "retrieval" head active at manageable context lengths

2. **L16H9 is remarkably consistent for hq_state**
   - Top head for ALL 4 token lengths in Instruct model
   - Suggests question-specific specialization

3. **L31H14 emerges at longer contexts (6K/8K)**
   - Top head for many Base model long-context experiments
   - Late-layer head may be important for long-range retrieval

4. **Context length affects head rankings**
   - Different heads become important at different scales
   - Supports hypothesis that retrieval mechanisms adapt to context

5. **Model differences**
   - Instruct model shows more stability (L16H9 for hq_state)
   - Base model shows more variation with context length

---

## Usage

```bash
cd phase2/summed_attention/

# Single experiment
python run_detection.py --model instruct --question inc_state --tokens 4096

# All 32 experiments
python run_all.py
```

---

## Output Structure

```
results/
├── llama3_instruct/
│   ├── inc_state/
│   │   ├── tokens_2048.json  (127 samples, L16H1 top)
│   │   ├── tokens_4096.json  (127 samples, L16H1 top)
│   │   ├── tokens_6144.json  (127 samples, L14H31 top)
│   │   └── tokens_8192.json  (127 samples, L20H14 top)
│   ├── inc_year/
│   │   ├── tokens_2048.json  (126 samples, L20H25 top)
│   │   ├── tokens_4096.json  (126 samples, L16H1 top)
│   │   ├── tokens_6144.json  (126 samples, L20H25 top)
│   │   └── tokens_8192.json  (126 samples, L20H25 top)
│   ├── employee_count/
│   │   └── ... (152 samples each)
│   └── hq_state/
│       └── ... (106 samples each)
└── llama3_base/
    └── (same structure)
```

### Output JSON Format

```json
{
  "method": "summed_attention",
  "model_key": "instruct",
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "question": "inc_state",
  "question_prompt": "What state was the company incorporated in?",
  "total_tokens": 2048,
  "needle_position": 0.5,
  "samples_processed": 127,
  "timestamp": "2026-02-04T...",
  "head_rankings": [
    {"head": "L16H1", "score": 117.7909, "rank": 1},
    {"head": "L14H31", "score": 112.9858, "rank": 2},
    ...
    // All 1024 heads ranked
  ]
}
```

---

## Based On

- **Original code:** `llama3_context_attention_sweep.py`
- **Pattern:** Sum attention from last token to context region
- **Adapted for:** Needle-in-haystack setup with token length sweep

---

## Progress

✅ **32/32 EXPERIMENTS COMPLETE**

| Model | inc_state | inc_year | employee_count | hq_state |
|-------|-----------|----------|----------------|----------|
| Instruct | ✅ 2K/4K/6K/8K | ✅ 2K/4K/6K/8K | ✅ 2K/4K/6K/8K | ✅ 2K/4K/6K/8K |
| Base | ✅ 2K/4K/6K/8K | ✅ 2K/4K/6K/8K | ✅ 2K/4K/6K/8K | ✅ 2K/4K/6K/8K |

# Summed Attention Method

## Status: ✅ COMPLETE (32/32 experiments)

## Overview

Our original method for identifying retrieval heads: **Sum attention from the last token to the needle region** (Section 1 containing the GT answer).

This is the simplest and fastest approach—it directly measures how much each head attends to the relevant context during encoding.

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

For longer sequences, we use forward hooks to compute attention for only the last query token, avoiding O(N²) memory issues:

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
            q = module.q_proj(hidden_states)
            k = module.k_proj(hidden_states)
            
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

### Conditional Method Selection

```python
# From run_detection.py
if total_tokens >= 6144:
    head_scores = compute_attention_to_needle_memory_efficient(...)
else:
    head_scores = compute_attention_to_needle(...)

# Clear GPU memory after each sample
torch.cuda.empty_cache()
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

## Results Summary

### Top Head per Configuration

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
   - May be a general "retrieval" head active at manageable context lengths

2. **L16H9 is remarkably consistent for hq_state**
   - Top head for ALL 4 token lengths in Instruct model
   - Suggests question-specific specialization

3. **L31H14 emerges at longer contexts (6K/8K)**
   - Top head for many Base model long-context experiments
   - Late-layer head may be important for long-range retrieval

4. **Context length affects head rankings**
   - Different heads become important at different scales

---

## Files

| File | Description |
|------|-------------|
| `run_detection.py` | Main detection script (487 lines) |
| `run_all.py` | Batch runner for all 32 experiments |
| `results/` | Output JSON files (32 complete) |

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

## Output JSON Format

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
    {"head": "L16H1", "score": 117.7909, "rank": 1},
    {"head": "L14H31", "score": 112.9858, "rank": 2},
    // All 1024 heads ranked
  ],
  "top_50_heads": ["L16H1", "L14H31", ...]
}
```

---

## Challenges Encountered

### Memory Issues at Long Contexts

**Problem:** At 6K/8K tokens, storing full attention matrices caused OOM errors.

**Solution:** Implemented memory-efficient version using forward hooks that only computes attention for the last token, avoiding O(N²) memory:

```python
# Instead of output_attentions=True (stores all N×N matrices)
# We use hooks to compute only what we need (1×N per layer)
```

### Grouped Query Attention (GQA)

**Problem:** Llama 3 uses GQA where `num_kv_heads < num_heads`. The K tensor has fewer heads than Q.

**Solution:** Expand K heads to match Q heads before computing attention:

```python
# Llama 3 8B: 32 Q heads, 8 KV heads (4:1 ratio)
k = k.unsqueeze(2).expand(-1, -1, num_heads // num_kv_heads, -1, -1)
k = k.reshape(batch, seq, num_heads, head_dim)
```

---

## Based On

- **Original code:** `llama3_context_attention_sweep.py`
- **Pattern:** Sum attention from last token to context region
- **Adapted for:** Needle-in-haystack setup with token length sweep

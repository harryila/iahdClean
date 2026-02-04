# Retrieval Head Analysis Deliverable - Execution Plan

## Overview

This deliverable investigates how attention heads in Llama models retrieve information across 4 different ground truth (GT) categories, using 3 different attention analysis methods.

**Research Questions:**
1. Do the same number of ablated heads affect different questions equally?
2. Are there similarities between numerical and state heads?
3. Are ablations equally effective on each of the categories?
4. Do we see similarities among the numerical and state questions?

---

## Experiment Matrix

### Models (2)
| Model | HuggingFace Path | Notes |
|-------|------------------|-------|
| Llama-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | Chat-tuned, follows instructions |
| Llama-3-8B | `meta-llama/Meta-Llama-3-8B` | Base model, no instruction tuning |

### GT Categories (4)
| Category | Type | Example Answer | Notes |
|----------|------|----------------|-------|
| Incorporation State | Categorical | "Delaware" | US state name |
| Incorporation Year | Numerical | "1998" | 4-digit year |
| Total Employee Count | Numerical | "12500" | Integer count |
| HQ State | Categorical | "California" | US state name |

### Attention Methods (3)
| Method | Description | Source Code |
|--------|-------------|-------------|
| **Summed Attention** | Sum of attention from last token to Section 1 region | Our existing `llama3_context_attention_sweep.py` |
| **Retrieval Head (Wu24)** | Argmax-based: Does top attention point to needle during decoding? | `WU_Retrieval_Head/retrieval_head_detection.py` |
| **QRHead** | Query-focused: Sum of query→document attention with calibration | `QRHead/src/qrretriever/attn_retriever.py` |

### Total Experiments
**2 models × 3 methods × 4 questions = 24 experiments**

---

## Phase 1: Data Preparation

### 1.1 Ground Truth

**Source file:** `edgar_gt_verified_slim.csv` (250 samples)

**Already contains ALL 4 GT categories:**
| Deliverable Question | CSV Column |
|---------------------|------------|
| Incorporation State | `original_Inc_state_truth` |
| Incorporation Year | `original_Inc_year_truth` |
| Employee Count | `employee_count_truth` |
| HQ State | `headquarters_state_truth` |

**No additional data collection needed!**

**Action items:**
- [x] Verify CSV has all required columns ✓
- [ ] Check for NULL values per category
- [ ] Document valid sample counts per category

### 1.2 Train/Test Split

```python
from sklearn.model_selection import train_test_split

# 80/20 split, stratified if possible
train_data, test_data = train_test_split(
    data, 
    test_size=0.2, 
    random_state=42
)
```

**Purpose:**
- **Training set (80%)**: Identify which heads are important for each question type
- **Test set (20%)**: Evaluate ablation effects (prevents overfitting to specific samples)

---

## Phase 2: Head Identification (Training Set)

For each of the 24 experiment configurations, identify the top 50 attention heads.

### 2.1 Method 1: Summed Attention (Our Method)

**Source:** Adapted from `llama3_context_attention_sweep.py`

**Algorithm:**
```python
# For each head, compute attention sum over the relevant region
for layer_idx, layer_attn in enumerate(outputs.attentions):
    last_pos_attn = layer_attn[0, :, -1, :].float()  # (n_heads, seq_len)
    
    for head_idx in range(last_pos_attn.shape[0]):
        head_attn = last_pos_attn[head_idx].cpu().numpy()
        
        # Sum over Section 1 (or answer tokens)
        score = head_attn[section_start:section_end].sum()
        head_scores[f"L{layer_idx}H{head_idx}"] = score
```

**Reference:** `llama3_context_attention_sweep.py` lines 119-133

**Key decisions:**
- Use **last token** attention (like our existing code)
- Sum over **entire Section 1** (contextAlg) or **answer tokens only** (keyAlg)
- Average scores across all training samples

### 2.2 Method 2: Retrieval Head (Wu24)

**Source:** `WU_Retrieval_Head/retrieval_head_detection.py`

**Algorithm:**
```python
def retrieval_calculate(attention_matrix, retrieval_score, inp, step_token, topk=1):
    for layer_idx in range(self.layer_num):
        for head_idx in range(self.head_num):
            # Get top-1 attention position
            values, idx = attention_matrix[layer_idx][0][head_idx][-1].topk(topk)
            
            for v, i in zip(values, idx):
                # Check if top attention points to needle AND matches generated token
                if needle_start <= i < needle_end and inp.item() == prompt_ids[i].item():
                    retrieval_score[layer_idx][head_idx][0] += 1/(needle_end - needle_start)
                    break
```

**Reference:** `WU_Retrieval_Head/retrieval_head_detection.py` lines 221-229

**Key characteristics:**
- Uses **argmax** (top-1) attention, not sum
- Only scores during **decoding** (when generating answer)
- Requires token **matching** - the attended token must match the generated token
- Only accumulates scores when retrieval **succeeds** (ROUGE > 50)

**Adaptation needed:**
- Replace San Francisco needle with our SEC filing questions
- Handle 4 different question types
- Adapt token matching for state names, years, employee counts

### 2.3 Method 3: QRHead

**Source:** `QRHead/src/qrretriever/attn_retriever.py`

**Algorithm:**
```python
def score_docs_per_head_for_detection(query, docs):
    # Get attention scores with actual query
    per_token_scores, kv_cache = score_per_token_attention_to_query(prompt, query_span, None, 0)
    
    # Get attention scores with null query for calibration
    null_per_token_scores, _ = score_per_token_attention_to_query(null_prompt, null_query_span, kv_cache, start_idx)
    
    # Calibrate: subtract null query attention
    per_token_scores_CAL = per_token_scores - null_per_token_scores  # (n_layers, n_heads, n_tok)
    
    # For each document region, aggregate scores
    for doc_span in doc_spans:
        curr_scores = per_token_scores_CAL[:, :, doc_span[0]:doc_span[1]+1]
        
        # Remove outliers (below mean - 2*std)
        threshold = curr_scores.mean(dim=-1) - 2*curr_scores.std(dim=-1)
        tok_mask = curr_scores > threshold.unsqueeze(-1)
        
        # Sum masked scores -> (n_layers, n_heads)
        masked_scores = curr_scores.masked_fill(~tok_mask, 0.0).sum(dim=-1)
```

**Reference:** `QRHead/src/qrretriever/attn_retriever.py` lines 258-320

**Key characteristics:**
- Uses **query tokens** (not just last token) to compute attention
- **Calibration** with null query ("N/A") to remove baseline attention
- **Outlier removal** (tokens below mean - 2*std are masked)
- Returns per-head scores for each document

**Adaptation needed:**
- Our "documents" are SEC filing sections, not paragraphs
- Need to define query span (the question) and document span (Section 1)
- Adapt prompt format for Llama-3

---

## Phase 3: Ablation Study (Test Set)

For each experiment, ablate heads and measure accuracy drop.

### 3.1 Ablation Levels

Ablate k heads where k ∈ {0, 10, 20, 30, 40, 50}

### 3.2 Ablation Configurations

For each experiment:
1. **Top-k Retrieval Heads**: Ablate the top k heads identified in Phase 2
2. **Random Baseline**: Ablate k random non-retrieval heads

### 3.3 Ablation Implementation

**Source:** `WU_Retrieval_Head/needle_in_haystack_with_mask.py`

```python
# Load pre-computed head scores
with open(f"head_score/{model_name}.json", "r") as file:
    stable_block_list = json.loads(file.readline())

# Sort by score, get top heads
stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True)
block_list = [[int(ll) for ll in l[0].split("-")] for l in stable_block_list][:100]

# During inference, pass block_list to model
outputs = model(input_ids=inp, past_key_values=past_kv, use_cache=True,
                output_attentions=False, block_list=block_list)
```

**Reference:** `WU_Retrieval_Head/needle_in_haystack_with_mask.py` lines 218-228, 277-278

**Key:** The Wu24 code uses custom model implementations in `faiss_attn/source/` that accept a `block_list` parameter to zero out specific heads.

### 3.4 Output: Figure 8 Replication

For each experiment, produce a plot showing:
- X-axis: Number of heads ablated (0, 10, 20, 30, 40, 50)
- Y-axis: Accuracy (%)
- Lines: Top retrieval heads (solid) vs Random heads (dashed)

---

## Phase 4: Analysis & Comparison

### 4.1 Cross-Question Comparison

**Question:** Do the same heads matter for different questions?

- Compute Jaccard similarity of top-50 heads between question types
- Heatmap: question × question head overlap

### 4.2 Numerical vs Categorical

**Question:** Are there similarities between numerical (year, employee count) and categorical (state) questions?

- Compare layer distributions of top heads
- Compare ablation sensitivity curves

### 4.3 Cross-Method Comparison

**Question:** Do different methods identify the same heads?

- For each question type, compare heads identified by:
  - Summed Attention
  - Retrieval Head (Wu24)
  - QRHead
- Venn diagram or overlap matrix

### 4.4 Model Comparison

**Question:** Do Instruct and non-Instruct models use different heads?

- Compare head rankings between model variants
- Compare ablation sensitivity

---

## Folder Organization

```
deliverable_head_analysis/
├── PLAN.md                           # This file
├── README.md                         # High-level overview and results summary
│
├── data/
│   ├── README.md                     # Data documentation
│   ├── edgar_gt_expanded.csv         # Ground truth with all 4 categories
│   ├── train_samples.json            # 80% training split
│   └── test_samples.json             # 20% test split
│
├── phase1/                          # Data preparation
│   ├── README.md
│   ├── prepare_data.py
│   ├── train_samples.json           # Output: 80% for head identification
│   └── test_samples.json            # Output: 20% for ablation
│
├── phase2/                          # Head identification
│   ├── README.md
│   ├── summed_attention/
│   │   ├── README.md
│   │   ├── run_detection.py         # Our method
│   │   └── results/                 # Output: head scores per question
│   │       ├── llama_instruct_inc_state.json
│   │       └── ...
│   ├── retrieval_head_wu24/
│   │   ├── README.md
│   │   ├── run_detection.py         # Adapted from WU_Retrieval_Head
│   │   └── results/
│   └── qrhead/
│       ├── README.md
│       ├── run_detection.py         # Adapted from QRHead
│       └── results/
│
├── phase3/                          # Ablation study
│   ├── README.md
│   ├── summed_attention/
│   │   ├── README.md
│   │   ├── run_ablation.py
│   │   └── results/                 # Output: ablation accuracy curves
│   ├── retrieval_head_wu24/
│   │   ├── README.md
│   │   ├── run_ablation.py
│   │   └── results/
│   └── qrhead/
│       ├── README.md
│       ├── run_ablation.py
│       └── results/
│
└── phase4/                          # Analysis & figures
    ├── README.md
    ├── figures/                     # Output: all plots
    │   ├── figure8_replications/
    │   ├── cross_question_heatmaps/
    │   └── method_comparison/
    ├── analysis/                    # Analysis scripts
    │   ├── generate_figures.py
    │   └── cross_question_analysis.py
    └── FINDINGS.md                  # Final report
```

---

## Implementation Order

### Phase 1: Data Preparation (30 min)
- [ ] Run `phase1/prepare_data.py`
- [ ] Verify train/test split sizes
- Output: `train_samples.json`, `test_samples.json`

### Phase 2: Head Identification

**Summed Attention** (`phase2/summed_attention/`)
- [ ] Implement `run_detection.py`
- [ ] Run: Llama-Instruct × 4 questions
- [ ] Run: Llama-Base × 4 questions
- Output: 8 JSON files in `results/`

**Retrieval Head Wu24** (`phase2/retrieval_head_wu24/`)
- [ ] Implement `run_detection.py` (adapt from `WU_Retrieval_Head/retrieval_head_detection.py`)
- [ ] Run: Llama-Instruct × 4 questions
- [ ] Run: Llama-Base × 4 questions
- Output: 8 JSON files in `results/`

**QRHead** (`phase2/qrhead/`)
- [ ] Implement `run_detection.py` (adapt from `QRHead/exp_scripts/detection/detect_qrhead_lme.py`)
- [ ] Run: Llama-Instruct × 4 questions
- [ ] Run: Llama-Base × 4 questions
- Output: 8 JSON files in `results/`

### Phase 3: Ablation Study

For each method folder (`phase3/{method}/`):
- [ ] Implement `run_ablation.py` (adapt from `WU_Retrieval_Head/needle_in_haystack_with_mask.py`)
- [ ] Run ablation at k = 0, 10, 20, 30, 40, 50
- [ ] Compare top-k vs random-k heads
- Output: JSON files with accuracy curves

### Phase 4: Analysis
- [ ] Generate Figure 8 replications (24 plots)
- [ ] Generate cross-question heatmaps
- [ ] Write FINDINGS.md

8. **Day 13-14: Run ablation experiments**
   - [ ] 24 experiments × 6 ablation levels × 2 baselines = 288 runs
   - [ ] May need to batch and checkpoint

### Week 4: Analysis & Writeup (Phase 4)

9. **Day 15-16: Generate figures**
   - [ ] Figure 8 replications (24 plots)
   - [ ] Cross-question heatmaps
   - [ ] Method comparison plots

10. **Day 17-18: Analysis & Report**
    - [ ] Answer research questions
    - [ ] Write findings document
    - [ ] Create presentation-ready figures

---

## Code Dependencies

### From WU_Retrieval_Head

**Two scripts from the paper:**

| Script | Purpose | Usage |
|--------|---------|-------|
| `retrieval_head_detection.py` | **Detect** retrieval heads | `python retrieval_head_detection.py --model_path $model --s 0 --e 50000` |
| `needle_in_haystack_with_mask.py` | **Ablate** heads (Figure 8) | `python needle_in_haystack_with_mask.py --mask_top 30 ...` |

**What we use:**

| File | What We Use | How |
|------|-------------|-----|
| `retrieval_head_detection.py` | `retrieval_calculate()` lines 221-229 | Adapt for our 4 questions |
| `needle_in_haystack_with_mask.py` | Ablation with `block_list` | Adapt for our ablation study |
| `faiss_attn/source/modeling_llama.py` | Custom Llama with `block_list` param | Import for ablation |

### From QRHead

**Detection script only** (NOT the retrieval/generation benchmarks):

| Script | Purpose | Usage |
|--------|---------|-------|
| `detect_qrhead_lme.py` | **Detect** QRHeads | `python detect_qrhead_lme.py --input_file data.json --output_file scores.json` |

**What we use:**

| File | What We Use | How |
|------|-------------|-----|
| `exp_scripts/detection/detect_qrhead_lme.py` | Detection pattern, `score_heads()` | Adapt for our data format |
| `src/qrretriever/attn_retriever.py` | `FullHeadRetriever`, `score_docs_per_head_for_detection()` | Import and use |
| `src/qrretriever/custom_cache.py` | `DynamicCacheWithQuery` | Import and use |

**NOT using** (different tasks):
- `run_retrieval.py` - BEIR/CLIPPER/LME benchmarks
- `run_generation_*.py` - Generation tasks  
- `eval_*.py` - Benchmark evaluation

### Our Existing Code

| File | What We Use | How |
|------|-------------|-----|
| `llama3_context_attention_sweep.py` | Summed attention pattern | Refactor into reusable module |
| `needle_haystack_sweep.py` | Data loading, context creation | Import utilities |
| `edgar_gt_verified_slim.csv` | Base ground truth | Extend with more columns |

---

## Risk Mitigation

### Risk 1: Missing Ground Truth Data
- **Impact:** Cannot run experiments on all 4 question types
- **Mitigation:** Start with available data (incorporation state), add others incrementally

### Risk 2: Memory/Compute Constraints
- **Impact:** Cannot run all experiments
- **Mitigation:** 
  - Use GH200 (97GB VRAM) for longer contexts
  - Batch experiments, checkpoint results
  - Consider subset of context lengths if needed

### Risk 3: Method Adaptation Issues
- **Impact:** Incorrect head identification
- **Mitigation:**
  - Verify each method on known examples
  - Cross-check with original paper results where possible
  - Document all adaptations

### Risk 4: Incompatible Codebases
- **Impact:** Cannot integrate WU/QRHead code
- **Mitigation:**
  - Test imports early
  - Create wrapper classes to isolate dependencies
  - Document version requirements

---

## Success Criteria

1. **Head Scores:** JSON files with top 50 heads for all 24 configurations
2. **Ablation Results:** Accuracy at 6 ablation levels for all 24 configurations
3. **Figure 8 Replications:** 24 plots showing ablation curves
4. **Analysis Report:** Answers to all 4 research questions with supporting evidence
5. **Documentation:** README files explaining methodology and reproducing results

---

## Next Steps

1. Create folder structure
2. Implement data preparation
3. Adapt each method
4. Run experiments
5. Generate analysis

**Ready to proceed?**

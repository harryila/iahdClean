# Phase 4: Analysis & Figures

## Purpose

Generate visualizations, answer Eric's research questions, and produce the comprehensive `FINDINGS.md` report. This phase also includes deep exploration of the Phase 2/3 data to discover additional insights.

## Key Outputs

1. **`FINDINGS.md`** - Comprehensive findings report with all results and figure references
2. **`figures/`** - Main analysis figures (13 files) answering research questions
3. **`exploration_figures/`** - Deep exploration figures (14 files) with additional insights

---

## Scripts

### `analysis/run_all_analysis.py`
Master script that runs all analysis—**start here!**

### `analysis/generate_figure8.py`
Generates Figure 8 replications in three formats:
- **Option A:** Per-question (comparing methods)
- **Option B:** Per-method (comparing questions)
- **Option C:** Complete 4×3 grid

### `analysis/research_questions.py`
Generates visualizations answering Eric's research questions:
- Q1: Equal impact across questions
- Q2: Numerical vs categorical head similarity
- Q3: Ablation effectiveness by category
- Q4: Cross-question head patterns

Also generates method comparison, model comparison, and token length analysis.

### `analysis/generate_findings.py`
Compiles all results into `FINDINGS.md` report.

### `analysis/deep_exploration.py`
Deep analysis of Phase 2 head rankings:
- Head overlap between questions, models, methods
- Universal heads identification
- Layer distribution by method

### `analysis/deep_findings.py`
Finds surprising patterns in the data:
- Score magnitude differences
- Head volatility analysis
- Question specialists
- Instruct vs Base differences
- Token length sensitivity
- Cross-method agreement/disagreement

### `analysis/key_findings_viz.py`
Creates visualizations for key findings:
- Method disagreement
- Consensus heads
- Question specialists
- Instruct vs Base differences
- Head volatility
- Score dominance

---

## Usage

```bash
# Run all analysis (recommended)
cd phase4/analysis
python run_all_analysis.py

# Or run individual scripts
python generate_figure8.py
python research_questions.py
python generate_findings.py
python deep_exploration.py
python deep_findings.py
python key_findings_viz.py
```

---

## Main Figures (`figures/`)

| Figure | Description | Answers |
|--------|-------------|---------|
| `figure8_option_a_by_question.png` | Ablation curves comparing methods, grouped by question | Figure 8 replication |
| `figure8_option_b_by_method.png` | Ablation curves comparing questions, grouped by method | Figure 8 replication |
| `figure8_option_c_grid.png` | Complete 4×3 grid of all combinations | Figure 8 replication |
| `accuracy_drop_summary.png` | Bar chart of drops at 50 heads | Overall effectiveness |
| `q1_equal_impact.png` | Box plots of ablation impact per question | Q1: Equal impact? |
| `q2_head_overlap_questions.png` | Head overlap heatmap between questions | Q2: Head similarity? |
| `q2_numerical_vs_categorical.png` | Category comparison | Q2: Numerical vs categorical? |
| `q3_ablation_effectiveness.png` | Numerical vs categorical ablation effectiveness | Q3: Effectiveness by category? |
| `q4_head_patterns.png` | Layer-head frequency heatmaps | Q4: Patterns? |
| `method_head_overlap.png` | Method overlap heatmap | Method comparison |
| `method_unique_heads.png` | Unique heads per method | Method comparison |
| `model_comparison.png` | Instruct vs Base comparison | Model comparison |
| `token_length_effect.png` | Context length analysis | Token length effect |

---

## Exploration Figures (`exploration_figures/`)

| Figure | Description | Key Finding |
|--------|-------------|-------------|
| `consensus_heads.png` | Only 2 heads in top-100 for all methods | L20H14 and L14H31 are universal |
| `method_disagreement.png` | Same heads ranked differently by methods | Methods capture different phenomena |
| `question_specialists.png` | Heads important for only one question | Question-specific circuits exist |
| `instruct_vs_base_differences.png` | Heads with dramatic model differences | Instruction tuning reshapes retrieval |
| `head_volatility.png` | Most volatile heads across configs | Some heads swing rank 1→1000 |
| `score_dominance.png` | Gap between #1 and #2 heads | Wu24 has extreme #1 dominance (24%) |
| `overlap_summary.png` | Jaccard similarity summary | Method overlap is only ~12% |
| `question_overlap_heatmaps.png` | Question overlap by method-model | ~45% overlap between questions |
| `token_length_overlap.png` | Context length overlap | ~41% overlap across token lengths |
| `method_overlap.png` | Method agreement heatmap | ~12% overlap between methods |
| `model_overlap.png` | Instruct vs Base overlap | ~39% overlap between models |
| `universal_heads.png` | Top 30 universal heads (all methods) | Aggregated across 96 configs |
| `universal_heads_by_method.png` | Universal heads per method | Each method has different top heads |
| `layer_distribution_by_method.png` | Layer preferences by method | Methods focus on different layers |

---

## Key Findings Summary

### Eric's Research Questions

1. **Q1: Equal impact?** No—Inc. Year shows 42.6% drop at 50 heads, Inc. State only 26%
2. **Q2: Numerical vs categorical similarity?** ~65-73% overlap within category, ~67% between
3. **Q3: Equal effectiveness?** No—numerical questions show larger drops (19.6% vs 15.9%)
4. **Q4: Patterns?** Yes—each question type has distinct layer preferences

### Deep Exploration Findings

1. **Only 2 heads agree across all methods:** L20H14 and L14H31
2. **Methods identify different layers:** Summed Attn (13-14, 20-21), Wu24 (distributed), QRHead (9-10, 14-16)
3. **Question specialists exist:** L16H9 is rank #2 for HQ.State but rank ~423 for others
4. **Instruction tuning changes everything:** Some heads change rank by 600+ positions
5. **Some heads are wildly volatile:** L31H14 swings from rank #1 to #1024
6. **Wu24 has extreme #1 dominance:** 24.3% gap between #1 and #2 (vs ~3.5% for others)
7. **Method overlap is only ~12%:** The methods capture fundamentally different phenomena

---

## What Worked Well

1. **Hook-based ablation:** Consistent mechanism across all methods
2. **Incremental ablation levels:** Creates clear curves for Figure 8
3. **Train/test split:** Prevents overfitting in head identification
4. **One-word answer format:** Enables consistent evaluation

## Challenges Encountered

### Write Tool Size Limits

**Problem:** Initial attempts to write the `deep_exploration.py` script failed due to file size limits in the Write tool.

**Solution:** Used Shell with heredoc (`cat > file << 'EOF'`) to write larger scripts.

### sklearn Import Error

**Problem:** sklearn's `train_test_split` failed due to numpy version conflict.

**Solution:** Implemented manual shuffle-based split using Python's `random` module.

### Memory at Long Contexts

**Problem:** Full attention matrices caused OOM at 6K/8K tokens.

**Solution:** Forward hooks compute attention only for last token, avoiding O(N²) memory.

---

## Final Report

See **`FINDINGS.md`** for the complete, comprehensive findings report with all figure references.

# Retrieval Head Analysis Deliverable

## Overview

This deliverable contains a comprehensive study comparing three attention head detection methods for identifying "retrieval heads" in Llama-3-8B models. The study validates that all three methods identify causally important heads through ablation experiments.

**Key Finding:** While all methods successfully identify heads that matter for retrieval (validated by ablation), they identify **largely different sets of heads** (only ~12% overlap), suggesting each method captures different aspects of "retrieval behavior."

## Folder Structure

```
deliverable_head_analysis/
├── README.md                          # This file
├── PLAN.md                            # Original methodology plan
│
├── phase1/                            # Data preparation (train/test split)
│   ├── README.md                      # Phase 1 documentation
│   ├── prepare_data.py                # Train/test split script
│   ├── train_samples.json             # 80% for head identification (200 samples)
│   ├── test_samples.json              # 20% for ablation (50 samples)
│   └── data_summary.json              # Split metadata
│
├── phase2/                            # Head identification (3 methods)
│   ├── README.md                      # Phase 2 overview
│   ├── summed_attention/              # Method 1: Sum of attention to needle
│   │   ├── README.md                  # Method documentation
│   │   ├── run_detection.py           # Detection script
│   │   ├── run_all.py                 # Batch runner
│   │   └── results/                   # 32/32 experiments complete
│   ├── retrieval_head_wu24/           # Method 2: Wu24 copy-based detection
│   │   ├── README.md                  # Method documentation  
│   │   ├── run_detection.py           # Detection script
│   │   ├── run_all.py                 # Batch runner
│   │   └── results/                   # 32/32 experiments complete
│   └── qrhead/                        # Method 3: Query-relevant head detection
│       ├── README.md                  # Method documentation
│       ├── run_detection.py           # Detection script
│       ├── run_all.py                 # Batch runner
│       └── results/                   # 32/32 experiments complete
│
├── phase3/                            # Ablation study (3 methods)
│   ├── README.md                      # Phase 3 overview
│   ├── summed_attention/              # Ablation using Method 1 heads
│   │   ├── README.md
│   │   ├── run_ablation.py
│   │   └── results/                   # 32/32 experiments complete
│   ├── retrieval_head_wu24/           # Ablation using Method 2 heads
│   │   ├── README.md
│   │   ├── run_ablation.py
│   │   └── results/                   # 32/32 experiments complete
│   └── qrhead/                        # Ablation using Method 3 heads
│       ├── README.md
│       ├── run_ablation.py
│       └── results/                   # 32/32 experiments complete
│
└── phase4/                            # Analysis & figures
    ├── README.md                      # Phase 4 documentation
    ├── FINDINGS.md                    # Comprehensive findings report
    ├── figures/                       # Main analysis figures (13 files)
    ├── exploration_figures/           # Deep exploration figures (14 files)
    └── analysis/                      # Analysis scripts
        ├── run_all_analysis.py        # Master script
        ├── generate_figure8.py        # Figure 8 replications
        ├── research_questions.py      # Q1-Q4 visualizations
        ├── generate_findings.py       # FINDINGS.md generator
        ├── deep_exploration.py        # Head overlap analysis
        ├── deep_findings.py           # Deep dive analysis
        └── key_findings_viz.py        # Key finding visualizations
```

## Experiment Matrix

| Model | Method | Questions | Token Lengths |
|-------|--------|-----------|---------------|
| Llama-3-8B-Instruct | Summed Attention | Inc State, Inc Year, Employee Count, HQ State | 2K, 4K, 6K, 8K |
| Llama-3-8B-Instruct | Retrieval Head (Wu24) | Inc State, Inc Year, Employee Count, HQ State | 2K, 4K, 6K, 8K |
| Llama-3-8B-Instruct | QRHead | Inc State, Inc Year, Employee Count, HQ State | 2K, 4K, 6K, 8K |
| Llama-3-8B (base) | Summed Attention | Inc State, Inc Year, Employee Count, HQ State | 2K, 4K, 6K, 8K |
| Llama-3-8B (base) | Retrieval Head (Wu24) | Inc State, Inc Year, Employee Count, HQ State | 2K, 4K, 6K, 8K |
| Llama-3-8B (base) | QRHead | Inc State, Inc Year, Employee Count, HQ State | 2K, 4K, 6K, 8K |

**Total Experiments:**
- Phase 2: 2 models × 3 methods × 4 questions × 4 token lengths = **96 head identification runs**
- Phase 3: 2 models × 3 methods × 4 questions × 4 token lengths × 6 ablation levels = **576 ablation conditions**

## Key Results Summary

### 1. All Methods Identify Causally Important Heads

When ablating top-50 heads identified by each method, accuracy drops significantly compared to random baseline:

| Method | Accuracy Drop @ 50 Heads | Random Baseline |
|--------|-------------------------|-----------------|
| Summed Attention | 30.8% | ~5% |
| Wu24 Retrieval Head | 30.9% | ~5% |
| QRHead | 30.8% | ~5% |

### 2. Methods Identify Different Heads (~12% Overlap)

Despite all methods "working," they identify largely different head sets:

| Dimension | Jaccard Similarity |
|-----------|-------------------|
| Question overlap | 44.9% |
| Token length overlap | 40.7% |
| Model overlap (Instruct vs Base) | 39.1% |
| **Method overlap** | **12.0%** |

Only **2 heads** (L20H14 and L14H31) are in the top-100 for ALL three methods.

### 3. Question Type Matters

Numerical questions show larger ablation effects than categorical:

| Question | Avg Accuracy Drop |
|----------|------------------|
| Inc Year (numerical) | 30.9% |
| Employee Count (numerical) | 15.3% |
| HQ State (categorical) | 23.6% |
| Inc State (categorical) | 7.6% |

## Quick Start

```bash
# Phase 1: Prepare data (creates train/test split)
python phase1/prepare_data.py

# Phase 2: Run head identification
python phase2/summed_attention/run_all.py
python phase2/retrieval_head_wu24/run_all.py
python phase2/qrhead/run_all.py

# Phase 3: Run ablation study
python phase3/summed_attention/run_all.py
python phase3/retrieval_head_wu24/run_all.py
python phase3/qrhead/run_all.py

# Phase 4: Generate analysis and figures
python phase4/analysis/run_all_analysis.py
```

## Prompt Format

All experiments use the same needle-in-haystack structure with **one-word answer requirement**:

```
[Haystack padding - Alice in Wonderland text]
[Section 1 content - the "needle" containing GT answer]
[Haystack padding - Alice in Wonderland text]

Question: {question}
Answer in one word:
```

**Important constraints:**
- `max_new_tokens=10` - Model generates at most 10 tokens
- Answers are normalized for comparison (lowercase, strip punctuation)
- One-word answers ensure consistent evaluation across all methods

## Data Source

- **Ground truth:** `../edgar_gt_verified_slim.csv` (250 SEC 10-K filings with verified answers)
- **Questions:** 4 types (Inc State, Inc Year, Employee Count, HQ State)
- **Haystack:** Alice in Wonderland text for padding
- **Needle:** Entire Section 1 from each SEC filing

## Method Comparison

| Aspect | Summed Attention | Wu24 | QRHead |
|--------|------------------|------|--------|
| **When computed** | Encoding | Decoding | Encoding |
| **What's measured** | Sum of attention to needle | argmax + token match | Query-calibrated attention |
| **Success filter** | None | ROUGE > 50% | None |
| **Unique feature** | Simple, fast | Captures copy behavior | Calibration removes query-independent attention |
| **Top universal head** | L20H14 | L24H27 | L14H31 |

## For More Details

- **FINDINGS.md:** Comprehensive findings with all figures referenced
- **Phase-specific READMEs:** Detailed documentation for each phase
- **Method-specific READMEs:** Algorithm details and implementation notes

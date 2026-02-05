# Retrieval Head Analysis Deliverable

## Folder Structure

```
deliverable_head_analysis/
├── PLAN.md                          # Detailed methodology
├── README.md                        # This file
│
├── phase1/                          # Data preparation
│   ├── README.md
│   ├── prepare_data.py              # Train/test split
│   ├── train_samples.json           # 80% for head identification
│   └── test_samples.json            # 20% for ablation
│
├── phase2/                          # Head identification (3 methods)
│   ├── summed_attention/
│   │   ├── README.md
│   │   ├── run_detection.py         # Our method
│   │   └── results/                 # Head scores per question
│   ├── retrieval_head_wu24/
│   │   ├── README.md
│   │   ├── run_detection.py         # Adapted from WU_Retrieval_Head
│   │   └── results/
│   └── qrhead/
│       ├── README.md
│       ├── run_detection.py         # Adapted from QRHead
│       └── results/
│
├── phase3/                          # Ablation study (3 methods)
│   ├── summed_attention/
│   │   ├── README.md
│   │   ├── run_ablation.py
│   │   └── results/
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
    ├── figures/                     # All plots
    ├── analysis/                    # Analysis scripts
    └── FINDINGS.md                  # Final report
```

## Experiment Matrix

| Model | Method | Questions |
|-------|--------|-----------|
| Llama-3-8B-Instruct | Summed Attention | Inc State, Inc Year, Employee Count, HQ State |
| Llama-3-8B-Instruct | Retrieval Head (Wu24) | Inc State, Inc Year, Employee Count, HQ State |
| Llama-3-8B-Instruct | QRHead | Inc State, Inc Year, Employee Count, HQ State |
| Llama-3-8B (base) | Summed Attention | Inc State, Inc Year, Employee Count, HQ State |
| Llama-3-8B (base) | Retrieval Head (Wu24) | Inc State, Inc Year, Employee Count, HQ State |
| Llama-3-8B (base) | QRHead | Inc State, Inc Year, Employee Count, HQ State |

**Total: 2 models × 3 methods × 4 questions = 24 experiments**

## Quick Start

```bash
# Phase 1: Prepare data
python phase1/prepare_data.py

# Phase 2: Run head identification (pick one method)
python phase2/summed_attention/run_detection.py --model llama_instruct --question inc_state
python phase2/retrieval_head_wu24/run_detection.py --model llama_instruct --question inc_state
python phase2/qrhead/run_detection.py --model llama_instruct --question inc_state

# Phase 3: Run ablation
python phase3/summed_attention/run_ablation.py --model llama_instruct --question inc_state

# Phase 4: Generate figures
python phase4/analysis/generate_figures.py
```

## Prompt Format

All experiments use the same prompt structure with **one-word answer requirement**:

```
[Haystack padding - Alice in Wonderland text]
[Section 1 content - the "needle"]
[Haystack padding - Alice in Wonderland text]

Question: {question}
Answer in one word:
```

**Important constraints:**
- `max_new_tokens=10` - Model generates at most 10 tokens (sufficient for one-word answers)
- Answers are normalized for comparison (lowercase, strip punctuation, handle numeric formats)
- One-word answers ensure consistent evaluation across all methods

## Data Source

Ground truth: `../edgar_gt_verified_slim.csv` (250 samples, all 4 GT categories included)

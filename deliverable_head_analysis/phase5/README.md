# Phase 5: Research Validation & Circuit Analysis

## Purpose

Validate the most significant findings from Phases 2-4 and conduct follow-up experiments needed to support a top-venue publication. This phase does NOT modify any existing Phase 1-4 code or results.

## Key Documents

| File | Description |
|------|-------------|
| `RESEARCH_SYNTHESIS.md` | Comprehensive synthesis of all findings with publication-readiness assessment |
| `RESEARCH_PLAN.md` | Living plan with decision tree, timelines, and status tracking |

## Sub-Phases

### A. Statistical Validation (`statistical_validation/`)

**Goal:** Determine if non-monotonic ablation curves (recovery → collapse) are statistically real or sampling noise.

**Why:** With n=32-33 test samples per config, a single sample flipping changes accuracy by ~3%. The "recovery" pattern could be noise. This is the gatekeeper: it determines which paper we write.

| File | Description |
|------|-------------|
| `run_ablation_with_samples.py` | Modified ablation script that saves **per-sample** results (filename, generated, correct) for every ablation level. Supports all 3 methods via `--method` flag. |
| `run_critical_configs.py` | Runs only the 3 critical non-monotonic configs |
| `bootstrap_validation.py` | Bootstrap resampling (10K iterations) with 95% CIs and non-monotonicity significance test. **No GPU required.** |
| `results/` | Output with per-sample data |

**The 3 critical configs:**
1. `summed_attention / base / inc_year / 2048` — 87.5% drop at 5 heads, recovery at 20
2. `summed_attention / instruct / inc_year / 2048` — 68.8% drop at 10, recovery at 30
3. `qrhead / instruct / inc_year / 2048` — 50% drop at 5, recovery at 30

**Usage:**
```bash
cd phase5/statistical_validation/

# Step 1: Run ablation with per-sample logging (GPU required)
python run_critical_configs.py

# Step 2: Bootstrap validation (no GPU)
python bootstrap_validation.py
```

### B. Circuit Analysis (`circuit_analysis/`) — PENDING

**Prerequisite:** Phase A confirms non-monotonicity is statistically significant.

**Goal:** Show that when primary retrieval heads are ablated, backup heads increase their attention to the needle (compensatory activation).

Scripts will be added here when Phase A results are in.

---

## Relationship to Other Phases

- **Phase 1:** Uses same train/test split and data
- **Phase 2:** Loads head rankings from Phase 2 results (unchanged)
- **Phase 3:** Same ablation logic, but saves per-sample data. Phase 3 results are untouched.
- **Phase 4:** Findings inform what we validate here

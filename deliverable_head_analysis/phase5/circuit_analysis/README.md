# Phase 5C: Circuit Analysis — Attention Comparison (Clean vs Ablated)

## Purpose

Tests the **backup circuit hypothesis**: when the top-5 "primary" retrieval heads are ablated, do other heads (ranked 6–20 in Phase 2) increase their attention to the needle, compensating for the lost circuit?

This experiment directly explains the **non-monotonic ablation curve** validated in Phase A (bootstrap analysis):
- At 5 heads ablated: accuracy drops (primary circuit removed)
- At 20 heads ablated: accuracy **recovers** (why? backup circuit engaged)
- At 30 heads ablated: accuracy collapses again (backup circuit also removed)

## Experiment Design

For each test sample, two forward passes:

| Pass | Hooks Active | What's Recorded |
|------|-------------|-----------------|
| **Clean** | None | `attention_to_needle[head]` for all 1024 heads |
| **Ablated** | Top-5 heads zeroed via o_proj pre-hook | Same metric, all 1024 heads |

Then:
- Δ[head] = ablated_attention − clean_attention
- Positive Δ → head **increased** attention to needle (potential backup)
- Negative Δ → head decreased attention (cascade failure)

## Configs

| # | Config | Why |
|---|--------|-----|
| 1 | summed_attention / base / inc_year / 2048, ablate=5 | Primary: strongest non-monotonic signal |
| 2 | Same config, ablate=10 | Trace behavior just before recovery |
| 3 | Same config, ablate=20 | The recovery point — are backup heads now ablated too? |

## Running

```bash
# Run all 3 experiments
python run_circuit_analysis.py

# Or run individually
python attention_comparison.py --method summed_attention --model base --question inc_year --tokens 2048 --ablate-top-n 5
```

## Expected Outcomes

**If backup circuit hypothesis is supported:**
- Heads ranked 6-20 show significantly positive Δ (attend MORE to needle under ablation)
- These same heads, when also ablated (ablate=20 run), show the system losing its backup
- Unranked heads (50+) show near-zero Δ

**If backup circuit hypothesis is rejected:**
- Δ is uniformly small or random — no heads compensate
- Recovery was due to some other mechanism (e.g., layer norm redistribution)

## Output

Results saved to `results/llama3_base/inc_year/` as JSON files containing:
- Per-head Δ values (all 1024 heads)
- Group-level summary (primary, backup, unranked)
- Cross-reference with Phase 2 rankings
- Verdict on backup hypothesis

# Research Plan: Dynamic Retrieval Circuits in Transformers

**Created:** 2026-02-07
**Last Updated:** 2026-02-13
**Status:** Phase A COMPLETE → Phase C in progress (Circuit Analysis)

---

## Thesis (Working)

> Retrieval in transformers is not implemented by fixed "retrieval heads" — it is mediated by dynamic, redundant circuit ensembles that reorganize under perturbation and across context lengths.

This challenges the framing of Wu et al. 2024 and similar work that treats retrieval heads as static model properties.

---

## The Evidence We Have

| Finding | Strength | Robustness Risk |
|---------|----------|-----------------|
| Non-monotonic ablation (recovery → collapse) | **CONFIRMED (Config 1)** | ~~HIGH RISK~~ → **VALIDATED** by bootstrap (50pt CI gap) |
| Context-length circuit replacement (Jaccard=0.026) | Very strong | **LOW RISK** — computed over 128+ training samples |
| 12% inter-method overlap, all causally validated | Very strong | **LOW RISK** — computed over full training set |
| 5 heads → 87.5% drop (extreme concentration) | Strong | MEDIUM — single config, small test set |
| Instruction tuning redistributes retrieval | Supporting | LOW RISK — consistent across configs |
| Question-type specialization | Supporting | LOW RISK — consistent pattern |

**Critical insight:** ~~The non-monotonic finding determines which paper we write. Validate it first.~~ **VALIDATED.** Config 1 non-monotonicity is statistically significant (Phase A complete). Proceeding with backup circuit paper framing (Scenario 1).

---

## Phase A: Statistical Validation of Non-Monotonicity

**Goal:** Determine if the recovery-then-collapse ablation pattern is real or noise.
**GPU Required:** No (resampling existing data)
**Estimated Effort:** ~1 day

### Method: Bootstrap Confidence Intervals

Resample the per-sample correct/incorrect results (n=32-33) with replacement, 10,000 times, to get confidence intervals on accuracy at each ablation level.

**What we checked:**
- [x] Do Phase 3 JSON files store per-sample results? **NO — only aggregates.**
  - `evaluate_accuracy()` collects per-sample `(filename, gt_value, generated, correct)` but only saves aggregate `accuracy`, `correct`, `total` to JSON.
  - **Action needed:** Modify `run_ablation.py` to include per-sample results in the JSON, then re-run only the 3 critical configs. Small code change, but requires GPU for re-run.

### Configs to Validate

The three most dramatic non-monotonic cases:

| # | Config | Pattern |
|---|--------|---------|
| 1 | Summed Attention / Base / inc_year / 2048 | 9.4% → 9.4% → **84.4%** → 6.3% → 9.4% → 9.4% |
| 2 | Summed Attention / Instruct / inc_year / 2048 | 93.8% → 25.0% → 53.1% → **87.5%** → 75.0% → 62.5% |
| 3 | QRHead / Instruct / inc_year / 2048 | 43.8% → 59.4% → 65.6% → **84.4%** → 43.8% → 34.4% |

### Success Criteria

- **REAL:** The recovery point's 95% CI does NOT overlap with the collapse points' CIs on both sides. (e.g., accuracy at 20 heads is significantly higher than at 10 AND at 30.)
- **NOISE:** CIs overlap substantially → the "recovery" is within sampling error.

### Decision

- **If REAL →** Proceed to Phase C (circuit analysis). Non-monotonicity anchors the paper.
- **If NOISE →** Skip Phase C. Paper pivots to context-length circuit replacement + method disagreement framing. Both are statistically robust (computed over 128+ samples).

### Implementation Steps

1. **Modify `run_ablation.py`** (all 3 methods) to save per-sample results in JSON output
   - Add the per-sample `results` list (filename, gt_value, generated, correct) to each ablation level's entry
   - Small change: ~5 lines per script
2. **Re-run only the 3 critical configs** (GPU required):
   - `phase3/summed_attention/run_ablation.py --model base --question inc_year --tokens 2048`
   - `phase3/summed_attention/run_ablation.py --model instruct --question inc_year --tokens 2048`
   - `phase3/qrhead/run_ablation.py --model instruct --question inc_year --tokens 2048`
3. **Write bootstrap script** — resamples per-sample results 10,000 times
4. **Run bootstrap and interpret** → decide paper direction

### Status

- [x] Check if per-sample data exists in Phase 3 JSONs → **NO, needs code change + re-run**
- [x] Modify run_ablation.py to save per-sample results → `phase5/statistical_validation/run_ablation_with_samples.py`
- [x] Re-run 3 critical configs (GPU) → COMPLETE
- [x] Write bootstrap script → `phase5/statistical_validation/bootstrap_validation.py`
- [x] Run bootstrap on 3 configs → COMPLETE
- [x] Interpret results → **DECISION: Non-monotonicity is REAL. Proceed with circuit analysis.**

### Bootstrap Results (2026-02-13)

**Config 1 (summed_attention / base / inc_year / 2048) — SIGNIFICANT:**
```
 5 heads:  9.4%  CI [0.0%, 21.9%]     ← collapse
10 heads:  9.4%  CI [0.0%, 21.9%]     ← still collapsed  
20 heads: 84.4%  CI [71.9%, 96.9%]    ← RECOVERY (50pt CI gap vs neighbors)
30 heads:  6.3%  CI [0.0%, 15.6%]     ← collapse again
```
Recovery at 20 heads: significant in BOTH directions (CI gap = 50.0% vs prev, 56.3% vs next).

**Config 2 (summed_attention / instruct / inc_year / 2048) — PARTIAL:**
Recovery at 30 heads significant vs previous (CI gap = 6.3%) but NOT vs next (CIs overlap).

**Config 3 (qrhead / instruct / inc_year / 2048) — PARTIAL:**
Recovery at 30 heads significant vs next (CI gap = 9.4%) but NOT vs previous (CIs overlap).

---

## Phase B: Second Model Family (Mistral-7B)

**Goal:** Show findings aren't Llama-specific.
**GPU Required:** Yes
**Estimated Effort:** ~2-3 days (pipeline adaptation + runs)

### Model Choice: Mistral-7B-v0.3

**Why Mistral:**
- Same scale as Llama 3 8B (7B params)
- Uses GQA (like Llama 3) → pipeline ports with minimal changes
- Different pretraining data/team → genuine replication
- Widely recognized in the community

**Rejected alternatives:**
- Qwen-2.5-7B — good diversity pick, could add as third model if time permits
- Gemma-2-9B — slightly different scale, more adaptation needed
- Phi-3-mini (3.8B) — too different in scale to be a fair comparison

### What to Run (Targeted Subset)

We don't need the full 96-experiment matrix. Focus on the strongest-signal config:

**Phase 2 (head identification) — 6 runs:**

| Method | Question | Tokens |
|--------|----------|--------|
| Summed Attention | inc_year | 2048 |
| Summed Attention | inc_year | 8192 |
| Wu24 | inc_year | 2048 |
| Wu24 | inc_year | 8192 |
| QRHead | inc_year | 2048 |
| QRHead | inc_year | 8192 |

**Phase 3 (ablation) — 6 runs:**
Same 6 configs as above.

**What this tests:**
- Does context-length circuit replacement happen on Mistral? (compare 2048 vs 8192 head rankings)
- Is extreme head concentration Llama-specific? (ablation curve shape)
- Do the three methods still disagree on Mistral? (cross-method overlap)
- If Phase A confirms non-monotonicity: does the recovery pattern appear on Mistral?

### Pipeline Adaptation Needed

- [ ] Update model loading code for Mistral architecture
- [ ] Verify GQA head expansion logic works for Mistral's KV head count
- [ ] Verify tokenizer compatibility (BOS token, padding, etc.)
- [ ] Test on single sample before full run
- [ ] Run Phase 2 (6 experiments)
- [ ] Run Phase 3 (6 experiments)
- [ ] Compare head rankings to Llama 3 results

### Status

- [ ] Pipeline adaptation
- [ ] Phase 2 runs
- [ ] Phase 3 runs
- [ ] Analysis and comparison

---

## Phase C: Circuit Analysis (Backup Circuit Validation)

**Goal:** Show that when primary retrieval heads are ablated, other heads change their behavior to compensate.
**GPU Required:** Yes
**Estimated Effort:** ~1-2 days
**Prerequisite:** Phase A confirms non-monotonicity is real.

### Why NOT Full Circuit Tracing

The Neel Nanda / Anthropic "Mathematical Framework for Transformer Circuits" gives us the conceptual vocabulary — residual stream, heads as readers/writers, composition. But their full framework is designed for understanding complete circuits from scratch in small models. We don't need that.

We have a much more targeted question: **do specific heads change behavior when other heads are ablated?** That requires a simpler experiment.

### The Experiment: Attention Pattern Comparison (Clean vs. Ablated)

**Technique chosen:** Direct attention comparison — record how every head attends to the needle with and without primary heads ablated.

**Why this over alternatives:**

| Technique | What It Shows | Complexity | Our Pick? |
|-----------|--------------|------------|-----------|
| **Attention pattern comparison** | Which heads change where they attend | Low | **YES — start here** |
| Activation patching | Causal role of specific activations | Medium | Maybe as follow-up |
| Logit lens | What info is represented at each layer | Medium | Useful supporting evidence |
| ACDC (Conmy et al. 2023) | Automated full circuit map | Very high | Overkill for our question |

### Concrete Steps

**Config:** Base / inc_year / 2048 (the most extreme case: 5 heads → 87.5% drop, then recovery at 20).

For each test sample:

```
Step 1: CLEAN forward pass (no ablation)
  → Record: attention_to_needle[head] for all 1024 heads
  → Specifically: attn[head, last_token, needle_start:needle_end].sum()

Step 2: ABLATED forward pass (top 5 heads zeroed)
  → Record: attention_to_needle[head] for all remaining 1019 heads

Step 3: Compute per-head change
  → Δ[head] = ablated_attention[head] - clean_attention[head]
  → Positive Δ = head INCREASED attention to needle (potential backup)
  → Negative Δ = head decreased attention (cascade failure)

Step 4: Average Δ across all test samples

Step 5: Check if high-Δ heads correspond to heads ranked 6-20 
         in Phase 2 (the "recovery" window)
```

### What Would Confirm the Backup Circuit Hypothesis

- Heads ranked ~6-20 in Phase 2 show significantly positive Δ (they attend MORE to needle when heads 1-5 are ablated)
- These same heads, when also ablated (the 20→30 head level), cause the second collapse
- Heads ranked 50+ show near-zero Δ (uninvolved heads don't change)

### What Would Reject It

- Δ is uniformly small or random across all heads → no compensation is happening
- The "recovery" heads don't correspond to any identifiable pattern → noise

### Potential Follow-Up: Logit Attribution

If attention comparison confirms backup heads exist, we could also:
- Use logit lens to check if the correct answer appears in intermediate representations at the backup heads' layers
- This would show the backup heads aren't just attending to the needle — they're actually routing the answer information

### Status

- [x] Waiting on Phase A result → **CONFIRMED, proceeding**
- [x] Write attention recording code → `phase5/circuit_analysis/attention_comparison.py`
- [ ] Run clean vs. ablated comparison (GPU required)
- [ ] Analyze Δ patterns
- [ ] Map backup heads to Phase 2 rankings
- [ ] (Optional) Logit lens follow-up

---

## Phase D: Paper Writing

### Which Paper We Write (Depends on Phases A-C)

**Scenario 1: Non-monotonicity confirmed + backup circuits found**

> **"Redundant Retrieval Circuits in Transformers: Evidence for Dynamic Compensation"**

- Lead: backup circuit activation under ablation (Phase C)
- Section 2: context-length circuit replacement (existing data)
- Section 3: method disagreement as evidence of multiple circuits (existing data)
- Section 4: Mistral replication (Phase B)
- Target: **NeurIPS or ICLR** (mechanistic interpretability)

**Scenario 2: Non-monotonicity confirmed but no clear backup mechanism**

> **"Retrieval Circuits in Transformers Are Dynamic, Not Fixed"**

- Lead: context-length circuit replacement (Jaccard=0.026)
- Section 2: non-monotonic ablation as evidence of redundancy
- Section 3: method disagreement
- Section 4: Mistral replication
- Target: **ICLR or EMNLP**

**Scenario 3: Non-monotonicity is noise**

> **"Context-Dependent Retrieval Circuits Challenge the Retrieval Head Hypothesis"**

- Lead: context-length circuit replacement
- Section 2: 12% method overlap despite equal causal validation
- Section 3: extreme concentration + question-type specialization
- Section 4: Mistral replication
- Target: **EMNLP or ACL**

### Key Framing Rules (from ChatGPT's analysis, agreed)

- Do NOT frame as methods comparison
- Do NOT lead with "retrieval heads behave differently"
- DO frame as architectural claim about how transformers organize retrieval
- DO use "circuits" not "heads" as the primary unit of analysis
- Findings about question-type specialization and instruction tuning are supporting sections, not the thesis

### Status

- [ ] Waiting on Phases A-C to determine paper direction
- [ ] Outline
- [ ] Draft
- [ ] Figures
- [ ] Submission target and deadline

---

## Execution Order

```
Week 1:
  ├── Day 1-2: Phase A (bootstrap validation)
  │     └── Decision point: is non-monotonicity real?
  │
  ├── Day 2-3: Phase B (Mistral pipeline adaptation)
  │     └── Start runs while analyzing Phase A
  │
  └── Day 3-5: Phase B runs complete
        └── Compare Mistral vs Llama results

Week 2 (if non-monotonicity is real):
  ├── Day 1-2: Phase C (circuit analysis)
  │     └── Attention comparison experiment
  │
  ├── Day 3: Analyze circuit results
  │     └── Decision point: paper framing finalized
  │
  └── Day 4-5: Begin Phase D (paper outline + key figures)

Week 2 (if non-monotonicity is noise):
  ├── Day 1: Finalize paper framing (Scenario 3)
  ├── Day 2-3: Additional analysis on context-length findings
  └── Day 4-5: Begin Phase D
```

---

## Open Questions

1. **Which conference deadline are we targeting?** This determines timeline urgency.
2. **GPU budget/availability?** Phase B and C need GPU time. How much do we have?
3. **Should we add Qwen as a third model?** Strengthens generalizability but costs more time.
4. **Do we need to run the SEC filing needle-in-haystack on Mistral, or could we also test on a generic benchmark (e.g., RULER, Needle-in-a-Haystack) for broader appeal?**
5. **Co-author situation?** Affects writing timeline and division of labor.

---

## Log

| Date | Action | Result |
|------|--------|--------|
| 2026-02-07 | Created research plan | — |
| 2026-02-07 | Checked Phase 3 JSONs for per-sample data | **NOT STORED** — only aggregates saved. Code change + re-run needed. |
| 2026-02-13 | Phase A complete: bootstrap validation run | **Config 1 non-monotonicity CONFIRMED** (50pt CI gap). Configs 2,3 partially significant. |
| 2026-02-13 | Decision: proceed with circuit analysis | Paper direction = Scenario 1 (backup circuit framing) |
| 2026-02-13 | Phase C code written | `circuit_analysis/attention_comparison.py` — ready for GPU run |
| | | |

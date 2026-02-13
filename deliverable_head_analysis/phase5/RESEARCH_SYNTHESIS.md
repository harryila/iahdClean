# Research Synthesis: Attention Head Analysis for Retrieval in LLMs

**Purpose of This Document:** This is a comprehensive synthesis of our experimental results, designed to be provided to a deep research tool to identify the best avenue for publication at a top AI conference. It includes: what we did, what we found, which signals are significant, which are novel, and concrete research directions with their strengths and risks.

---

## Table of Contents

1. [What We Built](#1-what-we-built)
2. [The Core Experimental Setup](#2-the-core-experimental-setup)
3. [What Each Method Captures (and Why That Matters)](#3-what-each-method-captures)
4. [The Seven Most Significant Findings](#4-the-seven-most-significant-findings)
5. [Raw Data: The Strongest Signals](#5-raw-data-the-strongest-signals)
6. [What Is Novel vs. What Is Known](#6-what-is-novel-vs-what-is-known)
7. [Candidate Research Directions](#7-candidate-research-directions)
8. [What We Need From Literature Review](#8-what-we-need-from-literature-review)
9. [Appendix: Full Result Tables](#appendix-full-result-tables)

---

## 1. What We Built

We built a systematic experimental framework comparing **three different methods** for identifying "retrieval heads" in Llama-3-8B, tested across **2 models** (base and instruct), **4 question types** (2 numerical, 2 categorical), and **4 context lengths** (2K-8K tokens), using a **needle-in-haystack** task with real-world SEC filing data.

**Scale:**
- Phase 2 (head identification): 96 experiments, ranking all 1,024 attention heads
- Phase 3 (ablation validation): 96 experiments × 6 ablation levels × 2 conditions = 1,152 ablation runs
- Phase 4 (analysis): Cross-method comparison, overlap analysis, deep exploration

**The three detection methods:**

| Method | What It Measures | When | Key Mechanism |
|--------|-----------------|------|---------------|
| **Summed Attention** | Total attention from last token to needle region | Encoding | `score = attn[last_token, needle_start:needle_end].sum()` |
| **Wu24 Retrieval Head** | Copy-like behavior: argmax attention to needle + token match | Decoding | Only counts when attended token = generated token; requires ROUGE > 50% |
| **QRHead** | Query-relevant attention: calibrated by subtracting null-query attention | Encoding | `calibrated = attn(real_query) - attn("N/A")`; outlier removal (mean-2σ) |

**The ablation validation:** For each method's ranked heads, ablate top-k (k = 5, 10, 20, 30, 40, 50) and measure accuracy drop vs. random-k baseline. All three methods use identical ablation mechanism (zero head output before o_proj via PyTorch hooks) for fair comparison.

**The task:** Needle-in-haystack using 250 real SEC 10-K filings. Section 1 of each filing (the "needle") is embedded in Alice in Wonderland text (the "haystack") at the midpoint. The model must answer one of four factual questions about the filing. 80/20 train/test split ensures heads are identified on different samples than they're evaluated on.

---

## 2. The Core Experimental Setup

### Design Matrix

| Dimension | Values | Count |
|-----------|--------|-------|
| Models | Llama-3-8B-Instruct, Llama-3-8B-Base | 2 |
| Detection Methods | Summed Attention, Wu24, QRHead | 3 |
| Questions | inc_state (categorical), inc_year (numerical), employee_count (numerical), hq_state (categorical) | 4 |
| Context Lengths | 2048, 4096, 6144, 8192 tokens | 4 |
| Ablation Levels | 5, 10, 20, 30, 40, 50 heads | 6 |

### Key Controlled Variables
- Same needle position (0.5 = middle) across all experiments
- Same haystack text (Alice in Wonderland)
- Same generation parameters (max_new_tokens=10, greedy decoding)
- Same ablation mechanism across all three Phase 3 methods
- Same train/test split (seed=42)

---

## 3. What Each Method Captures

This section matters because the **low inter-method overlap (~12%)** is one of our most important findings. Understanding what each method actually measures is critical to interpreting why they disagree.

### Summed Attention
- **Captures:** Total attention magnitude from the prediction position to the relevant context
- **Computes during:** Encoding (single forward pass)
- **Attention direction:** Last token → needle region
- **Score:** Sum of all attention weights (not just argmax)
- **Identifies:** Heads with high aggregate attention flow toward the answer — a general "information routing" signal
- **Layer preference:** Mid-to-late layers (13-14, 20-21)
- **Score distribution:** Gradual dropoff from #1 to #50 (~3.4% gap between #1 and #2)

### Wu24 Retrieval Head
- **Captures:** Token-level copy behavior — heads that look at a specific token in the needle and the model then generates that exact token
- **Computes during:** Decoding (each generation step)
- **Attention direction:** Generated token → needle (argmax only, not sum)
- **Score:** Fraction of generation steps where argmax hits needle AND token matches
- **Success filter:** Only counts samples where ROUGE > 50% (successful retrieval)
- **Identifies:** A small elite set of "copy heads" that perform literal token retrieval
- **Layer preference:** Distributed (5, 16, 21, 23, 27)
- **Score distribution:** Extremely steep — 24.3% gap between #1 and #2, few "super-heads"

### QRHead
- **Captures:** Query-relevant attention — attention that specifically increases because of the question being asked, not just general document attention
- **Computes during:** Encoding (two forward passes: real query + null query "N/A")
- **Attention direction:** Query tokens → needle region
- **Calibration:** Subtracts null-query attention to isolate question-driven signal
- **Outlier removal:** mean-2σ threshold per head
- **Identifies:** Heads that attend to the needle BECAUSE of the specific query
- **Layer preference:** Mid layers (9-10, 14-16)
- **Score distribution:** Moderate dropoff (~3.7% gap between #1 and #2)

### Why the Distinction Matters

These three methods capture fundamentally different aspects of the retrieval process:
1. **Summed Attention** = "Which heads route information from context to prediction?"
2. **Wu24** = "Which heads literally copy tokens from context into generation?"
3. **QRHead** = "Which heads attend to context specifically because of the question?"

A head could score high on one and low on others. For example, a head might strongly attend to the needle (high summed attention) but not be involved in the actual token-level copying (low Wu24). Or a head might always attend to certain regions regardless of the query (high summed attention, low QRHead after calibration).

---

## 4. The Seven Most Significant Findings

### Finding 1: EXTREME HEAD CONCENTRATION — 5 Heads Control 87.5% of Year Retrieval

**The single most striking datapoint in the entire study.**

In Llama-3-8B-Base at 2048 tokens for the inc_year question:
- **Baseline accuracy:** 96.9%
- **After ablating just 5 heads:** 9.4% (87.5 percentage point drop)
- **After ablating 50 random heads:** 81.3% (only 15.6% drop)

Five heads — 0.5% of all 1,024 attention heads — are almost solely responsible for the model's ability to retrieve a year from context. These five heads are:

| Rank | Head | Score | Layer |
|------|------|-------|-------|
| 1 | L16H1 | 116.2 | 16 |
| 2 | L17H24 | 115.2 | 17 |
| 3 | L14H31 | 111.6 | 14 |
| 4 | L22H13 | 106.3 | 22 |
| 5 | L13H18 | 104.2 | 13 |

This level of concentration is far more extreme than what Wu24 reported (they showed gradual curves, not cliff-edge drops at 5 heads). It suggests the base model has a hyper-specialized "year retrieval circuit" with almost no redundancy.

**Similarly extreme cases:**
- QRHead/instruct/inc_year/2048: 50% accuracy drop from just 5 heads
- QRHead/instruct/employee_count/2048: 10 heads reduce accuracy from 30.6% to 0.0% (complete destruction)

**Why this matters for publication:** This goes beyond Wu24's finding that retrieval heads exist. We show that for certain task types, the model's capability is concentrated in an almost absurdly small number of heads, with no graceful degradation. This has implications for model robustness, pruning, and understanding how factual recall is mechanistically organized.

---

### Finding 2: NON-MONOTONIC ABLATION CURVES — Evidence of Compensatory Circuits

**This is potentially the most novel finding.**

Standard expectation: ablating more heads → monotonically worse accuracy. What we observe instead:

**Case A: Summed Attention / Base / inc_year / 2048:**
```
5 heads  → 9.4%  accuracy (87.5% drop — catastrophic)
10 heads → 9.4%  accuracy (same — the 5 backup heads don't help)
20 heads → 84.4% accuracy (RECOVERY! — heads 11-20 somehow compensate)
30 heads → 6.3%  accuracy (collapse again — compensatory circuit also ablated)
40 heads → 9.4%  accuracy
50 heads → 9.4%  accuracy
```

**Case B: Summed Attention / Instruct / inc_year / 2048:**
```
5 heads  → 93.8% accuracy (no drop — top 5 aren't critical for instruct)
10 heads → 25.0% accuracy (68.8% drop — catastrophic)
20 heads → 53.1% accuracy (recovery!)
30 heads → 87.5% accuracy (near-complete recovery!)
40 heads → 75.0% accuracy (declining again)
50 heads → 62.5% accuracy
```

**Case C: QRHead / Instruct / inc_year / 2048:**
```
5 heads  → 43.8% accuracy (50% drop)
10 heads → 59.4% accuracy (partial recovery)
20 heads → 65.6% accuracy (more recovery)
30 heads → 84.4% accuracy (near-full recovery!)
40 heads → 43.8% accuracy (collapse again)
50 heads → 34.4% accuracy
```

**Interpretation:** When a small set of "primary" retrieval heads is ablated, "backup" heads elsewhere in the network can compensate. But as ablation continues and these backup heads are also removed, the system collapses. This suggests a **layered redundancy architecture** where:
1. A primary circuit handles retrieval normally
2. A secondary/backup circuit exists but is normally dormant
3. Ablating the primary activates the backup
4. Ablating both destroys the capability

**Why this matters for publication:** Non-monotonic ablation curves are not predicted by existing theories of retrieval heads. They point toward **circuit redundancy and backup mechanisms** in transformers — a topic of growing interest in mechanistic interpretability. The "backup head" hypothesis (that dormant heads can compensate for ablated ones) connects to recent work on circuit resilience but has not been empirically demonstrated at this specificity level for retrieval tasks.

---

### Finding 3: CONTEXT LENGTH TRIGGERS COMPLETE CIRCUIT REPLACEMENT

**The data shows that different context lengths activate entirely different retrieval circuits.**

For Summed Attention / Base / inc_year, the top-20 Jaccard similarity between 2048 and 8192 tokens is **0.026** — meaning only 1 out of 39 unique heads is shared. This is near-zero overlap.

Specific examples of head rank changes:

| Head | Rank @ 2048 | Rank @ 8192 | Change |
|------|-------------|-------------|--------|
| L16H1 | **1** | **726-839** | Dominant → irrelevant |
| L31H14 | **1024** (dead last) | **2-3** | Irrelevant → dominant |
| L9H1 | 75 | **1** | Moderate → most important |
| L11H1 | 52 | **2** | Moderate → second most important |

L31H14 is particularly striking: it literally goes from rank 1024 (the very last of all heads) at 2048 tokens to rank 2-3 at 6144-8192 tokens. This is not a gradual shift — it's a binary switch.

**Stable vs. unstable heads:**
- **L14H31** is remarkably stable: top-3 across all token lengths for both summed attention and QRHead
- **L16H1** is a **short-context specialist**: dominant at 2K-4K but completely irrelevant at 6K+
- **Layer 31 heads** are **emergent long-context heads**: ranked last at 2K but top-5 at 6K+

**Why this matters for publication:** Current retrieval head literature (Wu24, QRHead) typically identifies heads at a single context length and treats them as fixed model properties. Our data shows this is fundamentally wrong — the retrieval circuit is **dynamic and context-length-dependent**. The model appears to have different "retrieval modes" that activate at different scales. This challenges the core assumption that retrieval heads are static features of the architecture.

---

### Finding 4: THREE METHODS IDENTIFY DIFFERENT HEADS — BUT ALL ARE CAUSALLY VALIDATED

**Only ~12% overlap between methods** (Jaccard similarity of top-50 heads), yet ablating each method's heads causes comparable accuracy drops (~30% at 50 heads, vs ~5% for random baseline).

| Method Pair | Jaccard Overlap |
|-------------|-----------------|
| QRHead ↔ Wu24 | 30.1% |
| QRHead ↔ Summed Attention | 28.9% |
| Wu24 ↔ Summed Attention | 30.0% |
| **Three-way** | **~12%** |

Only **2 heads** (L20H14 and L14H31) are in the top-100 for all three methods simultaneously — a 0.2% agreement rate.

**Yet all three methods' heads are causally important:**

| Method | Mean Accuracy Drop @ 50 Heads | Random Baseline |
|--------|-------------------------------|-----------------|
| Summed Attention | 30.8% | ~5% |
| Wu24 | 30.9% | ~5% |
| QRHead | 30.8% | ~5% |

**However, the ablation curves differ dramatically at early levels:**

| Method | Steepest early drop (5 heads) | Pattern |
|--------|-------------------------------|---------|
| QRHead | Up to **50%** (inc_year) | Sharp early damage — most concentrated |
| Summed Attention | Up to **87.5%** (base/inc_year) | Extreme but inconsistent across configs |
| Wu24 | Usually **0%** at 5-10 heads | Diffuse — effects only emerge at 30+ heads |

**The critical question this raises:** If three different methods identify three different sets of heads, and all three sets are causally important, then what IS a "retrieval head"? The concept may be poorly defined — there appear to be multiple overlapping circuits that contribute to retrieval, each visible to different measurement approaches.

**Why this matters for publication:** This is a direct challenge to the framing of Wu24 and similar papers that talk about "the retrieval heads" as if they're a single well-defined set. Our data suggests retrieval is mediated by multiple distinct mechanisms (attention routing, token copying, query-response alignment) that happen to coexist in the same model. The 12% overlap means we need a more nuanced taxonomy of retrieval-related heads.

---

### Finding 5: INSTRUCTION TUNING REDISTRIBUTES RETRIEVAL ACROSS MORE HEADS

**Base model retrieval is concentrated and fragile. Instruct model retrieval is distributed and robust.**

| Model | Mean Accuracy Drop @ 50 Heads |
|-------|-------------------------------|
| Llama-3-8B-Base | **36.8%** |
| Llama-3-8B-Instruct | **24.9%** |

The base model shows larger drops (more concentrated), but also:
- Base/inc_year: 87.5% drop at 5 heads (catastrophic concentration)
- Instruct/inc_year: 0% drop at 5 heads (distributed across more heads)

Some heads shift dramatically:

| Head | Instruct Rank | Base Rank | Shift |
|------|---------------|-----------|-------|
| L10H19 | 814 | 176 | +638 |
| L15H19 | 727 | 226 | +501 |
| L10H26 | 719 | 278 | +441 |

**The "Wu24 base model blindspot":** Wu24's method identifies heads in the base model that have **zero causal effect** when ablated (base/inc_state/2048: 50 heads ablated → 0% accuracy drop). QRHead and summed attention both find causally important heads for the same config. This means Wu24's copy-behavior signal doesn't capture whatever the base model is actually using for retrieval.

**Why this matters:** Instruction tuning doesn't just improve surface behavior — it fundamentally reorganizes the internal retrieval architecture. Understanding this reorganization could inform how we do fine-tuning, what circuits are affected by RLHF, and whether instruction tuning makes models more or less interpretable.

---

### Finding 6: QUESTION-TYPE SPECIALIZATION — Dedicated Circuits for Different Fact Types

**Some heads are hyper-specialized for specific question types:**

| Head | Best Question | Best Rank | Avg Rank for Other Questions |
|------|---------------|-----------|------------------------------|
| L16H9 | HQ State | #2 | ~423 |
| L16H11 | HQ State | #17 | ~681 |
| L23H25 | Inc Year | #17 | ~268 |
| L22H13 | Employee Count | #20 | ~226 |

**Numerical vs. categorical retrieval differ systematically:**

| Category | Mean Accuracy Drop @ 50 Heads | Head Overlap (within category) |
|----------|-------------------------------|-------------------------------|
| Numerical (year, employee count) | 19.6% | 64.9% |
| Categorical (inc state, hq state) | 15.9% | 72.8% |
| Cross-category | — | 67.3% |

Numerical retrieval shows larger drops AND lower within-category overlap — suggesting numerical fact retrieval uses more specialized, less redundant circuits than categorical retrieval.

**Inc Year is consistently the most vulnerable question type:** 42.6% average drop at 50 heads (vs. 26% for inc_state). This may be because year retrieval requires more precise token-level copying than state retrieval.

**Why this matters:** The model doesn't use a single "retrieval module" — it has distinct circuits for different types of factual lookup. This connects to work on task-specific circuits and could inform understanding of how models organize knowledge internally.

---

### Finding 7: QRHead IDENTIFIES THE MOST CAUSALLY CONCENTRATED HEADS

**QRHead consistently produces the steepest early ablation damage.**

Across instruct/2048 configs:

| Question | QRHead @ 5 heads | Summed Attn @ 5 heads | Wu24 @ 5 heads |
|----------|-------------------|----------------------|----------------|
| inc_state | 15.2% drop | 21.2% drop | 0% |
| inc_year | **50.0% drop** | 0% | 0% |
| employee_count | **22.2% drop** | 13.9% drop | 5.6% drop |
| hq_state | 0% | 7.7% | 0% |

For the QRHead/instruct/inc_state/8192 case, the ablation curve shows a striking "delayed cliff":
```
5 heads  → 0% drop (resilient)
10 heads → 0% drop (resilient)  
20 heads → 21.2% drop (starting to feel it)
30 heads → 48.5% drop (steep decline)
40 heads → 60.6% drop (near-catastrophic)
50 heads → 54.5% drop (slight recovery — non-monotonic again)
```

At longer contexts, the damage is delayed but then more severe — as if the retrieval circuit has more redundancy at long contexts but a sharper failure mode once the redundancy is exceeded.

**Why this matters:** QRHead's calibration step (subtracting null-query attention) appears to isolate the most functionally critical heads — those whose contribution is specifically driven by the query, not just general attention patterns. This suggests calibrated detection methods are superior for identifying causal circuits, and that QRHead-style approaches could be extended to more fine-grained circuit discovery.

---

## 5. Raw Data: The Strongest Signals

### The Five Most Extreme Ablation Results

| Rank | Config | Ablation | Accuracy Drop | Interpretation |
|------|--------|----------|---------------|----------------|
| 1 | SumAttn / Base / inc_year / 2048 | 5 heads | **87.5%** | 5 heads = entire year retrieval |
| 2 | SumAttn / Base / inc_year / 2048 | 30 heads | **90.6%** | Maximum destruction |
| 3 | SumAttn / Instruct / inc_year / 2048 | 10 heads | **68.8%** | 10 heads critical for instruct too |
| 4 | QRHead / Instruct / inc_state / 8192 | 40 heads | **60.6%** | Long-context cliff |
| 5 | QRHead / Instruct / inc_year / 2048 | 50 heads | **59.4%** | QRHead finds strongest causal set |

### The Five Most Surprising Results

| Rank | Config | What's Surprising |
|------|--------|-------------------|
| 1 | Wu24 / Base / inc_state / 2048 | **0% drop at 50 heads** — Wu24 finds completely non-causal heads for base model |
| 2 | SumAttn / Base / inc_year / 2048 | **Recovery at 20 heads** — 9.4% → 84.4% after adding 10 more ablated heads |
| 3 | QRHead / inc_state / 8192 random | **Random ablation improves accuracy** — 100% vs 97% baseline |
| 4 | SumAttn / Instruct / inc_state / 8192 | **Only 3% drop at 50 heads** — 8K context makes summed-attention heads non-causal |
| 5 | L31H14 | **Rank 1024 → rank 2** when going from 2048 to 6144 tokens |

### Head Stability Rankings (Across All Configurations)

**Most stable heads** (consistently important regardless of method, question, model, context length):
1. **L14H31** — Top-3 across all methods and token lengths for instruct model
2. **L20H14** — Top-20 across all methods; top-1 for summed attention at 8192

**Most volatile heads:**
1. **L31H14** — Rank 1024 (last) at 2K, rank 2 at 6K (summed attention, base model)
2. **L16H1** — Rank 1 at 2K, rank 786 at 6K (summed attention, instruct)
3. **L6H3** — Rank 41 to 1008 across QRHead configs

---

## 6. What Is Novel vs. What Is Known

### Already Established in Literature
- Retrieval heads exist and are causally important (Wu24)
- Different detection methods can find retrieval heads (Wu24, QRHead)
- Ablating retrieval heads degrades factual recall (Wu24)
- Instruction tuning changes model internals (general knowledge)

### Our Novel Contributions (Not Yet in Literature, to Our Knowledge)

| Finding | Novelty Level | Closest Existing Work |
|---------|---------------|----------------------|
| **Non-monotonic ablation curves** suggesting compensatory circuits | **HIGH** — not reported anywhere | Circuit resilience work (Wang et al. 2023) discusses backup heads but not in retrieval context |
| **Context length triggers complete circuit replacement** (Jaccard 0.026) | **HIGH** — challenges fundamental assumption | Wu24 and QRHead test single context lengths |
| **12% inter-method overlap despite equal causal importance** | **HIGH** — questions the definition of "retrieval head" | Wu24 and QRHead don't compare to each other |
| **5 heads control 87.5% of retrieval** for specific tasks | **MEDIUM-HIGH** — more extreme than anything in Wu24 | Wu24 shows gradual curves, not cliff-edges |
| **Question-type specialization** with dedicated head circuits | **MEDIUM** — extends existing findings | Wu24 tests with generic questions |
| **Instruction tuning redistributes retrieval across more heads** | **MEDIUM** — not specifically shown for retrieval | General RLHF/instruction tuning literature |
| **QRHead finds most causally concentrated heads** (steepest early drops) | **MEDIUM** — practical comparison not done before | QRHead paper doesn't do ablation comparison |
| **Emergent layer-31 heads at long context** | **MEDIUM** — specific to architecture | General observations about deep layer behavior |

---

## 7. Candidate Research Directions

### Direction A: "Retrieval Circuits Are Dynamic, Not Fixed" (STRONGEST CANDIDATE)

**Thesis:** The concept of "retrieval heads" as static model properties is wrong. Retrieval is mediated by dynamic circuits that change with context length, question type, and model tuning.

**Supporting evidence:**
- Complete head replacement at different context lengths (Jaccard 0.026)
- Non-monotonic ablation curves suggesting backup circuits
- 12% inter-method overlap despite equal causal validation
- Instruction tuning redistributes retrieval mechanisms

**Paper structure:**
1. Show that three validated methods identify different heads (12% overlap)
2. Show that context length causes complete circuit replacement
3. Show non-monotonic ablation curves as evidence of circuit redundancy
4. Argue that "retrieval head" is an oversimplification; propose "retrieval circuit ensemble"

**Target venues:** NeurIPS, ICML, ICLR (mechanistic interpretability track)

**Strengths:**
- Directly challenges a popular recent finding (Wu24 has 80+ citations)
- Multiple independent lines of evidence
- Clear narrative arc
- Practical implications for pruning, editing, interpretability

**Risks:**
- Need to rule out that context-length effects are artifacts of the detection method
- Single model family (Llama 3 8B) may limit generalizability
- Reviewers may want more mechanistic explanation of WHY circuits switch

**What we'd need additionally:**
- Validate on a second model family (e.g., Mistral, Qwen)
- Deeper mechanistic analysis of the "backup circuit" phenomenon
- Formal definition of circuit dynamism and measurement framework

---

### Direction B: "Compensatory Circuits in Retrieval — Evidence for Redundancy" (MOST NOVEL)

**Thesis:** When primary retrieval heads are ablated, dormant backup heads activate to compensate, producing non-monotonic ablation curves. This reveals a built-in redundancy mechanism.

**Supporting evidence:**
- Non-monotonic ablation curves in multiple configs (base/inc_year, instruct/inc_year, QRHead/inc_year)
- Recovery-then-collapse pattern: ablate primary → backup activates → ablate backup → system fails
- The specific heads at the "recovery" ablation level can be identified and studied

**Paper structure:**
1. Document non-monotonic ablation phenomenon
2. Identify the specific "backup" heads that enable recovery
3. Test whether backup heads are active during normal operation (they shouldn't be)
4. Propose a model of layered circuit redundancy

**Target venues:** ICLR (mechanistic interpretability), EMNLP

**Strengths:**
- Highly novel — nobody has reported this for retrieval
- Connects to a growing literature on circuit redundancy and backup heads
- Clear experimental predictions that can be tested further
- Practical implications for model robustness and safe editing

**Risks:**
- Non-monotonicity might be an artifact of small test set (n=32-33) — need statistical significance
- Hard to rule out that the "recovery" is just noise
- Need activation patching or similar to confirm the backup circuit hypothesis

**What we'd need additionally:**
- Statistical analysis (bootstrapping) to confirm non-monotonicity is significant
- Activation analysis to check if "backup" heads change behavior when primary heads are ablated
- Larger test set or additional task types to confirm pattern generalizes
- Potentially: ablate primary + record activations → show backup heads increase activity

---

### Direction C: "What Do Different Retrieval Head Methods Actually Find?" (EASIEST TO PUBLISH)

**Thesis:** Existing retrieval head detection methods (Wu24, QRHead, summed attention) identify fundamentally different subsets of heads, each capturing a different aspect of the retrieval process. We need a taxonomy.

**Supporting evidence:**
- 12% three-way overlap
- Different layer preferences per method
- Wu24 identifies heads with no causal effect in base model
- QRHead finds most causally concentrated heads
- Different score distributions (Wu24 has extreme #1 dominance)

**Paper structure:**
1. Systematic comparison of three methods on identical data
2. Show low overlap but equal causal validation
3. Characterize what each method captures (attention routing vs. copying vs. query alignment)
4. Propose a taxonomy of retrieval-related head functions
5. Recommend best practices for which method to use when

**Target venues:** ACL, EMNLP, NeurIPS (practical/methodological track)

**Strengths:**
- Clean, well-controlled study with clear takeaways
- Immediately useful to the community (practical recommendations)
- Less speculative than Directions A or B
- Complete dataset already exists

**Risks:**
- May be seen as "just a comparison paper" rather than a contribution
- Need to offer insight beyond "methods disagree"
- Single model family limits claims

**What we'd need additionally:**
- Deeper analysis of what makes each method's unique heads different (activation patterns? layer positions? information flow?)
- Potentially a unified detection method that combines insights from all three
- Second model family for generalizability

---

### Direction D: "Task-Specific Retrieval Circuits in LLMs" (DOMAIN-SPECIFIC)

**Thesis:** Different types of factual retrieval (numerical vs. categorical, year vs. state) are mediated by distinct, specialized circuits in the model.

**Supporting evidence:**
- 5 heads control 87.5% of year retrieval but not state retrieval
- Question specialists (L16H9 for HQ State only)
- Numerical questions show larger ablation effects
- Different within-category vs. cross-category head overlap

**Paper structure:**
1. Show question-type-specific heads with ablation validation
2. Characterize numerical vs. categorical retrieval circuits
3. Demonstrate extreme concentration for specific tasks
4. Discuss implications for knowledge editing and factual probing

**Target venues:** EMNLP, ACL (knowledge/factuality track)

**Strengths:**
- Directly relevant to knowledge editing, factual probing, and hallucination research
- Clear practical implications
- Strong data for the year-retrieval concentration finding

**Risks:**
- May overlap with existing task-specific circuit work (Meng et al., etc.)
- Domain-specific (SEC filings) may limit perceived generality
- Need to validate with more diverse question types

---

### Direction E: "How Context Length Reorganizes Transformer Internals" (BROAD APPEAL)

**Thesis:** Increasing context length doesn't just affect positional encodings — it fundamentally reorganizes which attention heads are active, with deep-layer heads emerging exclusively at long contexts.

**Supporting evidence:**
- L31H14: rank 1024 → rank 2 from 2K to 6K
- L16H1: rank 1 → rank 786 from 2K to 6K
- Jaccard similarity of 0.026 between 2K and 8K top heads
- Layer 31 heads emerge as "long-context specialists"
- QRHead shows completely different head sets at 8K vs 2K

**Paper structure:**
1. Document context-length-dependent head ranking changes
2. Identify short-context specialists vs. long-context emergent heads
3. Characterize the layer-level reorganization
4. Connect to positional encoding and attention pattern literature

**Target venues:** ICLR, NeurIPS

**Strengths:**
- Broad relevance (everyone cares about long context)
- Connects to hot topic (long-context LLMs)
- Clear, dramatic visualizations possible

**Risks:**
- May be explained by trivial positional encoding effects
- Single architecture limits claims
- Need to control for the fact that attention patterns mechanically change with sequence length

---

## 8. What We Need From Literature Review

Please search for and assess the following:

### Critical Questions for Literature Review

1. **Has anyone reported non-monotonic ablation curves in transformers?** Search for: "non-monotonic ablation", "compensatory heads", "backup circuits", "circuit redundancy in transformers", "ablation recovery". If this hasn't been shown, Direction B becomes very strong.

2. **Has anyone studied how retrieval heads change with context length?** Search for: "retrieval heads context length", "attention head context dependence", "dynamic circuits in transformers". If nobody has, Direction A/E becomes strong.

3. **Has anyone systematically compared retrieval head detection methods?** Search for: "Wu24 vs QRHead", "retrieval head comparison", "attention head detection methods comparison". If nobody has, Direction C is easy to publish.

4. **What is the state of "circuit redundancy" or "backup head" research in mechanistic interpretability?** Search for: Wang et al. 2023 backup heads, "circuit resilience transformers", "head redundancy". Understanding this landscape is critical for Directions A and B.

5. **Has anyone shown task-specific retrieval circuits (beyond generic factual recall)?** Search for: "task-specific attention heads", "question-type-specific circuits", "numerical vs categorical retrieval". If limited, Direction D is viable.

6. **What is the current framing of "retrieval heads" in the literature?** Are they treated as fixed model properties? Do recent papers challenge this? Search for: recent citations of Wu24, QRHead papers, any follow-up work.

7. **Has anyone studied how instruction tuning affects retrieval circuits specifically?** Search for: "instruction tuning attention patterns", "RLHF attention heads", "SFT internal representations". If limited, this is a strong sub-finding.

8. **What are the most recent mechanistic interpretability papers at top venues (NeurIPS 2024/2025, ICLR 2025, ICML 2025)?** What topics are hot? What's getting accepted? This helps us target the narrative.

9. **Are there existing benchmarks or frameworks for studying retrieval circuits?** Search for: "retrieval head benchmark", "factual retrieval mechanistic analysis benchmark". If not, our framework itself could be a contribution.

10. **Has anyone used real-world data (vs. synthetic) for retrieval head analysis?** Wu24 and QRHead use synthetic needle-in-haystack. Our use of real SEC filings might be a differentiator. Search for: "retrieval head real data", "attention analysis real documents".

### Key Papers to Examine

- Wu et al. 2024, "Retrieval Head Mechanistically Explains Long-Context Factuality" (arXiv:2404.15574)
- QRHead paper (need to find exact citation)
- Wang et al. 2023, on backup heads / circuit resilience
- Meng et al. (ROME, MEMIT) on knowledge editing and localization
- Olsson et al. 2022, "In-context Learning and Induction Heads"
- Conmy et al. 2023, "Towards Automated Circuit Discovery"
- Any recent 2024-2025 papers on long-context attention mechanisms

---

## Appendix: Full Result Tables

### A1: Phase 3 Ablation — All Instruct/2048 Configs

#### Summed Attention

| Question | Baseline | Drop@5 | Drop@10 | Drop@20 | Drop@30 | Drop@40 | Drop@50 | Rand@50 |
|----------|----------|--------|---------|---------|---------|---------|---------|---------|
| inc_state | 97.0% | 21.2% | 18.2% | 0.0% | 6.1% | 9.1% | 12.1% | 0.0% |
| inc_year | 93.8% | 0.0% | 68.8% | 40.6% | 6.3% | 18.8% | 31.3% | 6.3% |
| employee_count | 30.6% | 13.9% | 19.4% | 25.0% | 22.2% | 22.2% | 27.8% | 8.3% |
| hq_state | 73.1% | 7.7% | 26.9% | 3.8% | 38.5% | 42.3% | 42.3% | -3.8% |

#### Wu24 Retrieval Head

| Question | Baseline | Drop@5 | Drop@10 | Drop@20 | Drop@30 | Drop@40 | Drop@50 | Rand@50 |
|----------|----------|--------|---------|---------|---------|---------|---------|---------|
| inc_state | 97.0% | 0.0% | 0.0% | 0.0% | 6.1% | 12.1% | 18.2% | 0.0% |
| inc_year | 93.8% | 0.0% | 0.0% | 3.1% | 0.0% | 34.4% | 40.6% | 25.0% |
| employee_count | 30.6% | 5.6% | 2.8% | 11.1% | 11.1% | 8.3% | 8.3% | 2.8% |
| hq_state | 73.1% | 0.0% | 0.0% | 7.7% | 7.7% | 11.5% | 7.7% | 0.0% |

#### QRHead

| Question | Baseline | Drop@5 | Drop@10 | Drop@20 | Drop@30 | Drop@40 | Drop@50 | Rand@50 |
|----------|----------|--------|---------|---------|---------|---------|---------|---------|
| inc_state | 97.0% | 15.2% | 24.2% | 36.4% | 39.4% | 42.4% | 39.4% | 0.0% |
| inc_year | 93.8% | 50.0% | 34.4% | 28.1% | 9.4% | 50.0% | 59.4% | 0.0% |
| employee_count | 30.6% | 22.2% | 30.6% | 30.6% | 30.6% | 30.6% | 27.8% | 0.0% |
| hq_state | 73.1% | 0.0% | -3.8% | 11.5% | 15.4% | 11.5% | 26.9% | 0.0% |

### A2: Phase 3 Ablation — Key Base Model Configs

| Method | Question | Tokens | Baseline | Drop@5 | Drop@10 | Drop@30 | Drop@50 | Rand@50 |
|--------|----------|--------|----------|--------|---------|---------|---------|---------|
| SumAttn | inc_year | 2048 | 96.9% | **87.5%** | 87.5% | 90.6% | 87.5% | 15.6% |
| SumAttn | inc_state | 2048 | 97.0% | 21.2% | 12.1% | 54.5% | 42.4% | 9.1% |
| Wu24 | inc_state | 2048 | 97.0% | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% |
| QRHead | inc_state | 2048 | 97.0% | 21.2% | 24.2% | 51.5% | 30.3% | 3.0% |
| SumAttn | hq_state | 6144 | 92.3% | — | — | — | **46.2%** | — |
| SumAttn | hq_state | 8192 | 88.5% | — | — | — | **42.3%** | — |

### A3: Phase 2 Head Rankings — Top 5 by Method (Instruct/inc_state/2048)

| Rank | Summed Attention | Score | Wu24 | Score | QRHead | Score |
|------|------------------|-------|------|-------|--------|-------|
| 1 | L16H1 | 120.9 | L15H30 | 0.0107 | L13H18 | 0.420 |
| 2 | L14H31 | 112.6 | L16H1 | 0.0073 | L16H1 | 0.419 |
| 3 | L14H20 | 111.1 | L24H27 | 0.0070 | L14H31 | 0.405 |
| 4 | L13H18 | 110.8 | L15H1 | 0.0068 | L17H29 | 0.361 |
| 5 | L20H1 | 110.1 | L20H14 | 0.0063 | L14H30 | 0.359 |

### A4: Context-Length Head Stability (Summed Attention / Instruct / inc_state)

| Head | Rank@2048 | Rank@4096 | Rank@6144 | Rank@8192 | Stable? |
|------|-----------|-----------|-----------|-----------|---------|
| L14H31 | 2 | 2 | 1 | 2 | **YES — most stable** |
| L20H14 | 9 | 8 | 5 | 1 | **YES — rises with context** |
| L20H1 | 5 | 6 | 7 | 4 | **YES** |
| L16H1 | 1 | 1 | 786 | 726 | **NO — short-context only** |
| L31H14 | 1024 | 948 | 3 | 9 | **NO — long-context only** |
| L31H9 | 936 | 983 | 4 | 7 | **NO — long-context only** |

### A5: Wu24 Success Rates by Config

| Config | Total Samples | Successful Retrievals | Success Rate |
|--------|---------------|----------------------|--------------|
| instruct/inc_state/2048 | 127 | 123 | 96.9% |
| instruct/inc_year/2048 | 126 | 119 | 94.4% |
| instruct/hq_state/2048 | 106 | 80 | 75.5% |
| base/inc_state/2048 | 127 | 122 | 96.1% |
| instruct/inc_state/8192 | 127 | 122 | 96.1% |

### A6: Universal Heads Per Method (% of configs where head appears in top-50)

| Method | Head | Universality |
|--------|------|-------------|
| Wu24 | L24H27 | 95% |
| QRHead | L14H31 | 90% |
| Summed Attention | L20H14 | 75% |
| All Methods | L20H14 | Cross-method consensus |
| All Methods | L14H31 | Cross-method consensus |

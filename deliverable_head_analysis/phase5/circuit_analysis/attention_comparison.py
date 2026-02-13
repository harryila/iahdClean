#!/usr/bin/env python3
"""
Phase 5 — Circuit Analysis: Attention Pattern Comparison (Clean vs. Ablated)

Tests the backup circuit hypothesis by comparing attention patterns with
and without primary retrieval heads ablated.

Hypothesis: When the top-5 "primary" retrieval heads are ablated, "backup"
heads (ranked ~6-20 in Phase 2) increase their attention to the needle,
compensating for the lost primary circuit. When those backup heads are also
ablated (at the 30-head level), the system collapses — explaining the
non-monotonic ablation curve validated in Phase A.

Experiment:
  For each test sample:
    1. CLEAN forward pass → record attention_to_needle[head] for all 1024 heads
    2. ABLATED forward pass (top-5 zeroed) → record same metric for all heads
    3. Compute Δ = ablated - clean per head
  Average Δ across samples, then check if high-Δ heads match Phase 2 ranks 6-20.

Primary config: summed_attention / base / inc_year / 2048
(This is the config with the strongest bootstrap-confirmed non-monotonic signal.)
"""
import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS = {
    "instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "base": "meta-llama/Meta-Llama-3-8B",
}

QUESTIONS = {
    "inc_state": {
        "prompt": "What state was the company incorporated in?",
        "column": "original_Inc_state_truth",
    },
    "inc_year": {
        "prompt": "What year was the company incorporated?",
        "column": "original_Inc_year_truth",
    },
    "employee_count": {
        "prompt": "How many employees does the company have?",
        "column": "employee_count_truth",
    },
    "hq_state": {
        "prompt": "What state is the company headquarters located in?",
        "column": "headquarters_state_truth",
    },
}

# Paths (relative to this script in phase5/circuit_analysis/)
PHASE1_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "phase1")
GT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "edgar_gt_verified_slim.csv")

PHASE2_DIRS = {
    "summed_attention": os.path.join(os.path.dirname(__file__), "..", "..", "phase2", "summed_attention", "results"),
    "wu24": os.path.join(os.path.dirname(__file__), "..", "..", "phase2", "retrieval_head_wu24", "results"),
    "qrhead": os.path.join(os.path.dirname(__file__), "..", "..", "phase2", "qrhead", "results"),
}

ALICE_TEXT = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out again. The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs."""


# =============================================================================
# ABLATION MECHANISM (same as Phase 3/5)
# =============================================================================

class HeadAblator:
    """Zeros out attention head outputs BEFORE o_proj via forward pre-hooks."""

    def __init__(self, model, heads_to_ablate):
        self.model = model
        self.heads_to_ablate = heads_to_ablate
        self.hooks = []
        self.num_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads

    def _make_ablation_pre_hook(self, layer_idx, heads_in_layer):
        def hook(module, args):
            hidden_states = args[0]
            modified = hidden_states.clone()
            batch_size, seq_len, _ = modified.shape
            reshaped = modified.view(batch_size, seq_len, self.num_heads, self.head_dim)
            for head_idx in heads_in_layer:
                reshaped[:, :, head_idx, :] = 0
            modified = reshaped.view(batch_size, seq_len, -1)
            return (modified,)
        return hook

    def __enter__(self):
        heads_by_layer = defaultdict(list)
        for layer_idx, head_idx in self.heads_to_ablate:
            heads_by_layer[layer_idx].append(head_idx)
        for layer_idx, heads in heads_by_layer.items():
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_pre_hook(
                self._make_ablation_pre_hook(layer_idx, heads)
            )
            self.hooks.append(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_data():
    test_path = os.path.join(PHASE1_DIR, "test_samples.json")
    with open(test_path, "r") as f:
        test_filenames = json.load(f)
    gt_df = pd.read_csv(GT_PATH)
    cache_path = os.path.join(PHASE1_DIR, "section1_cache.json")
    with open(cache_path, "r") as f:
        section1_cache = json.load(f)
    return test_filenames, gt_df, section1_cache


def load_top_heads_from_phase2(method, model_key, question_key, total_tokens, top_n=50):
    phase2_dir = PHASE2_DIRS[method]
    model_dir = "llama3_instruct" if model_key == "instruct" else "llama3_base"
    results_path = os.path.join(phase2_dir, model_dir, question_key, f"tokens_{total_tokens}.json")
    if not os.path.exists(results_path):
        print(f"Warning: Phase 2 results not found at {results_path}")
        return []
    with open(results_path, "r") as f:
        results = json.load(f)
    heads = []
    for h in results["head_rankings"][:top_n]:
        head_str = h["head"]
        parts = head_str.replace("L", "").split("H")
        heads.append((int(parts[0]), int(parts[1])))
    return heads


# =============================================================================
# PROMPT CONSTRUCTION (returns needle boundaries)
# =============================================================================

def create_prompt_with_needle(tokenizer, section1_text, question_prompt, total_tokens, needle_position=0.5):
    """
    Create prompt with needle in haystack. Returns prompt AND needle token boundaries.
    """
    needle_tokens = tokenizer.encode(section1_text, add_special_tokens=False)
    question_text = f"\n\nQuestion: {question_prompt}\nAnswer in one word:"
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)

    available_for_haystack = total_tokens - 1 - len(needle_tokens) - len(question_tokens)
    if available_for_haystack < 0:
        max_needle = total_tokens - 1 - len(question_tokens) - 100
        needle_tokens = needle_tokens[:max_needle]
        available_for_haystack = 100

    before_count = int(available_for_haystack * needle_position)
    after_count = available_for_haystack - before_count

    alice_tokens = tokenizer.encode(ALICE_TEXT, add_special_tokens=False)

    def get_haystack_tokens(count):
        if count <= 0:
            return []
        tokens = []
        while len(tokens) < count:
            tokens.extend(alice_tokens)
        return tokens[:count]

    before_tokens = get_haystack_tokens(before_count)
    after_tokens = get_haystack_tokens(after_count)

    # BOS + before + needle + after + question
    full_tokens = [tokenizer.bos_token_id] + before_tokens + needle_tokens + after_tokens + question_tokens

    needle_start = 1 + len(before_tokens)  # 1 for BOS
    needle_end = needle_start + len(needle_tokens)

    prompt = tokenizer.decode(full_tokens, skip_special_tokens=False)
    return prompt, needle_start, needle_end


# =============================================================================
# ATTENTION RECORDING
# =============================================================================

def record_attention_to_needle(model, tokenizer, prompt, needle_start, needle_end, heads_to_ablate=None):
    """
    Forward pass and record attention from last token to needle for every head.

    Args:
        model: The transformer model
        tokenizer: Tokenizer
        prompt: Full prompt string
        needle_start: Start token index of needle
        needle_end: End token index of needle
        heads_to_ablate: Optional list of (layer, head) tuples to ablate during this pass

    Returns:
        dict: {head_key: attention_score} for all 1024 heads
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]

    # Clamp needle bounds to actual sequence length
    ns = max(0, min(needle_start, seq_len - 1))
    ne = max(ns + 1, min(needle_end, seq_len))

    head_scores = {}

    if heads_to_ablate:
        ctx = HeadAblator(model, heads_to_ablate)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )

    for layer_idx, layer_attn in enumerate(outputs.attentions):
        # layer_attn: (batch, num_heads, seq_len, seq_len)
        last_pos_attn = layer_attn[0, :, -1, :].float()  # (num_heads, seq_len)
        for head_idx in range(last_pos_attn.shape[0]):
            score = last_pos_attn[head_idx, ns:ne].sum().item()
            head_scores[f"L{layer_idx}H{head_idx}"] = score

    del outputs, inputs
    torch.cuda.empty_cache()

    return head_scores


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 Circuit Analysis: Attention comparison (clean vs ablated)"
    )
    parser.add_argument("--method", default="summed_attention",
                        choices=["summed_attention", "wu24", "qrhead"],
                        help="Phase 2 method for head rankings (default: summed_attention)")
    parser.add_argument("--model", default="base", choices=["instruct", "base"],
                        help="Model variant (default: base)")
    parser.add_argument("--question", default="inc_year", choices=list(QUESTIONS.keys()),
                        help="Question type (default: inc_year)")
    parser.add_argument("--tokens", type=int, default=2048, choices=[2048, 4096, 6144, 8192],
                        help="Context length (default: 2048)")
    parser.add_argument("--ablate-top-n", type=int, default=5,
                        help="Number of top heads to ablate as 'primary' circuit (default: 5)")
    args = parser.parse_args()

    method = args.method
    model_key = args.model
    question_key = args.question
    total_tokens = args.tokens
    ablate_n = args.ablate_top_n
    model_name = MODELS[model_key]
    question_config = QUESTIONS[question_key]

    print("=" * 70)
    print("Phase 5 — Circuit Analysis: Attention Comparison (Clean vs Ablated)")
    print(f"Config: {method} / {model_key} / {question_key} / {total_tokens} tokens")
    print(f"Ablating top {ablate_n} heads as 'primary circuit'")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    test_filenames, gt_df, section1_cache = load_test_data()

    gt_column = question_config["column"]
    valid_samples = []
    for filename in test_filenames:
        row = gt_df[gt_df["filename"] == filename]
        if len(row) == 0:
            continue
        gt_value = row[gt_column].values[0]
        if pd.isna(gt_value):
            continue
        if filename not in section1_cache:
            continue
        gt_str = str(gt_value)
        if isinstance(gt_value, float) and gt_value == int(gt_value):
            gt_str = str(int(gt_value))
        valid_samples.append({
            "filename": filename,
            "gt_value": gt_str,
            "section1": section1_cache[filename],
        })

    print(f"Valid test samples: {len(valid_samples)}")

    # Load Phase 2 head rankings
    top_heads = load_top_heads_from_phase2(method, model_key, question_key, total_tokens, top_n=50)
    print(f"Loaded {len(top_heads)} heads from Phase 2 ({method})")

    primary_heads = top_heads[:ablate_n]
    print(f"Primary heads (to ablate): {[f'L{l}H{h}' for l, h in primary_heads]}")
    print(f"Expected backup heads (ranks {ablate_n+1}-20): {[f'L{l}H{h}' for l, h in top_heads[ablate_n:20]]}")

    # Load model (use eager attention so output_attentions works)
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()

    num_layers = model.config.num_hidden_layers
    num_heads_per_layer = model.config.num_attention_heads
    total_heads = num_layers * num_heads_per_layer
    print(f"Model has {total_heads} heads ({num_layers} layers × {num_heads_per_layer} heads)")

    # ---- Run experiment ----

    all_clean = defaultdict(list)   # head_key -> list of scores across samples
    all_ablated = defaultdict(list)
    per_sample_results = []

    print(f"\nProcessing {len(valid_samples)} samples...")
    for sample in tqdm(valid_samples, desc="Samples"):
        try:
            prompt, needle_start, needle_end = create_prompt_with_needle(
                tokenizer, sample["section1"],
                question_config["prompt"], total_tokens,
            )

            # Clean forward pass
            clean_scores = record_attention_to_needle(
                model, tokenizer, prompt, needle_start, needle_end,
                heads_to_ablate=None,
            )

            # Ablated forward pass (top-N heads zeroed)
            ablated_scores = record_attention_to_needle(
                model, tokenizer, prompt, needle_start, needle_end,
                heads_to_ablate=primary_heads,
            )

            # Store per-head scores
            sample_delta = {}
            for head_key in clean_scores:
                all_clean[head_key].append(clean_scores[head_key])
                all_ablated[head_key].append(ablated_scores[head_key])
                sample_delta[head_key] = ablated_scores[head_key] - clean_scores[head_key]

            per_sample_results.append({
                "filename": sample["filename"],
                "needle_start": needle_start,
                "needle_end": needle_end,
            })

        except Exception as e:
            print(f"Error processing {sample['filename']}: {e}")
            continue

        torch.cuda.empty_cache()

    n_samples = len(per_sample_results)
    print(f"\nProcessed {n_samples} samples successfully.")

    # ---- Compute average Δ per head ----

    head_deltas = {}
    for head_key in all_clean:
        clean_mean = np.mean(all_clean[head_key])
        ablated_mean = np.mean(all_ablated[head_key])
        delta = ablated_mean - clean_mean
        head_deltas[head_key] = {
            "clean_mean": float(clean_mean),
            "ablated_mean": float(ablated_mean),
            "delta": float(delta),
            "delta_std": float(np.std([
                all_ablated[head_key][i] - all_clean[head_key][i]
                for i in range(n_samples)
            ])),
            "clean_std": float(np.std(all_clean[head_key])),
            "ablated_std": float(np.std(all_ablated[head_key])),
        }

    # ---- Rank heads by Δ (largest positive = most compensatory) ----

    ranked_by_delta = sorted(head_deltas.items(), key=lambda x: x[1]["delta"], reverse=True)

    # ---- Cross-reference with Phase 2 rankings ----

    phase2_rank = {}
    for rank, (layer, head) in enumerate(top_heads):
        phase2_rank[f"L{layer}H{head}"] = rank + 1  # 1-indexed

    print("\n" + "=" * 70)
    print("TOP 30 HEADS BY Δ (attention increase when primary heads ablated)")
    print("=" * 70)
    print(f"{'Rank':>4}  {'Head':>8}  {'Δ':>10}  {'Clean':>10}  {'Ablated':>10}  {'Phase2 Rank':>12}  {'Backup?':>8}")
    print("-" * 75)

    for i, (head_key, stats) in enumerate(ranked_by_delta[:30]):
        p2_rank = phase2_rank.get(head_key, ">50")
        is_backup = "YES" if isinstance(p2_rank, int) and ablate_n < p2_rank <= 20 else ""
        is_primary = "PRIMARY" if isinstance(p2_rank, int) and p2_rank <= ablate_n else ""
        label = is_primary or is_backup
        print(f"{i+1:>4}  {head_key:>8}  {stats['delta']:>+10.4f}  {stats['clean_mean']:>10.4f}  {stats['ablated_mean']:>10.4f}  {str(p2_rank):>12}  {label:>8}")

    # ---- Check backup hypothesis ----

    print("\n" + "=" * 70)
    print("BACKUP CIRCUIT HYPOTHESIS TEST")
    print("=" * 70)

    # Expected backup heads: Phase 2 ranks (ablate_n+1) through 20
    backup_candidates = set()
    for rank_idx in range(ablate_n, 20):
        if rank_idx < len(top_heads):
            l, h = top_heads[rank_idx]
            backup_candidates.add(f"L{l}H{h}")

    # Primary heads (ablated)
    primary_set = set()
    for l, h in primary_heads:
        primary_set.add(f"L{l}H{h}")

    # Stats for different head groups
    groups = {
        f"Primary (ablated, ranks 1-{ablate_n})": primary_set,
        f"Expected backup (ranks {ablate_n+1}-20)": backup_candidates,
        "Ranks 21-50": set(f"L{l}H{h}" for l, h in top_heads[20:50]),
        "Unranked (ranks 51+)": set(k for k in head_deltas if k not in set(f"L{l}H{h}" for l, h in top_heads[:50])),
    }

    for group_name, head_set in groups.items():
        if not head_set:
            continue
        deltas = [head_deltas[k]["delta"] for k in head_set if k in head_deltas]
        if deltas:
            mean_d = np.mean(deltas)
            std_d = np.std(deltas)
            max_d = max(deltas)
            min_d = min(deltas)
            n_positive = sum(1 for d in deltas if d > 0)
            print(f"\n{group_name} ({len(deltas)} heads):")
            print(f"  Mean Δ: {mean_d:+.4f}  (std: {std_d:.4f})")
            print(f"  Range:  [{min_d:+.4f}, {max_d:+.4f}]")
            print(f"  Positive Δ: {n_positive}/{len(deltas)} ({100*n_positive/len(deltas):.0f}%)")

    # ---- Verdict ----

    backup_deltas = [head_deltas[k]["delta"] for k in backup_candidates if k in head_deltas]
    unranked_deltas = [head_deltas[k]["delta"] for k in head_deltas
                       if k not in set(f"L{l}H{h}" for l, h in top_heads[:50])]

    backup_mean = np.mean(backup_deltas) if backup_deltas else 0
    unranked_mean = np.mean(unranked_deltas) if unranked_deltas else 0

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if backup_mean > unranked_mean and backup_mean > 0:
        print(f"BACKUP CIRCUIT HYPOTHESIS: SUPPORTED")
        print(f"  Expected backup heads (ranks {ablate_n+1}-20) show mean Δ = {backup_mean:+.4f}")
        print(f"  Unranked heads show mean Δ = {unranked_mean:+.4f}")
        print(f"  Backup heads increase attention to needle MORE than random heads.")
    elif backup_mean > 0:
        print(f"BACKUP CIRCUIT HYPOTHESIS: WEAK SUPPORT")
        print(f"  Backup heads show positive Δ ({backup_mean:+.4f}) but not clearly above unranked ({unranked_mean:+.4f}).")
    else:
        print(f"BACKUP CIRCUIT HYPOTHESIS: NOT SUPPORTED")
        print(f"  Backup heads do not show increased attention to needle (mean Δ = {backup_mean:+.4f}).")

    # ---- Save full results ----

    output = {
        "experiment": "attention_comparison_clean_vs_ablated",
        "method": method,
        "model_key": model_key,
        "model_name": model_name,
        "question": question_key,
        "total_tokens": total_tokens,
        "ablate_top_n": ablate_n,
        "primary_heads": [f"L{l}H{h}" for l, h in primary_heads],
        "expected_backup_heads": sorted(backup_candidates),
        "n_samples": n_samples,
        "timestamp": datetime.now().isoformat(),
        "head_deltas": head_deltas,
        "ranked_by_delta_top50": [
            {"rank": i + 1, "head": k, "delta": v["delta"],
             "clean_mean": v["clean_mean"], "ablated_mean": v["ablated_mean"],
             "phase2_rank": phase2_rank.get(k, None)}
            for i, (k, v) in enumerate(ranked_by_delta[:50])
        ],
        "group_summary": {
            name: {
                "n_heads": len([head_deltas[k]["delta"] for k in heads if k in head_deltas]),
                "mean_delta": float(np.mean([head_deltas[k]["delta"] for k in heads if k in head_deltas])) if heads else 0,
                "std_delta": float(np.std([head_deltas[k]["delta"] for k in heads if k in head_deltas])) if heads else 0,
            }
            for name, heads in groups.items() if heads
        },
    }

    model_dir = "llama3_instruct" if model_key == "instruct" else "llama3_base"
    output_dir = os.path.join(os.path.dirname(__file__), "results", model_dir, question_key)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{method}_tokens_{total_tokens}_ablate{ablate_n}.json")

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

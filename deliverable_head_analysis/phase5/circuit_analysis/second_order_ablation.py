#!/usr/bin/env python3
"""
Phase 5 — Circuit Analysis C2: Second-Order Ablation (Causal Confirmation)

Tests the dormant compensator hypothesis causally. If the heads identified
in Phase C (highest positive Δ in attention-to-needle when primary heads
are ablated) are true compensators, then ablating them *on top of* the
primary ablation should destroy the recovery we observed.

Experimental conditions:
  1. Baseline            — no ablation
  2. Primary only        — ablate top-N Phase 2 heads
  3. Primary + compensators — ablate top-N Phase 2 heads + top-K compensator heads
  4. Compensators only   — ablate top-K compensator heads (control)

Compensator heads are loaded from Phase C attention_comparison results
(ranked_by_delta_top50), picking the top-K heads by positive Δ that are
NOT in the primary set.

Usage:
    python second_order_ablation.py \
        --method summed_attention --model base --question inc_year --tokens 2048 \
        --primary-n 5 --compensator-n 10
"""
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

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


def load_compensator_heads(method, model_key, question_key, total_tokens, primary_n, compensator_n):
    """
    Load top compensator heads from Phase C attention_comparison results.
    These are heads with the largest positive Δ (attention increase when
    primary heads are ablated), excluding the primary heads themselves.
    """
    model_dir = "llama3_instruct" if model_key == "instruct" else "llama3_base"
    results_path = os.path.join(
        RESULTS_DIR, model_dir, question_key,
        f"{method}_tokens_{total_tokens}_ablate{primary_n}.json"
    )

    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Phase C results not found at {results_path}. "
            f"Run attention_comparison.py first with --ablate-top-n {primary_n}."
        )

    with open(results_path, "r") as f:
        phase_c = json.load(f)

    primary_set = set(phase_c["primary_heads"])

    # Pick top compensator_n heads by delta, excluding primary heads
    compensators = []
    for entry in phase_c["ranked_by_delta_top50"]:
        if entry["head"] in primary_set:
            continue
        if entry["delta"] <= 0:
            break
        compensators.append(entry)
        if len(compensators) >= compensator_n:
            break

    # Parse head strings to (layer, head) tuples
    heads = []
    for entry in compensators:
        head_str = entry["head"]
        parts = head_str.replace("L", "").split("H")
        heads.append((int(parts[0]), int(parts[1])))

    return heads, compensators


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def create_prompt_with_needle(tokenizer, section1_text, question_prompt, total_tokens, needle_position=0.5):
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

    full_tokens = [tokenizer.bos_token_id] + before_tokens + needle_tokens + after_tokens + question_tokens
    prompt = tokenizer.decode(full_tokens, skip_special_tokens=False)
    return prompt


# =============================================================================
# EVALUATION
# =============================================================================

def generate_answer(model, tokenizer, prompt, max_new_tokens=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()


def generate_answer_with_ablation(model, tokenizer, prompt, heads_to_ablate, max_new_tokens=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with HeadAblator(model, heads_to_ablate):
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()


def normalize_answer(answer):
    answer = str(answer).lower().strip()
    answer = answer.replace(",", "").replace(".", "").replace("'", "")
    try:
        if float(answer) == int(float(answer)):
            answer = str(int(float(answer)))
    except:
        pass
    return answer


def check_answer(generated, ground_truth):
    gen_norm = normalize_answer(generated)
    gt_norm = normalize_answer(ground_truth)
    if gen_norm == gt_norm:
        return True
    if gt_norm in gen_norm:
        return True
    gen_first = gen_norm.split()[0] if gen_norm.split() else ""
    if gen_first == gt_norm:
        return True
    return False


def evaluate_condition(model, tokenizer, samples, question_config, total_tokens, heads_to_ablate=None, desc="Evaluating"):
    """Evaluate accuracy under a given ablation condition. Returns accuracy and per-sample results."""
    correct = 0
    total = 0
    results = []

    for sample in tqdm(samples, desc=desc, leave=False):
        try:
            prompt = create_prompt_with_needle(
                tokenizer, sample["section1"],
                question_config["prompt"], total_tokens,
            )

            if heads_to_ablate:
                generated = generate_answer_with_ablation(
                    model, tokenizer, prompt, heads_to_ablate,
                )
            else:
                generated = generate_answer(model, tokenizer, prompt)

            is_correct = check_answer(generated, sample["gt_value"])
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "filename": sample["filename"],
                "gt_value": sample["gt_value"],
                "generated": generated,
                "correct": is_correct,
            })

        except Exception as e:
            print(f"Error processing {sample['filename']}: {e}")
            continue

        torch.cuda.empty_cache()

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 C2: Second-order ablation — causal test for dormant compensator heads"
    )
    parser.add_argument("--method", default="summed_attention",
                        choices=["summed_attention", "wu24", "qrhead"])
    parser.add_argument("--model", default="base", choices=["instruct", "base"])
    parser.add_argument("--question", default="inc_year", choices=list(QUESTIONS.keys()))
    parser.add_argument("--tokens", type=int, default=2048, choices=[2048, 4096, 6144, 8192])
    parser.add_argument("--primary-n", type=int, default=5,
                        help="Number of top Phase 2 heads as primary circuit (default: 5)")
    parser.add_argument("--compensator-n", type=int, default=10,
                        help="Number of top compensator heads from Phase C (default: 10)")
    args = parser.parse_args()

    method = args.method
    model_key = args.model
    question_key = args.question
    total_tokens = args.tokens
    primary_n = args.primary_n
    compensator_n = args.compensator_n
    model_name = MODELS[model_key]
    question_config = QUESTIONS[question_key]

    print("=" * 70)
    print("Phase 5 C2 — Second-Order Ablation: Causal Compensator Test")
    print(f"Config: {method} / {model_key} / {question_key} / {total_tokens} tokens")
    print(f"Primary heads: top {primary_n} from Phase 2")
    print(f"Compensator heads: top {compensator_n} from Phase C (by attention Δ)")
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

    # Load Phase 2 head rankings (primary heads)
    top_heads = load_top_heads_from_phase2(method, model_key, question_key, total_tokens, top_n=50)
    primary_heads = top_heads[:primary_n]
    print(f"Primary heads (Phase 2 top {primary_n}): {[f'L{l}H{h}' for l, h in primary_heads]}")

    # Load compensator heads from Phase C results
    compensator_heads, compensator_info = load_compensator_heads(
        method, model_key, question_key, total_tokens, primary_n, compensator_n,
    )
    print(f"Compensator heads (Phase C top {len(compensator_heads)} by Δ):")
    for entry in compensator_info:
        print(f"  {entry['head']:>8}  Δ={entry['delta']:+.4f}  phase2_rank={entry.get('phase2_rank', 'N/A')}")

    combined_heads = list(set(primary_heads + compensator_heads))
    print(f"\nTotal unique heads in combined set: {len(combined_heads)}")

    # Load model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    print(f"Model loaded. Total heads: {model.config.num_hidden_layers * model.config.num_attention_heads}")

    # ---- Run 4 conditions ----

    conditions = [
        {
            "name": "Baseline (no ablation)",
            "heads": None,
        },
        {
            "name": f"Primary only (top {primary_n} Phase 2 heads)",
            "heads": primary_heads,
        },
        {
            "name": f"Primary + Compensators (top {primary_n} + top {len(compensator_heads)} compensators)",
            "heads": combined_heads,
        },
        {
            "name": f"Compensators only (top {len(compensator_heads)} compensators, control)",
            "heads": compensator_heads,
        },
    ]

    results = []

    for i, cond in enumerate(conditions):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(conditions)}] {cond['name']}")
        if cond["heads"]:
            print(f"  Ablating {len(cond['heads'])} heads")
        print("=" * 60)

        acc, sample_results = evaluate_condition(
            model, tokenizer, valid_samples, question_config, total_tokens,
            heads_to_ablate=cond["heads"],
            desc=cond["name"][:40],
        )

        n_correct = sum(1 for s in sample_results if s["correct"])
        print(f"Accuracy: {acc:.1%} ({n_correct}/{len(sample_results)})")

        results.append({
            "condition": cond["name"],
            "heads_ablated": [f"L{l}H{h}" for l, h in cond["heads"]] if cond["heads"] else [],
            "n_heads_ablated": len(cond["heads"]) if cond["heads"] else 0,
            "accuracy": acc,
            "correct": n_correct,
            "total": len(sample_results),
            "sample_results": sample_results,
        })

        torch.cuda.empty_cache()

    # ---- Summary ----

    print(f"\n{'='*70}")
    print("SECOND-ORDER ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Condition':<55} {'Accuracy':>10} {'Drop':>8}")
    print("-" * 75)

    baseline_acc = results[0]["accuracy"]
    for r in results:
        drop = baseline_acc - r["accuracy"]
        print(f"{r['condition']:<55} {r['accuracy']:>10.1%} {drop:>+8.1%}")

    # ---- Causal verdict ----

    primary_acc = results[1]["accuracy"]
    combined_acc = results[2]["accuracy"]
    compensator_only_acc = results[3]["accuracy"]

    print(f"\n{'='*70}")
    print("CAUSAL VERDICT")
    print("=" * 70)

    # Key test: does adding compensator ablation on top of primary ablation
    # cause a further significant drop?
    further_drop = primary_acc - combined_acc

    print(f"Baseline accuracy:           {baseline_acc:.1%}")
    print(f"Primary-only accuracy:       {primary_acc:.1%}  (drop from baseline: {baseline_acc - primary_acc:+.1%})")
    print(f"Primary+Compensator accuracy:{combined_acc:.1%}  (drop from baseline: {baseline_acc - combined_acc:+.1%})")
    print(f"Compensator-only accuracy:   {compensator_only_acc:.1%}  (drop from baseline: {baseline_acc - compensator_only_acc:+.1%})")
    print(f"\nFurther drop from adding compensators: {further_drop:+.1%}")

    if further_drop > 0.05:  # >5% further drop
        print(f"\n→ COMPENSATOR HYPOTHESIS: SUPPORTED")
        print(f"  Ablating compensator heads on top of primary heads causes a further")
        print(f"  {further_drop:.1%} accuracy drop, confirming they provide backup retrieval.")
    elif further_drop > 0:
        print(f"\n→ COMPENSATOR HYPOTHESIS: WEAK SUPPORT")
        print(f"  Small further drop ({further_drop:.1%}) when compensators are also ablated.")
    else:
        print(f"\n→ COMPENSATOR HYPOTHESIS: NOT SUPPORTED")
        print(f"  No further accuracy drop when compensators are ablated on top of primary heads.")

    if compensator_only_acc < baseline_acc - 0.05:
        print(f"\n  Note: Compensator-only ablation also drops accuracy ({baseline_acc - compensator_only_acc:.1%}),")
        print(f"  suggesting these heads contribute to retrieval even without primary ablation.")
    else:
        print(f"\n  Note: Compensator-only ablation has minimal effect ({baseline_acc - compensator_only_acc:.1%} drop),")
        print(f"  consistent with dormant/backup role that only activates when primary heads fail.")

    # ---- Save results ----

    output = {
        "experiment": "second_order_ablation",
        "method": method,
        "model_key": model_key,
        "model_name": model_name,
        "question": question_key,
        "total_tokens": total_tokens,
        "primary_n": primary_n,
        "compensator_n": len(compensator_heads),
        "primary_heads": [f"L{l}H{h}" for l, h in primary_heads],
        "compensator_heads": [f"L{l}H{h}" for l, h in compensator_heads],
        "compensator_info": compensator_info,
        "n_samples": len(valid_samples),
        "timestamp": datetime.now().isoformat(),
        "conditions": results,
        "summary": {
            "baseline_acc": baseline_acc,
            "primary_only_acc": primary_acc,
            "primary_plus_compensator_acc": combined_acc,
            "compensator_only_acc": compensator_only_acc,
            "further_drop_from_compensators": further_drop,
        },
    }

    model_dir = "llama3_instruct" if model_key == "instruct" else "llama3_base"
    output_dir = os.path.join(RESULTS_DIR, model_dir, question_key)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"second_order_{method}_tokens_{total_tokens}_primary{primary_n}_comp{len(compensator_heads)}.json"
    )

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

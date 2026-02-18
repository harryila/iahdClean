#!/usr/bin/env python3
"""
Phase 6 — Cross-Task Ablation

Takes the top-N heads identified for each question type in Phase 2 and ablates
them while evaluating on every other question type.  Produces a 4×4 matrix
(source-task heads × target-task evaluation) at each context length, plus
random-head controls, revealing whether retrieval circuits are task-specific
or shared.

Question groupings:
    Numerical : inc_year, employee_count
    Textual   : inc_state, hq_state

Usage:
    python cross_task_ablation.py [--model base] [--method summed_attention]
                                  [--ablate-n 20] [--random-seeds 3]
"""

import argparse
import json
import os
import random
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
    "inc_year": {
        "prompt": "What year was the company incorporated?",
        "column": "original_Inc_year_truth",
        "group": "numerical",
    },
    "employee_count": {
        "prompt": "How many employees does the company have?",
        "column": "employee_count_truth",
        "group": "numerical",
    },
    "inc_state": {
        "prompt": "What state was the company incorporated in?",
        "column": "original_Inc_state_truth",
        "group": "textual",
    },
    "hq_state": {
        "prompt": "What state is the company headquarters located in?",
        "column": "headquarters_state_truth",
        "group": "textual",
    },
}

TASK_ORDER = ["inc_year", "employee_count", "inc_state", "hq_state"]
TOKEN_LENGTHS = [2048, 8192]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE1_DIR = os.path.join(SCRIPT_DIR, "..", "phase1")
GT_PATH = os.path.join(SCRIPT_DIR, "..", "..", "edgar_gt_verified_slim.csv")

PHASE2_DIRS = {
    "summed_attention": os.path.join(SCRIPT_DIR, "..", "phase2", "summed_attention", "results"),
    "wu24": os.path.join(SCRIPT_DIR, "..", "phase2", "retrieval_head_wu24", "results"),
    "qrhead": os.path.join(SCRIPT_DIR, "..", "phase2", "qrhead", "results"),
}

ALICE_TEXT = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out again. The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs."""


# =============================================================================
# ABLATION MECHANISM
# =============================================================================

class HeadAblator:
    def __init__(self, model, heads_to_ablate):
        self.model = model
        self.heads_to_ablate = heads_to_ablate
        self.hooks = []
        self.num_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads

    def _make_pre_hook(self, heads_in_layer):
        def hook(module, args):
            h = args[0].clone()
            bs, seq, _ = h.shape
            h = h.view(bs, seq, self.num_heads, self.head_dim)
            for head_idx in heads_in_layer:
                h[:, :, head_idx, :] = 0
            return (h.view(bs, seq, -1),)
        return hook

    def __enter__(self):
        by_layer = defaultdict(list)
        for layer, head in self.heads_to_ablate:
            by_layer[layer].append(head)
        for layer, heads in by_layer.items():
            o_proj = self.model.model.layers[layer].self_attn.o_proj
            self.hooks.append(o_proj.register_forward_pre_hook(self._make_pre_hook(heads)))
        return self

    def __exit__(self, *args):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_data():
    with open(os.path.join(PHASE1_DIR, "test_samples.json")) as f:
        test_filenames = json.load(f)
    gt_df = pd.read_csv(GT_PATH)
    with open(os.path.join(PHASE1_DIR, "section1_cache.json")) as f:
        section1_cache = json.load(f)
    return test_filenames, gt_df, section1_cache


def build_samples(test_filenames, gt_df, section1_cache, question_key):
    col = QUESTIONS[question_key]["column"]
    samples = []
    for fn in test_filenames:
        row = gt_df[gt_df["filename"] == fn]
        if len(row) == 0:
            continue
        val = row[col].values[0]
        if pd.isna(val):
            continue
        if fn not in section1_cache:
            continue
        gt_str = str(int(val)) if isinstance(val, float) and val == int(val) else str(val)
        samples.append({"filename": fn, "gt_value": gt_str, "section1": section1_cache[fn]})
    return samples


def load_top_heads(method, model_key, question_key, total_tokens, top_n=50):
    model_dir = "llama3_instruct" if model_key == "instruct" else "llama3_base"
    path = os.path.join(PHASE2_DIRS[method], model_dir, question_key, f"tokens_{total_tokens}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Phase 2 results not found: {path}")
    with open(path) as f:
        data = json.load(f)
    heads = []
    for h in data["head_rankings"][:top_n]:
        parts = h["head"].replace("L", "").split("H")
        heads.append((int(parts[0]), int(parts[1])))
    return heads


def get_random_heads(model_config, exclude, n, seed=42):
    rng = random.Random(seed)
    all_heads = [
        (l, h)
        for l in range(model_config.num_hidden_layers)
        for h in range(model_config.num_attention_heads)
    ]
    available = [x for x in all_heads if x not in set(exclude)]
    return rng.sample(available, min(n, len(available)))


# =============================================================================
# PROMPT + EVAL
# =============================================================================

def create_prompt(tokenizer, section1_text, question_prompt, total_tokens, needle_position=0.5):
    needle_tok = tokenizer.encode(section1_text, add_special_tokens=False)
    q_text = f"\n\nQuestion: {question_prompt}\nAnswer in one word:"
    q_tok = tokenizer.encode(q_text, add_special_tokens=False)

    avail = total_tokens - 1 - len(needle_tok) - len(q_tok)
    if avail < 0:
        needle_tok = needle_tok[: total_tokens - 1 - len(q_tok) - 100]
        avail = 100

    alice_tok = tokenizer.encode(ALICE_TEXT, add_special_tokens=False)

    def hay(count):
        if count <= 0:
            return []
        t = []
        while len(t) < count:
            t.extend(alice_tok)
        return t[:count]

    before = int(avail * needle_position)
    after = avail - before
    full = [tokenizer.bos_token_id] + hay(before) + needle_tok + hay(after) + q_tok
    return tokenizer.decode(full, skip_special_tokens=False)


def normalize(answer):
    a = str(answer).lower().strip().replace(",", "").replace(".", "").replace("'", "")
    try:
        if float(a) == int(float(a)):
            a = str(int(float(a)))
    except Exception:
        pass
    return a


def check_answer(gen, gt):
    g, t = normalize(gen), normalize(gt)
    if g == t:
        return True
    if t in g:
        return True
    first = g.split()[0] if g.split() else ""
    return first == t


def evaluate(model, tokenizer, samples, question_key, total_tokens, heads=None, desc="eval"):
    q_cfg = QUESTIONS[question_key]
    correct = 0
    total = 0
    per_sample = []

    for s in tqdm(samples, desc=desc[:45], leave=False):
        try:
            prompt = create_prompt(tokenizer, s["section1"], q_cfg["prompt"], total_tokens)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            if heads:
                with HeadAblator(model, heads):
                    with torch.no_grad():
                        out = model.generate(
                            **inputs, max_new_tokens=10, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )
            else:
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=10, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

            gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            ok = check_answer(gen, s["gt_value"])
            correct += ok
            total += 1
            per_sample.append({"filename": s["filename"], "gt": s["gt_value"], "gen": gen, "correct": ok})
        except Exception as e:
            print(f"  error {s['filename']}: {e}")
        torch.cuda.empty_cache()

    acc = correct / total if total else 0
    return acc, correct, total, per_sample


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 6: Cross-Task Ablation")
    parser.add_argument("--method", default="summed_attention", choices=list(PHASE2_DIRS))
    parser.add_argument("--model", default="base", choices=list(MODELS))
    parser.add_argument("--ablate-n", type=int, default=20)
    parser.add_argument("--random-seeds", type=int, default=3,
                        help="Number of random-head seeds to average over")
    args = parser.parse_args()

    method = args.method
    model_key = args.model
    ablate_n = args.ablate_n
    model_name = MODELS[model_key]

    print("=" * 72)
    print("Phase 6 — Cross-Task Ablation")
    print(f"  Method: {method}  Model: {model_key}  Ablate-N: {ablate_n}")
    print(f"  Tasks: {TASK_ORDER}")
    print(f"  Token lengths: {TOKEN_LENGTHS}")
    print("=" * 72)

    # --- data ---
    test_filenames, gt_df, section1_cache = load_test_data()
    task_samples = {}
    for task in TASK_ORDER:
        task_samples[task] = build_samples(test_filenames, gt_df, section1_cache, task)
        print(f"  {task}: {len(task_samples[task])} samples")

    # --- heads per (source_task, token_length) ---
    source_heads = {}
    for task in TASK_ORDER:
        for tl in TOKEN_LENGTHS:
            heads = load_top_heads(method, model_key, task, tl, top_n=ablate_n)
            source_heads[(task, tl)] = heads
            print(f"  heads[{task},{tl}] top-{len(heads)}: "
                  f"{[f'L{l}H{h}' for l,h in heads[:5]]}...")

    # --- head overlap matrix ---
    print("\n--- Head Overlap (top-{}) ---".format(ablate_n))
    for tl in TOKEN_LENGTHS:
        print(f"  token_length={tl}")
        for t1 in TASK_ORDER:
            for t2 in TASK_ORDER:
                if t2 <= t1:
                    continue
                s1 = set(source_heads[(t1, tl)])
                s2 = set(source_heads[(t2, tl)])
                shared = len(s1 & s2)
                print(f"    {t1:16s} ∩ {t2:16s} = {shared}/{ablate_n}")

    # --- load model ---
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()
    print("Model loaded.\n")

    # --- run experiment ---
    # Structure: results[token_length] = {
    #   "baselines": {target_task: acc},
    #   "cross":     {(source, target): acc},
    #   "random":    {target_task: avg_acc},
    # }
    all_results = {}

    for tl in TOKEN_LENGTHS:
        print(f"\n{'='*72}")
        print(f"TOKEN LENGTH = {tl}")
        print(f"{'='*72}")

        baselines = {}
        cross = {}
        random_ctrl = {}

        # 1) Baselines: one per target task
        for target in TASK_ORDER:
            samples = task_samples[target]
            acc, c, t, _ = evaluate(model, tokenizer, samples, target, tl,
                                    heads=None, desc=f"baseline {target} @{tl}")
            baselines[target] = {"accuracy": acc, "correct": c, "total": t}
            print(f"  Baseline {target:16s}: {acc:.1%} ({c}/{t})")

        # 2) Cross-task: source heads × target eval (includes diagonal = same-task)
        for source in TASK_ORDER:
            heads = source_heads[(source, tl)]
            for target in TASK_ORDER:
                samples = task_samples[target]
                label = f"{source}→{target}"
                acc, c, t, per = evaluate(model, tokenizer, samples, target, tl,
                                          heads=heads, desc=f"{label} @{tl}")
                drop = baselines[target]["accuracy"] - acc
                cross[(source, target)] = {
                    "accuracy": acc, "correct": c, "total": t,
                    "drop": drop,
                    "source_heads": [f"L{l}H{h}" for l, h in heads],
                }
                tag = "SELF" if source == target else "    "
                print(f"  {tag} {label:35s}: {acc:.1%}  (drop {drop:+.1%})")

        # 3) Random control: average over seeds, one per target task
        for target in TASK_ORDER:
            samples = task_samples[target]
            seed_accs = []
            for seed in range(args.random_seeds):
                rnd_heads = get_random_heads(model.config, [], ablate_n, seed=seed + 100)
                acc, c, t, _ = evaluate(model, tokenizer, samples, target, tl,
                                        heads=rnd_heads,
                                        desc=f"random→{target} s{seed} @{tl}")
                seed_accs.append(acc)
            avg = sum(seed_accs) / len(seed_accs)
            drop = baselines[target]["accuracy"] - avg
            random_ctrl[target] = {"avg_accuracy": avg, "drop": drop, "per_seed": seed_accs}
            print(f"  RAND {target:16s}: {avg:.1%} avg  (drop {drop:+.1%})")

        all_results[tl] = {
            "baselines": baselines,
            "cross": {f"{s}->{t}": v for (s, t), v in cross.items()},
            "random_control": random_ctrl,
        }

    # --- summary tables ---
    print(f"\n{'='*72}")
    print("CROSS-TASK ABLATION SUMMARY")
    print(f"{'='*72}")

    for tl in TOKEN_LENGTHS:
        r = all_results[tl]
        print(f"\n--- {tl} tokens ---")
        header = f"{'Source→Target':<30} {'Baseline':>8} {'Ablated':>8} {'Drop':>8} {'Rnd Drop':>8} {'Excess':>8}"
        print(header)
        print("-" * len(header))

        for source in TASK_ORDER:
            for target in TASK_ORDER:
                key = f"{source}->{target}"
                bl = r["baselines"][target]["accuracy"]
                ab = r["cross"][key]["accuracy"]
                drop = r["cross"][key]["drop"]
                rnd_drop = r["random_control"][target]["drop"]
                excess = drop - rnd_drop
                tag = " *" if source == target else ""
                print(f"  {source:>14}→{target:<13}{tag} {bl:>7.1%} {ab:>7.1%} {drop:>+7.1%} {rnd_drop:>+7.1%} {excess:>+7.1%}")

    # --- save ---
    model_dir = "llama3_instruct" if model_key == "instruct" else "llama3_base"
    out_dir = os.path.join(SCRIPT_DIR, "results", model_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cross_task_{method}_ablate{ablate_n}.json")

    output = {
        "experiment": "cross_task_ablation",
        "method": method,
        "model_key": model_key,
        "model_name": model_name,
        "ablate_n": ablate_n,
        "random_seeds": args.random_seeds,
        "task_order": TASK_ORDER,
        "token_lengths": TOKEN_LENGTHS,
        "question_groups": {t: QUESTIONS[t]["group"] for t in TASK_ORDER},
        "head_overlap": {},
        "timestamp": datetime.now().isoformat(),
        "results": {str(tl): v for tl, v in all_results.items()},
    }

    for tl in TOKEN_LENGTHS:
        overlaps = {}
        for t1 in TASK_ORDER:
            for t2 in TASK_ORDER:
                s1 = set(source_heads[(t1, tl)])
                s2 = set(source_heads[(t2, tl)])
                overlaps[f"{t1}->{t2}"] = len(s1 & s2)
        output["head_overlap"][str(tl)] = overlaps

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

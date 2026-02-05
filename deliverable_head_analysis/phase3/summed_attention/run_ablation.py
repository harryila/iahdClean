#!/usr/bin/env python3
"""
Phase 3 Method 1: Summed Attention Ablation Study

Tests causal importance of heads identified by the Summed Attention method
in Phase 2 by ablating them and measuring accuracy drop.

Features:
- Incremental ablation: Tests 5, 10, 20, 30, 40, 50 heads
- Random baseline: For each level, also ablates random heads for comparison
- Replicates Figure 8 pattern from Wu24 paper

The heads are loaded dynamically from Phase 2 Summed Attention results.

Ablation Mechanism:
- Uses forward pre-hooks on o_proj to zero out head outputs before projection
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

# Ablation levels for Figure 8 replication
ABLATION_LEVELS = [5, 10, 20, 30, 40, 50]

# Paths
PHASE1_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "phase1")
PHASE2_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "phase2", "summed_attention", "results")
GT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "edgar_gt_verified_slim.csv")

# Alice in Wonderland text for padding
ALICE_TEXT = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out again. The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs."""


# =============================================================================
# ABLATION MECHANISM
# =============================================================================

class HeadAblator:
    """
    Context manager for ablating specific attention heads.
    
    Ablation is done by zeroing out the attention output of specific heads
    BEFORE the output projection (o_proj).
    """
    
    def __init__(self, model, heads_to_ablate):
        """
        Args:
            model: The transformer model
            heads_to_ablate: List of (layer_idx, head_idx) tuples
        """
        self.model = model
        self.heads_to_ablate = heads_to_ablate
        self.hooks = []
        self.num_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
    
    def _make_ablation_pre_hook(self, layer_idx, heads_in_layer):
        """
        Create a PRE-hook on o_proj that zeros out specific heads in its INPUT.
        """
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
        """Register ablation hooks."""
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
        """Remove ablation hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_data():
    """Load test samples and ground truth."""
    test_path = os.path.join(PHASE1_DIR, "test_samples.json")
    with open(test_path, "r") as f:
        test_filenames = json.load(f)
    
    gt_df = pd.read_csv(GT_PATH)
    
    cache_path = os.path.join(PHASE1_DIR, "section1_cache.json")
    with open(cache_path, "r") as f:
        section1_cache = json.load(f)
    
    return test_filenames, gt_df, section1_cache


def load_top_heads_from_phase2(model_key, question_key, total_tokens, top_n=50):
    """
    Load top N heads from Phase 2 Summed Attention results.
    Returns list of (layer_idx, head_idx) tuples.
    """
    model_dir = "llama3_instruct" if model_key == "instruct" else "llama3_base"
    results_path = os.path.join(PHASE2_DIR, model_dir, question_key, f"tokens_{total_tokens}.json")
    
    if not os.path.exists(results_path):
        print(f"Warning: Phase 2 results not found at {results_path}")
        return []
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    # Parse head strings like "L15H30" into (15, 30) tuples
    heads = []
    for h in results["head_rankings"][:top_n]:
        head_str = h["head"]  # e.g., "L15H30"
        parts = head_str.replace("L", "").split("H")
        layer_idx = int(parts[0])
        head_idx = int(parts[1])
        heads.append((layer_idx, head_idx))
    
    return heads


def get_all_heads(model_config):
    """Get all possible (layer, head) pairs for random baseline."""
    num_layers = model_config.num_hidden_layers
    num_heads = model_config.num_attention_heads
    all_heads = []
    for layer in range(num_layers):
        for head in range(num_heads):
            all_heads.append((layer, head))
    return all_heads


def get_random_heads(all_heads, top_heads, n, seed=42):
    """
    Select N random heads, excluding the top heads.
    This ensures random baseline doesn't accidentally include important heads.
    """
    random.seed(seed)
    available = [h for h in all_heads if h not in top_heads]
    return random.sample(available, min(n, len(available)))


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def create_prompt_with_needle(tokenizer, section1_text, question_prompt, total_tokens, needle_position=0.5):
    """Create a prompt with needle in haystack at specified position."""
    needle_tokens = tokenizer.encode(section1_text, add_special_tokens=False)
    question_text = f"\n\nQuestion: {question_prompt}\nAnswer in one word:"
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    
    available_for_haystack = total_tokens - 1 - len(needle_tokens) - len(question_tokens)
    
    if available_for_haystack < 0:
        max_needle = total_tokens - 1 - len(question_tokens) - 100
        needle_tokens = needle_tokens[:max_needle]
        available_for_haystack = 100
    
    before_tokens_count = int(available_for_haystack * needle_position)
    after_tokens_count = available_for_haystack - before_tokens_count
    
    alice_tokens = tokenizer.encode(ALICE_TEXT, add_special_tokens=False)
    
    def get_haystack_tokens(count):
        if count <= 0:
            return []
        tokens = []
        while len(tokens) < count:
            tokens.extend(alice_tokens)
        return tokens[:count]
    
    before_tokens = get_haystack_tokens(before_tokens_count)
    after_tokens = get_haystack_tokens(after_tokens_count)
    
    full_tokens = [tokenizer.bos_token_id] + before_tokens + needle_tokens + after_tokens + question_tokens
    prompt = tokenizer.decode(full_tokens, skip_special_tokens=False)
    
    return prompt


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def generate_answer(model, tokenizer, prompt, max_new_tokens=10):
    """Generate an answer without ablation. Max 10 tokens for one-word answers."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()


def generate_answer_with_ablation(model, tokenizer, prompt, heads_to_ablate, max_new_tokens=10):
    """Generate an answer with specified heads ablated."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with HeadAblator(model, heads_to_ablate):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()


def normalize_answer(answer):
    """Normalize answer for comparison."""
    answer = str(answer).lower().strip()
    answer = answer.replace(",", "").replace(".", "").replace("'", "")
    try:
        if float(answer) == int(float(answer)):
            answer = str(int(float(answer)))
    except:
        pass
    return answer


def check_answer(generated, ground_truth):
    """Check if generated answer matches ground truth."""
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


def evaluate_accuracy(model, tokenizer, samples, question_config, total_tokens, heads_to_ablate=None):
    """
    Evaluate model accuracy on samples, optionally with head ablation.
    """
    correct = 0
    total = 0
    results = []
    
    for sample in tqdm(samples, desc="Evaluating", leave=False):
        try:
            prompt = create_prompt_with_needle(
                tokenizer,
                sample["section1"],
                question_config["prompt"],
                total_tokens
            )
            
            if heads_to_ablate:
                generated = generate_answer_with_ablation(
                    model, tokenizer, prompt, heads_to_ablate
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
                "correct": is_correct
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
    parser = argparse.ArgumentParser(description="Phase 3 Method 1: Summed Attention Ablation Study")
    parser.add_argument("--model", choices=["instruct", "base"], required=True)
    parser.add_argument("--question", choices=list(QUESTIONS.keys()), required=True)
    parser.add_argument("--tokens", type=int, default=2048, choices=[2048, 4096, 6144, 8192])
    parser.add_argument("--skip-random", action="store_true", help="Skip random baseline (faster)")
    args = parser.parse_args()
    
    model_key = args.model
    question_key = args.question
    total_tokens = args.tokens
    model_name = MODELS[model_key]
    question_config = QUESTIONS[question_key]
    
    print("=" * 70)
    print(f"Phase 3 Method 1 (Summed Attention) Ablation: {model_key} / {question_key} / {total_tokens} tokens")
    print(f"Ablation levels: {ABLATION_LEVELS}")
    print(f"Random baseline: {'SKIP' if args.skip_random else 'ENABLED'}")
    print("=" * 70)
    
    # Load data
    print("Loading data...")
    test_filenames, gt_df, section1_cache = load_test_data()
    
    # Filter to valid test samples
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
            "section1": section1_cache[filename]
        })
    
    print(f"Valid test samples for {question_key}: {len(valid_samples)}")
    
    # Load top heads from Phase 2 Summed Attention results
    max_heads_needed = max(ABLATION_LEVELS)
    top_heads_full = load_top_heads_from_phase2(model_key, question_key, total_tokens, top_n=max_heads_needed)
    print(f"Loaded {len(top_heads_full)} top heads from Phase 2 Summed Attention results")
    
    if len(top_heads_full) < max_heads_needed:
        print(f"Warning: Only {len(top_heads_full)} heads available, adjusting ablation levels")
        ABLATION_LEVELS_ADJUSTED = [n for n in ABLATION_LEVELS if n <= len(top_heads_full)]
    else:
        ABLATION_LEVELS_ADJUSTED = ABLATION_LEVELS
    
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
    
    # Get all heads for random baseline
    all_heads = get_all_heads(model.config)
    print(f"Total heads in model: {len(all_heads)} ({model.config.num_hidden_layers} layers x {model.config.num_attention_heads} heads)")
    
    results = {
        "method": "summed_attention_ablation_incremental",
        "model_key": model_key,
        "model_name": model_name,
        "question": question_key,
        "question_prompt": question_config["prompt"],
        "total_tokens": total_tokens,
        "test_samples": len(valid_samples),
        "ablation_levels": ABLATION_LEVELS_ADJUSTED,
        "heads_source": "phase2_summed_attention_results",
        "timestamp": datetime.now().isoformat(),
        "baseline": None,
        "top_heads_ablations": [],
        "random_heads_ablations": [],
    }
    
    # 1. Baseline (no ablation)
    print("\n" + "=" * 50)
    print("BASELINE (no ablation)")
    print("=" * 50)
    baseline_acc, baseline_results = evaluate_accuracy(
        model, tokenizer, valid_samples, question_config, total_tokens, heads_to_ablate=None
    )
    print(f"Baseline accuracy: {baseline_acc:.1%} ({sum(1 for r in baseline_results if r['correct'])}/{len(baseline_results)})")
    
    results["baseline"] = {
        "accuracy": baseline_acc,
        "correct": sum(1 for r in baseline_results if r["correct"]),
        "total": len(baseline_results)
    }
    
    # 2. Incremental ablation of top heads
    print("\n" + "=" * 50)
    print("TOP HEADS ABLATION (Incremental)")
    print("=" * 50)
    
    for n in ABLATION_LEVELS_ADJUSTED:
        heads_to_ablate = top_heads_full[:n]
        print(f"\n--- Ablating top {n} heads ---")
        
        ablated_acc, ablated_results = evaluate_accuracy(
            model, tokenizer, valid_samples, question_config, total_tokens, heads_to_ablate=heads_to_ablate
        )
        
        accuracy_drop = baseline_acc - ablated_acc
        print(f"Accuracy: {ablated_acc:.1%} (drop: {accuracy_drop:.1%})")
        
        results["top_heads_ablations"].append({
            "num_heads": n,
            "heads": [[h[0], h[1]] for h in heads_to_ablate],
            "heads_str": [f"L{h[0]}H{h[1]}" for h in heads_to_ablate],
            "accuracy": ablated_acc,
            "accuracy_drop": accuracy_drop,
            "correct": sum(1 for r in ablated_results if r["correct"]),
            "total": len(ablated_results)
        })
        
        torch.cuda.empty_cache()
    
    # 3. Random heads baseline (for comparison)
    if not args.skip_random:
        print("\n" + "=" * 50)
        print("RANDOM HEADS ABLATION (Baseline comparison)")
        print("=" * 50)
        
        for n in ABLATION_LEVELS_ADJUSTED:
            random_heads = get_random_heads(all_heads, top_heads_full, n)
            print(f"\n--- Ablating {n} random heads ---")
            
            random_acc, random_results = evaluate_accuracy(
                model, tokenizer, valid_samples, question_config, total_tokens, heads_to_ablate=random_heads
            )
            
            accuracy_drop = baseline_acc - random_acc
            print(f"Accuracy: {random_acc:.1%} (drop: {accuracy_drop:.1%})")
            
            results["random_heads_ablations"].append({
                "num_heads": n,
                "heads": [[h[0], h[1]] for h in random_heads],
                "heads_str": [f"L{h[0]}H{h[1]}" for h in random_heads],
                "accuracy": random_acc,
                "accuracy_drop": accuracy_drop,
                "correct": sum(1 for r in random_results if r["correct"]),
                "total": len(random_results)
            })
            
            torch.cuda.empty_cache()
    
    # Save results
    model_dir = "llama3_instruct" if model_key == "instruct" else "llama3_base"
    output_dir = os.path.join(os.path.dirname(__file__), "results", model_dir, question_key)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"tokens_{total_tokens}.json")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved: {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline accuracy: {baseline_acc:.1%}")
    print("\nTop Heads Ablation:")
    for abl in results["top_heads_ablations"]:
        print(f"  Top {abl['num_heads']:2d}: {abl['accuracy']:.1%} (drop: {abl['accuracy_drop']:.1%})")
    
    if results["random_heads_ablations"]:
        print("\nRandom Heads Ablation (baseline):")
        for abl in results["random_heads_ablations"]:
            print(f"  Random {abl['num_heads']:2d}: {abl['accuracy']:.1%} (drop: {abl['accuracy_drop']:.1%})")


if __name__ == "__main__":
    main()

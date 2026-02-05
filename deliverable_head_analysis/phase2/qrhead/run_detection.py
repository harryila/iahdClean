"""
Phase 2, Method 3: QRHead Detection

This script identifies retrieval heads using the QRHead method:
- Compute attention FROM query tokens TO needle region
- Calibrate by subtracting null query attention
- Apply outlier removal (mask tokens below mean - 2*std)

Key code adapted from:
- QRHead/exp_scripts/detection/detect_qrhead_lme.py
- QRHead/src/qrretriever/attn_retriever.py

PREREQUISITES:
    Run phase1/cache_section1_data.py first to cache Section 1 content!

Usage:
    python run_detection.py --model instruct --question inc_state --tokens 4096
"""

import os
import sys
import json
import argparse
import math
from collections import defaultdict
from datetime import datetime
from itertools import product

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE1_DIR = os.path.join(SCRIPT_DIR, "..", "..", "phase1")
GT_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "edgar_gt_verified_slim.csv")
CACHE_PATH = os.path.join(PHASE1_DIR, "section1_cache.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

MODELS = {
    "instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "base": "meta-llama/Meta-Llama-3-8B",
}

TOKEN_LENGTHS = [2048, 4096, 6144, 8192]

QUESTIONS = {
    "inc_state": {
        "column": "original_Inc_state_truth",
        "prompt": "What state was the company incorporated in?",
    },
    "inc_year": {
        "column": "original_Inc_year_truth",
        "prompt": "What year was the company incorporated?",
    },
    "employee_count": {
        "column": "employee_count_truth",
        "prompt": "How many employees does the company have?",
    },
    "hq_state": {
        "column": "headquarters_state_truth",
        "prompt": "What state is the company headquarters located in?",
    },
}

# Null query for calibration (from QRHead)
NULL_QUERY = "N/A"

# Alice in Wonderland haystack
ALICE_TEXT = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it.

In another moment down went Alice after it, never once considering how in the world she was to get out again. The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.

Down, down, down. Would the fall never come to an end? "I wonder how many miles I've fallen by this time?" she said aloud. "I must be getting somewhere near the centre of the earth."

Alice was not a bit hurt, and she jumped up on to her feet in a moment: she looked up, but it was all dark overhead; before her was another long passage, and the White Rabbit was still in sight, hurrying down it. There was not a moment to be lost: away went Alice like the wind."""

# =============================================================================
# DATA LOADING
# =============================================================================

def load_section1_cache():
    """Load cached Section 1 data."""
    if not os.path.exists(CACHE_PATH):
        print(f"ERROR: Section 1 cache not found at {CACHE_PATH}")
        print("Please run: python phase1/cache_section1_data.py first")
        sys.exit(1)
    
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    print(f"Loaded Section 1 cache: {len(cache)} files")
    return cache

def load_training_samples():
    """Load training filenames from Phase 1 split."""
    path = os.path.join(PHASE1_DIR, "train_samples.json")
    with open(path) as f:
        return json.load(f)

def load_ground_truth():
    """Load GT CSV."""
    return pd.read_csv(GT_PATH)

# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def get_haystack_tokens(tokenizer, total_needed):
    """Get enough haystack tokens by repeating Alice text."""
    haystack = ""
    while len(tokenizer.encode(haystack, add_special_tokens=False)) < total_needed:
        haystack += ALICE_TEXT + "\n\n"
    return tokenizer.encode(haystack, add_special_tokens=False)[:total_needed]

def create_prompt_with_needle(tokenizer, section1_text, question_prompt, total_tokens, needle_position=0.5):
    """
    Create a prompt with needle in haystack at specified position.
    
    Returns: (full_prompt, full_tokens, needle_start, needle_end, query_start, query_end)
    """
    # Tokenize components
    needle_tokens = tokenizer.encode(section1_text, add_special_tokens=False)
    question_text = f"\n\nQuestion: {question_prompt}\nAnswer in one word:"
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    
    # Calculate haystack size needed
    available_for_haystack = total_tokens - 1 - len(needle_tokens) - len(question_tokens)
    
    if available_for_haystack < 0:
        max_needle = total_tokens - 1 - len(question_tokens) - 100
        needle_tokens = needle_tokens[:max_needle]
        available_for_haystack = 100
    
    # Split haystack before/after needle
    before_tokens_count = int(available_for_haystack * needle_position)
    after_tokens_count = available_for_haystack - before_tokens_count
    
    haystack_tokens = get_haystack_tokens(tokenizer, available_for_haystack + 1000)
    before_tokens = haystack_tokens[:before_tokens_count]
    after_tokens = haystack_tokens[before_tokens_count:before_tokens_count + after_tokens_count]
    
    # Build full sequence
    bos_token = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []
    
    full_tokens = bos_token + before_tokens + needle_tokens + after_tokens + question_tokens
    
    # Calculate positions
    needle_start = len(bos_token) + len(before_tokens)
    needle_end = needle_start + len(needle_tokens)
    
    # Query starts after "Question: " - find the question text start
    query_start = len(bos_token) + len(before_tokens) + len(needle_tokens) + len(after_tokens)
    query_end = len(full_tokens)
    
    full_prompt = tokenizer.decode(full_tokens, skip_special_tokens=False)
    
    return full_prompt, full_tokens, needle_start, needle_end, query_start, query_end

def create_null_query_prompt(tokenizer, section1_text, total_tokens, needle_position=0.5):
    """Create prompt with null query for calibration."""
    return create_prompt_with_needle(
        tokenizer, section1_text, NULL_QUERY, total_tokens, needle_position
    )

# =============================================================================
# QRHEAD SCORING (Adapted from QRHead/src/qrretriever/attn_retriever.py)
# =============================================================================

def compute_query_to_needle_attention(model, tokenizer, full_tokens, query_start, query_end, needle_start, needle_end):
    """
    Compute attention FROM query tokens TO needle region.
    
    Returns: (num_layers, num_heads, needle_length) tensor
             Averaged over query tokens
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor([full_tokens], device=device)
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            return_dict=True
        )
    
    # Collect attention from query tokens to needle region
    # Shape per layer: (batch, num_heads, seq_len, seq_len)
    query_to_needle_attn = []
    
    for layer_idx in range(num_layers):
        layer_attn = outputs.attentions[layer_idx]  # (1, num_heads, seq_len, seq_len)
        
        # Extract: attention FROM query tokens TO needle tokens
        # query_attn shape: (num_heads, num_query_tokens, needle_length)
        query_attn = layer_attn[0, :, query_start:query_end, needle_start:needle_end]
        
        # Average over query tokens (following QRHead pattern)
        # Shape: (num_heads, needle_length)
        query_attn_avg = query_attn.mean(dim=1)
        
        query_to_needle_attn.append(query_attn_avg)
    
    # Stack layers: (num_layers, num_heads, needle_length)
    result = torch.stack(query_to_needle_attn, dim=0)
    
    return result.float()  # Ensure float for subsequent operations

def compute_query_to_needle_attention_efficient(model, tokenizer, full_tokens, query_start, query_end, needle_start, needle_end):
    """
    Memory-efficient version using forward hooks.
    Computes attention from query tokens to needle region only.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor([full_tokens], device=device)
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)
    head_dim = model.config.hidden_size // num_heads
    
    query_to_needle_attn = {}
    
    def make_hook(layer_idx):
        def hook(module, args, kwargs, output):
            hidden_states = args[0] if args else kwargs.get('hidden_states')
            if hidden_states is not None:
                bsz, seq_len, _ = hidden_states.size()
                
                # Get Q and K projections
                q = module.q_proj(hidden_states)
                k = module.k_proj(hidden_states)
                
                # Reshape
                q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
                k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                
                # Handle GQA
                if num_kv_heads != num_heads:
                    n_rep = num_heads // num_kv_heads
                    k = k.unsqueeze(2).expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
                    k = k.reshape(bsz, num_heads, seq_len, head_dim)
                
                # Extract query tokens only
                q_query = q[:, :, query_start:query_end, :]  # (1, num_heads, num_query_tokens, head_dim)
                
                # Compute attention: Q_query @ K^T
                attn_weights = torch.matmul(q_query, k.transpose(-2, -1)) / math.sqrt(head_dim)
                attn_weights = torch.softmax(attn_weights, dim=-1)
                
                # Extract attention to needle region and average over query tokens
                # Shape: (num_heads, num_query_tokens, needle_length)
                needle_attn = attn_weights[0, :, :, needle_start:needle_end]
                
                # Average over query tokens: (num_heads, needle_length)
                needle_attn_avg = needle_attn.mean(dim=1).float()
                
                query_to_needle_attn[layer_idx] = needle_attn_avg.cpu()
                
                del q, k, q_query, attn_weights, needle_attn
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        hook = layer.self_attn.register_forward_hook(make_hook(layer_idx), with_kwargs=True)
        hooks.append(hook)
    
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, output_attentions=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
    
    # Stack results: (num_layers, num_heads, needle_length)
    result = torch.stack([query_to_needle_attn[i] for i in range(num_layers)], dim=0)
    
    del input_ids
    torch.cuda.empty_cache()
    
    return result

def compute_qrhead_scores(actual_attn, null_attn, num_layers, num_heads):
    """
    Compute QRHead scores with calibration and outlier removal.
    
    Adapted from QRHead/src/qrretriever/attn_retriever.py lines 291-312
    
    Args:
        actual_attn: (num_layers, num_heads, needle_length) - attention with actual query
        null_attn: (num_layers, num_heads, needle_length) - attention with null query
    
    Returns:
        dict mapping head -> score
    """
    # Step 1: Calibration - subtract null query attention
    # From attn_retriever.py lines 291-292
    min_length = min(actual_attn.shape[-1], null_attn.shape[-1])
    calibrated_attn = actual_attn[:, :, :min_length] - null_attn[:, :, :min_length]
    
    head_scores = {}
    
    # Step 2: For each head, apply outlier removal and sum
    # From attn_retriever.py lines 304-312
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            # Get calibrated scores for this head
            scores = calibrated_attn[layer_idx, head_idx]  # (needle_length,)
            
            # Outlier removal: threshold = mean - 2*std
            threshold = scores.mean() - 2 * scores.std()
            
            # Mask: keep only scores above threshold
            mask = scores > threshold
            
            # Sum masked scores
            masked_scores = scores * mask.float()
            head_score = masked_scores.sum().item()
            
            head_key = f"L{layer_idx}H{head_idx}"
            head_scores[head_key] = head_score
    
    return head_scores

# =============================================================================
# HEAD SCORING AGGREGATION (Adapted from detect_qrhead_lme.py)
# =============================================================================

def lme_eval(head_scores_per_sample):
    """
    Aggregate head scores across samples.
    
    Adapted from QRHead/exp_scripts/detection/detect_qrhead_lme.py lines 9-32
    
    Args:
        head_scores_per_sample: list of dicts, each mapping head -> score
    
    Returns:
        dict mapping head -> mean score across samples
    """
    # Collect all scores per head
    all_head_scores = defaultdict(list)
    
    for sample_scores in head_scores_per_sample:
        for head, score in sample_scores.items():
            all_head_scores[head].append(score)
    
    # Compute mean score per head
    mean_scores = {}
    for head, scores in all_head_scores.items():
        mean_scores[head] = np.mean(scores)
    
    return mean_scores

def score_heads(head_scores_per_sample):
    """
    Score and rank all heads.
    
    Adapted from QRHead/exp_scripts/detection/detect_qrhead_lme.py lines 66-101
    """
    mean_scores = lme_eval(head_scores_per_sample)
    
    # Sort by score descending
    head_scores_list = [(head, score) for head, score in mean_scores.items()]
    head_scores_list.sort(key=lambda x: x[1], reverse=True)
    
    return head_scores_list

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(model_key, question_key, total_tokens):
    """Run a single experiment configuration."""
    
    model_name = MODELS[model_key]
    question_config = QUESTIONS[question_key]
    
    print("=" * 70)
    print(f"Experiment: model={model_key}, question={question_key}, tokens={total_tokens}")
    print("=" * 70)
    
    # Load data
    section1_cache = load_section1_cache()
    train_filenames = load_training_samples()
    gt_df = load_ground_truth()
    
    # Filter to training samples with valid GT
    gt_column = question_config["column"]
    valid_samples = []
    
    for filename in train_filenames:
        row = gt_df[gt_df["filename"] == filename]
        if len(row) == 0:
            continue
        gt_value = row[gt_column].values[0]
        if pd.isna(gt_value):
            continue
        if filename not in section1_cache:
            continue
        valid_samples.append({
            "filename": filename,
            "gt_value": str(gt_value),
            "section1": section1_cache[filename]
        })
    
    print(f"Valid samples for {question_key}: {len(valid_samples)}")
    
    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    # Choose method based on sequence length
    use_efficient = total_tokens >= 6144
    if use_efficient:
        print("Using memory-efficient attention computation for long sequences")
        compute_fn = compute_query_to_needle_attention_efficient
    else:
        compute_fn = compute_query_to_needle_attention
    
    # Collect scores for all samples
    all_sample_scores = []
    
    for sample in tqdm(valid_samples, desc="Processing"):
        try:
            # Create prompt with actual query
            prompt, tokens, needle_start, needle_end, query_start, query_end = create_prompt_with_needle(
                tokenizer=tokenizer,
                section1_text=sample["section1"],
                question_prompt=question_config["prompt"],
                total_tokens=total_tokens,
                needle_position=0.5
            )
            
            # Create prompt with null query for calibration
            null_prompt, null_tokens, null_needle_start, null_needle_end, null_query_start, null_query_end = create_prompt_with_needle(
                tokenizer=tokenizer,
                section1_text=sample["section1"],
                question_prompt=NULL_QUERY,
                total_tokens=total_tokens,
                needle_position=0.5
            )
            
            # Compute attention with actual query
            actual_attn = compute_fn(
                model, tokenizer, tokens,
                query_start, query_end, needle_start, needle_end
            )
            
            # Compute attention with null query
            null_attn = compute_fn(
                model, tokenizer, null_tokens,
                null_query_start, null_query_end, null_needle_start, null_needle_end
            )
            
            # Compute QRHead scores (calibration + outlier removal)
            head_scores = compute_qrhead_scores(actual_attn, null_attn, num_layers, num_heads)
            
            all_sample_scores.append(head_scores)
            
        except Exception as e:
            print(f"Error processing {sample['filename']}: {e}")
            continue
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    # Aggregate scores across samples (LME evaluation)
    head_scores_list = score_heads(all_sample_scores)
    
    # Build rankings
    head_rankings = []
    for rank, (head, score) in enumerate(head_scores_list, 1):
        head_rankings.append({
            "head": head,
            "score": score,
            "rank": rank
        })
    
    # Prepare results
    results = {
        "method": "qrhead",
        "model_key": model_key,
        "model_name": model_name,
        "question": question_key,
        "question_prompt": question_config["prompt"],
        "total_tokens": total_tokens,
        "needle_position": 0.5,
        "null_query": NULL_QUERY,
        "samples_processed": len(all_sample_scores),
        "timestamp": datetime.now().isoformat(),
        "head_rankings": head_rankings,
        "top_50_heads": [h["head"] for h in head_rankings[:50]]
    }
    
    # Save results
    model_dir = "llama3_instruct" if model_key == "instruct" else "llama3_base"
    output_dir = os.path.join(RESULTS_DIR, model_dir, question_key)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"tokens_{total_tokens}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved: {output_path}")
    print(f"\nTop 10 heads:")
    for entry in head_rankings[:10]:
        print(f"  {entry['head']}: {entry['score']:.4f}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="QRHead Detection")
    parser.add_argument("--model", choices=["instruct", "base"], help="Model to use")
    parser.add_argument("--question", choices=list(QUESTIONS.keys()), help="Question type")
    parser.add_argument("--tokens", type=int, choices=TOKEN_LENGTHS, help="Total token length")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    
    args = parser.parse_args()
    
    if args.all:
        for model_key in MODELS:
            for question_key in QUESTIONS:
                for tokens in TOKEN_LENGTHS:
                    # Check if already done
                    model_dir = "llama3_instruct" if model_key == "instruct" else "llama3_base"
                    output_path = os.path.join(RESULTS_DIR, model_dir, question_key, f"tokens_{tokens}.json")
                    if os.path.exists(output_path):
                        print(f"SKIP: {output_path} exists")
                        continue
                    
                    run_experiment(model_key, question_key, tokens)
    else:
        if not all([args.model, args.question, args.tokens]):
            parser.error("Must specify --model, --question, and --tokens (or use --all)")
        
        run_experiment(args.model, args.question, args.tokens)

if __name__ == "__main__":
    main()

"""
Phase 2, Method 1: Summed Attention Head Detection

This script identifies important attention heads by measuring how much each head
attends to the needle region (Section 1) containing the GT answer.

PREREQUISITES:
    Run phase1/cache_section1_data.py first to cache Section 1 content!

Usage:
    python run_detection.py --model instruct --question inc_state --tokens 4096
    python run_detection.py --all
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from datetime import datetime

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

# Alice in Wonderland haystack (repeated to get enough tokens)
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

def get_valid_samples(train_samples, gt_df, section1_cache, question_key):
    """Get training samples with valid GT and Section 1 content."""
    col = QUESTIONS[question_key]["column"]
    valid = []
    
    for filename in train_samples:
        # Check GT exists
        row = gt_df[gt_df["filename"] == filename]
        if len(row) == 0:
            continue
        
        gt_value = row[col].values[0]
        if pd.isna(gt_value) or str(gt_value).upper() in ["NULL", "NAN", "NONE", ""]:
            continue
        
        # Check Section 1 cached
        if filename not in section1_cache:
            continue
        
        valid.append({
            "filename": filename,
            "gt_value": str(gt_value).strip(),
            "section_1": section1_cache[filename],
        })
    
    return valid

# =============================================================================
# NEEDLE IN HAYSTACK
# =============================================================================

def get_haystack_tokens(tokenizer, min_tokens=10000):
    """Get tokenized haystack, expanded to min_tokens."""
    text = ALICE_TEXT
    while True:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= min_tokens:
            return tokens
        text = text + "\n\n" + ALICE_TEXT

def create_prompt_with_needle(
    section_1: str,
    tokenizer,
    total_tokens: int,
    question_prompt: str,
    needle_position: float = 0.5
):
    """
    Create prompt with needle (Section 1) embedded in haystack.
    
    Returns:
        (prompt_str, needle_start_token, needle_end_token)
    """
    # Tokenize the needle (Section 1)
    needle_tokens = tokenizer.encode(section_1, add_special_tokens=False)
    
    # Tokenize the question part
    question_text = f"\n\nQuestion: {question_prompt}\nAnswer in one word:"
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    
    # Calculate haystack tokens needed
    # total = haystack_before + needle + haystack_after + question
    available_for_haystack = total_tokens - len(needle_tokens) - len(question_tokens)
    
    if available_for_haystack <= 0:
        # Needle too long, truncate it
        max_needle = total_tokens - len(question_tokens) - 200
        needle_tokens = needle_tokens[:max_needle]
        available_for_haystack = 200
    
    # Split haystack before/after needle based on position
    tokens_before = int(available_for_haystack * needle_position)
    tokens_after = available_for_haystack - tokens_before
    
    # Get haystack tokens
    haystack_tokens = get_haystack_tokens(tokenizer)
    before_tokens = haystack_tokens[:tokens_before]
    after_tokens = haystack_tokens[tokens_before:tokens_before + tokens_after]
    
    # Calculate needle boundaries in final sequence
    needle_start = len(before_tokens)
    needle_end = needle_start + len(needle_tokens)
    
    # Build full token sequence
    full_tokens = before_tokens + needle_tokens + after_tokens + question_tokens
    
    # Decode to text
    prompt = tokenizer.decode(full_tokens, skip_special_tokens=True)
    
    return prompt, needle_start, needle_end

# =============================================================================
# MODEL AND ATTENTION
# =============================================================================

def load_model(model_key):
    """Load model and tokenizer."""
    model_name = MODELS[model_key]
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    
    return model, tokenizer

def compute_attention_to_needle(model, tokenizer, prompt, needle_start, needle_end):
    """
    Forward pass and compute attention from last token to needle.
    Standard method - works well for shorter sequences (2k, 4k).
    
    Returns:
        dict: {head_key: attention_score} for all heads
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    
    # Clamp needle bounds
    needle_start = max(0, min(needle_start, seq_len - 1))
    needle_end = max(needle_start + 1, min(needle_end, seq_len))
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)
    
    head_scores = {}
    
    for layer_idx, layer_attn in enumerate(outputs.attentions):
        # layer_attn: (batch, n_heads, seq_len, seq_len)
        # Get attention from last token to all positions
        last_pos_attn = layer_attn[0, :, -1, :].float()  # (n_heads, seq_len)
        
        for head_idx in range(last_pos_attn.shape[0]):
            # Sum attention to needle region
            score = last_pos_attn[head_idx, needle_start:needle_end].sum().item()
            head_key = f"L{layer_idx}H{head_idx}"
            head_scores[head_key] = score
    
    # Free memory
    del outputs, inputs
    
    return head_scores


def compute_attention_to_needle_memory_efficient(model, tokenizer, prompt, needle_start, needle_end):
    """
    Memory-efficient version for longer sequences (6k, 8k).
    Only computes attention for the LAST query token to save memory.
    Uses manual attention computation with Q from last position only.
    Handles Grouped Query Attention (GQA) where num_kv_heads < num_q_heads.
    
    Returns:
        dict: {head_key: attention_score} for all heads
    """
    import math
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]
    
    # Clamp needle bounds
    needle_start = max(0, min(needle_start, seq_len - 1))
    needle_end = max(needle_start + 1, min(needle_end, seq_len))
    
    head_scores = {}
    
    # Use hooks to compute attention from last token only
    def make_hook(layer_idx):
        def hook(module, args, kwargs, output):
            # Get the Q, K from the attention module
            # hidden_states is first positional arg
            hidden_states = args[0] if args else kwargs.get('hidden_states')
            
            if hidden_states is not None:
                bsz, q_len, _ = hidden_states.size()
                
                # Get Q, K projections
                q = module.q_proj(hidden_states)
                k = module.k_proj(hidden_states)
                
                # Handle GQA: num_heads (Q) vs num_key_value_heads (K/V)
                num_heads = module.num_heads
                num_kv_heads = getattr(module, 'num_key_value_heads', num_heads)
                head_dim = module.head_dim
                
                # Reshape Q: (bsz, q_len, num_heads * head_dim) -> (bsz, num_heads, q_len, head_dim)
                q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                
                # Reshape K: (bsz, q_len, num_kv_heads * head_dim) -> (bsz, num_kv_heads, q_len, head_dim)
                k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
                
                # Expand K to match Q heads (GQA: repeat each KV head for grouped Q heads)
                if num_kv_heads != num_heads:
                    n_rep = num_heads // num_kv_heads
                    k = k.unsqueeze(2).expand(bsz, num_kv_heads, n_rep, q_len, head_dim)
                    k = k.reshape(bsz, num_heads, q_len, head_dim)
                
                # Only compute attention for last query position
                q_last = q[:, :, -1:, :]  # (bsz, num_heads, 1, head_dim)
                
                # Compute attention scores: Q_last @ K^T
                attn_weights = torch.matmul(q_last, k.transpose(-2, -1)) / math.sqrt(head_dim)
                attn_weights = torch.softmax(attn_weights, dim=-1)  # (bsz, num_heads, 1, seq_len)
                
                # Extract scores for needle region
                needle_attn = attn_weights[0, :, 0, needle_start:needle_end].float()  # (num_heads, needle_len)
                
                for head_idx in range(num_heads):
                    score = needle_attn[head_idx].sum().item()
                    head_key = f"L{layer_idx}H{head_idx}"
                    head_scores[head_key] = score
                
                # Free memory
                del q, k, q_last, attn_weights, needle_attn
                
        return hook
    
    hooks = []
    # Register hooks with return_inputs
    for layer_idx, layer in enumerate(model.model.layers):
        hook = layer.self_attn.register_forward_hook(make_hook(layer_idx), with_kwargs=True)
        hooks.append(hook)
    
    try:
        with torch.no_grad():
            # Run forward WITHOUT output_attentions to save memory
            _ = model(**inputs, output_attentions=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
    
    del inputs
    torch.cuda.empty_cache()
    
    return head_scores

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(model_key, question_key, total_tokens):
    """Run head detection for one configuration."""
    print("=" * 70)
    print(f"Experiment: model={model_key}, question={question_key}, tokens={total_tokens}")
    print("=" * 70)
    
    # Load data
    section1_cache = load_section1_cache()
    train_samples = load_training_samples()
    gt_df = load_ground_truth()
    
    samples = get_valid_samples(train_samples, gt_df, section1_cache, question_key)
    print(f"Valid samples for {question_key}: {len(samples)}")
    
    if len(samples) == 0:
        print("ERROR: No valid samples found!")
        return None
    
    # Load model
    model, tokenizer = load_model(model_key)
    
    # Accumulate head scores
    accumulated_scores = defaultdict(float)
    question_prompt = QUESTIONS[question_key]["prompt"]
    processed = 0
    
    # Choose attention computation method based on token length
    # Use memory-efficient method for 6k+ tokens
    use_memory_efficient = total_tokens >= 6000
    if use_memory_efficient:
        print("Using memory-efficient attention computation for long sequences")
    
    for sample in tqdm(samples, desc="Processing"):
        try:
            prompt, needle_start, needle_end = create_prompt_with_needle(
                section_1=sample["section_1"],
                tokenizer=tokenizer,
                total_tokens=total_tokens,
                question_prompt=question_prompt,
                needle_position=0.5,
            )
            
            # Use appropriate method based on sequence length
            if use_memory_efficient:
                head_scores = compute_attention_to_needle_memory_efficient(
                    model, tokenizer, prompt, needle_start, needle_end
                )
            else:
                head_scores = compute_attention_to_needle(
                    model, tokenizer, prompt, needle_start, needle_end
                )
            
            for head_key, score in head_scores.items():
                accumulated_scores[head_key] += score
            
            processed += 1
            
            # Clear GPU cache to avoid memory fragmentation
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error on {sample['filename']}: {e}")
            torch.cuda.empty_cache()
            continue
    
    # Rank heads
    ranked = sorted(accumulated_scores.items(), key=lambda x: -x[1])
    
    # Prepare results
    results = {
        "method": "summed_attention",
        "model_key": model_key,
        "model_name": MODELS[model_key],
        "question": question_key,
        "question_prompt": question_prompt,
        "total_tokens": total_tokens,
        "needle_position": 0.5,
        "samples_processed": processed,
        "timestamp": datetime.now().isoformat(),
        "head_rankings": [
            {"head": h, "score": float(s), "rank": i + 1}
            for i, (h, s) in enumerate(ranked)
        ],
        "top_50_heads": [h for h, s in ranked[:50]],
    }
    
    # Save
    model_dir = "llama3_instruct" if model_key == "instruct" else "llama3_base"
    output_dir = os.path.join(RESULTS_DIR, model_dir, question_key)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"tokens_{total_tokens}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")
    
    # Print top 10
    print(f"\nTop 10 heads:")
    for h, s in ranked[:10]:
        print(f"  {h}: {s:.4f}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results

def run_all():
    """Run all 32 experiments."""
    for model_key in ["instruct", "base"]:
        for question_key in QUESTIONS.keys():
            for tokens in TOKEN_LENGTHS:
                run_experiment(model_key, question_key, tokens)

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["instruct", "base"])
    parser.add_argument("--question", choices=list(QUESTIONS.keys()))
    parser.add_argument("--tokens", type=int, choices=TOKEN_LENGTHS)
    parser.add_argument("--all", action="store_true")
    
    args = parser.parse_args()
    
    if args.all:
        run_all()
    elif args.model and args.question and args.tokens:
        run_experiment(args.model, args.question, args.tokens)
    else:
        print("Usage:")
        print("  python run_detection.py --model instruct --question inc_state --tokens 4096")
        print("  python run_detection.py --all")
        sys.exit(1)

if __name__ == "__main__":
    main()

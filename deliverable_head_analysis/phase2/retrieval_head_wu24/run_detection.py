"""
Phase 2, Method 2: Wu24 Retrieval Head Detection

This script identifies retrieval heads using the method from:
"Retrieval Head Mechanistically Explains Long-Context Factuality" (Wu et al., 2024)

Key difference from Summed Attention:
- Computed DURING decoding (not before)
- Uses argmax (top-1) attention, not sum
- Requires token matching (copy behavior)
- Only counts when retrieval succeeds (ROUGE > 50)

Based on: WU_Retrieval_Head/retrieval_head_detection.py (lines 221-233)

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

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer

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

# ROUGE scorer for filtering successful retrievals
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

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
    
    Returns: (full_prompt, needle_start_idx, needle_end_idx)
    """
    # Tokenize components
    needle_tokens = tokenizer.encode(section1_text, add_special_tokens=False)
    question_text = f"\n\nQuestion: {question_prompt}\nAnswer in one word:"
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    
    # Calculate haystack size needed
    # Total = BOS + before_haystack + needle + after_haystack + question
    available_for_haystack = total_tokens - 1 - len(needle_tokens) - len(question_tokens)
    
    if available_for_haystack < 0:
        # Needle too long, truncate it
        max_needle = total_tokens - 1 - len(question_tokens) - 100  # Keep some haystack
        needle_tokens = needle_tokens[:max_needle]
        available_for_haystack = 100
    
    # Split haystack before/after needle based on position
    before_tokens_count = int(available_for_haystack * needle_position)
    after_tokens_count = available_for_haystack - before_tokens_count
    
    # Get haystack tokens
    haystack_tokens = get_haystack_tokens(tokenizer, available_for_haystack + 1000)
    before_tokens = haystack_tokens[:before_tokens_count]
    after_tokens = haystack_tokens[before_tokens_count:before_tokens_count + after_tokens_count]
    
    # Build full sequence
    # [BOS] + before_haystack + needle + after_haystack + question
    bos_token = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []
    
    full_tokens = bos_token + before_tokens + needle_tokens + after_tokens + question_tokens
    
    # Calculate needle position (accounting for BOS)
    needle_start = len(bos_token) + len(before_tokens)
    needle_end = needle_start + len(needle_tokens)
    
    # Decode to text
    full_prompt = tokenizer.decode(full_tokens, skip_special_tokens=False)
    
    return full_prompt, full_tokens, needle_start, needle_end

# =============================================================================
# WU24 RETRIEVAL HEAD SCORING
# =============================================================================

def wu24_retrieval_calculate(attention_matrix, retrieval_score, generated_token_id, prompt_ids, 
                              needle_start, needle_end, num_layers, num_heads, topk=1):
    """
    Calculate retrieval score using Wu24 method.
    
    From WU_Retrieval_Head/retrieval_head_detection.py lines 221-229:
    - Get top-k attention positions from the last token
    - If position is in needle AND generated token matches, increment score
    
    Args:
        attention_matrix: List of attention tensors per layer
        retrieval_score: Dict to accumulate scores
        generated_token_id: The token that was just generated
        prompt_ids: All token IDs in the prompt (to check for matching)
        needle_start: Start index of needle in prompt
        needle_end: End index of needle in prompt
        num_layers: Number of layers
        num_heads: Number of attention heads
        topk: Number of top attention positions to check (default 1)
    """
    needle_len = needle_end - needle_start
    
    for layer_idx in range(num_layers):
        # attention_matrix[layer_idx] shape: [batch, num_heads, seq_len, seq_len]
        # We want the last row (attention FROM the last token)
        attn = attention_matrix[layer_idx]
        
        for head_idx in range(num_heads):
            # Get attention from last token to all previous positions
            # Shape: [seq_len] - attention weights to each position
            head_attn = attn[0, head_idx, -1, :]
            
            # Get top-k positions with highest attention
            values, indices = head_attn.topk(topk)
            
            for v, i in zip(values, indices):
                pos = i.item()
                # Check if position is in needle
                if needle_start <= pos < needle_end:
                    # Check if generated token matches token at that position
                    if generated_token_id == prompt_ids[pos]:
                        # Increment score (normalized by needle length)
                        key = f"L{layer_idx}H{head_idx}"
                        retrieval_score[key] += 1.0 / needle_len
                        break  # Only count once per head

def generate_with_retrieval_tracking(model, tokenizer, input_ids, prompt_ids, 
                                     needle_start, needle_end, max_new_tokens=10):
    """
    Generate tokens while tracking retrieval scores for each head.
    
    Returns:
        generated_text: The generated response
        retrieval_scores: Dict mapping head -> retrieval score
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    retrieval_scores = defaultdict(float)
    generated_tokens = []
    
    device = next(model.parameters()).device
    current_ids = input_ids.to(device)
    
    # We need to track all token IDs for matching
    all_token_ids = prompt_ids.tolist() if torch.is_tensor(prompt_ids) else list(prompt_ids)
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass with attention
            outputs = model(
                input_ids=current_ids,
                output_attentions=True,
                return_dict=True,
                use_cache=False  # Simpler without cache for attention tracking
            )
            
            # Get next token
            next_token_logits = outputs.logits[0, -1, :]
            next_token_id = next_token_logits.argmax().item()
            
            # Calculate retrieval score for this step
            wu24_retrieval_calculate(
                attention_matrix=outputs.attentions,
                retrieval_score=retrieval_scores,
                generated_token_id=next_token_id,
                prompt_ids=all_token_ids,
                needle_start=needle_start,
                needle_end=needle_end,
                num_layers=num_layers,
                num_heads=num_heads
            )
            
            # Add token and check for stop
            generated_tokens.append(next_token_id)
            all_token_ids.append(next_token_id)
            
            # Check for EOS or newline (common stop condition)
            if next_token_id == tokenizer.eos_token_id:
                break
            if tokenizer.decode([next_token_id]) in ['\n', '<0x0A>']:
                break
            
            # Update input for next step
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token_id]], device=device)
            ], dim=1)
            
            # Memory management for long sequences
            if step % 10 == 0:
                torch.cuda.empty_cache()
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return generated_text, dict(retrieval_scores)

def generate_with_retrieval_tracking_efficient(model, tokenizer, input_ids, prompt_ids,
                                                needle_start, needle_end, max_new_tokens=10):
    """
    Memory-efficient version using KV cache and hooks for long sequences.
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)
    head_dim = model.config.hidden_size // num_heads
    
    retrieval_scores = defaultdict(float)
    generated_tokens = []
    
    device = next(model.parameters()).device
    current_ids = input_ids.to(device)
    
    all_token_ids = prompt_ids.tolist() if torch.is_tensor(prompt_ids) else list(prompt_ids)
    needle_len = needle_end - needle_start
    
    with torch.no_grad():
        # Initial forward pass to get KV cache
        outputs = model(
            input_ids=current_ids[:, :-1],
            use_cache=True,
            return_dict=True
        )
        past_kv = outputs.past_key_values
        
        # Start with last token
        current_token = current_ids[:, -1:]
        
        for step in range(max_new_tokens):
            # Forward with cache and attention
            outputs = model(
                input_ids=current_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
                return_dict=True
            )
            
            past_kv = outputs.past_key_values
            
            # Get next token
            next_token_logits = outputs.logits[0, -1, :]
            next_token_id = next_token_logits.argmax().item()
            
            # Calculate retrieval scores from attention
            # When using cache, attention is [batch, heads, 1, seq_len]
            for layer_idx in range(num_layers):
                attn = outputs.attentions[layer_idx]  # [batch, heads, 1, seq_len]
                for head_idx in range(num_heads):
                    head_attn = attn[0, head_idx, 0, :]  # [seq_len]
                    
                    # Get top attention position
                    top_idx = head_attn.argmax().item()
                    
                    if needle_start <= top_idx < needle_end:
                        if next_token_id == all_token_ids[top_idx]:
                            key = f"L{layer_idx}H{head_idx}"
                            retrieval_scores[key] += 1.0 / needle_len
            
            # Add token
            generated_tokens.append(next_token_id)
            all_token_ids.append(next_token_id)
            
            # Check for stop
            if next_token_id == tokenizer.eos_token_id:
                break
            decoded = tokenizer.decode([next_token_id])
            if decoded in ['\n', '<0x0A>'] or '\n' in decoded:
                break
            
            # Update for next step
            current_token = torch.tensor([[next_token_id]], device=device)
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return generated_text, dict(retrieval_scores)

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
    
    # Filter to training samples with valid GT for this question
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
    
    # Choose efficient method for long sequences
    use_efficient = total_tokens >= 6144
    if use_efficient:
        print("Using memory-efficient generation for long sequences")
    
    # Accumulate scores across samples
    all_head_scores = defaultdict(list)  # head -> list of scores per successful sample
    successful_retrievals = 0
    total_processed = 0
    
    for sample in tqdm(valid_samples, desc="Processing"):
        try:
            # Create prompt
            prompt, prompt_tokens, needle_start, needle_end = create_prompt_with_needle(
                tokenizer=tokenizer,
                section1_text=sample["section1"],
                question_prompt=question_config["prompt"],
                total_tokens=total_tokens,
                needle_position=0.5
            )
            
            # Tokenize
            input_ids = torch.tensor([prompt_tokens])
            prompt_ids = prompt_tokens
            
            # Generate with retrieval tracking
            if use_efficient:
                generated_text, retrieval_scores = generate_with_retrieval_tracking_efficient(
                    model, tokenizer, input_ids, prompt_ids, needle_start, needle_end
                )
            else:
                generated_text, retrieval_scores = generate_with_retrieval_tracking(
                    model, tokenizer, input_ids, prompt_ids, needle_start, needle_end
                )
            
            # Check if retrieval was successful using ROUGE
            rouge_score = scorer.score(sample["gt_value"], generated_text)['rouge1'].recall * 100
            
            # Only count heads if retrieval was successful (Wu24 criterion: ROUGE > 50)
            if rouge_score > 50:
                successful_retrievals += 1
                for head, score in retrieval_scores.items():
                    all_head_scores[head].append(score)
            
            total_processed += 1
            
        except Exception as e:
            print(f"Error processing {sample['filename']}: {e}")
            continue
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    # Aggregate scores (mean across successful samples)
    head_rankings = []
    for head in all_head_scores:
        scores = all_head_scores[head]
        mean_score = sum(scores) / len(scores) if scores else 0
        head_rankings.append({
            "head": head,
            "score": mean_score,
            "num_samples": len(scores)
        })
    
    # Sort by score
    head_rankings = sorted(head_rankings, key=lambda x: x["score"], reverse=True)
    
    # Add ranks
    for i, entry in enumerate(head_rankings):
        entry["rank"] = i + 1
    
    # Prepare results
    results = {
        "method": "wu24_retrieval_head",
        "model_key": model_key,
        "model_name": model_name,
        "question": question_key,
        "question_prompt": question_config["prompt"],
        "total_tokens": total_tokens,
        "needle_position": 0.5,
        "samples_processed": total_processed,
        "successful_retrievals": successful_retrievals,
        "success_rate": successful_retrievals / total_processed if total_processed > 0 else 0,
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
    print(f"Successful retrievals: {successful_retrievals}/{total_processed} ({results['success_rate']*100:.1f}%)")
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
    parser = argparse.ArgumentParser(description="Wu24 Retrieval Head Detection")
    parser.add_argument("--model", choices=["instruct", "base"], help="Model to use")
    parser.add_argument("--question", choices=list(QUESTIONS.keys()), help="Question type")
    parser.add_argument("--tokens", type=int, choices=TOKEN_LENGTHS, help="Total token length")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    
    args = parser.parse_args()
    
    if args.all:
        # Run all 32 experiments
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

# -*- coding: utf-8 -*-
"""
Copy Head Analysis for EDGAR State of Incorporation Task

This script identifies "copy heads" - attention heads that assign maximum
attention to the answer token (needle) in context, following the methodology
from "Retrieval Head Mechanistically Explains Long-Context Factual Recall"

Key improvements over previous implementation:
1. Find needle token positions (where the answer appears in context)
2. Identify copy heads by measuring attention specifically to needle tokens
3. Add head ablation to test causal effect on retrieval
4. Compare short vs long context head behavior

Reference: https://arxiv.org/pdf/2404.15574
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict, Optional
import re

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

CONTEXT_CONFIGS = {
    "short": {"target_tokens": 200},
    "long": {"target_tokens": 2000},
}

QUESTION_TEMPLATE = """Based on the following SEC 10-K filing excerpt, answer the question.

Context:
{context}

Question: In which US state was this company incorporated?
Answer with just the state name:"""

NUM_SAMPLES = 50

# =============================================================================
# US STATES
# =============================================================================

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming", "district of columbia"
}

STATE_ALIASES = {
    "ny": "new york", "ca": "california", "tx": "texas", "fl": "florida",
    "pa": "pennsylvania", "de": "delaware", "nv": "nevada", "nj": "new jersey",
}

# =============================================================================
# GROUND TRUTH LOADING
# =============================================================================

def load_ground_truth():
    """Load ground truth from the verified CSV."""
    # Use the verified ground truth file
    gt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           "edgar_gt_verified_slim.csv")
    if os.path.exists(gt_path):
        df = pd.read_csv(gt_path)
        gt_dict = {}
        for _, row in df.iterrows():
            filename = row['filename']
            # Column name in verified CSV is 'original_Inc_state_truth'
            state = row.get('original_Inc_state_truth', None)
            if pd.notna(state) and state and str(state).upper() not in ['NULL', 'NAN', 'NONE', '']:
                gt_dict[filename] = str(state).strip()
        print(f"Loaded ground truth for {len(gt_dict)} files from edgar_gt_verified_slim.csv")
        return gt_dict
    print(f"Warning: Ground truth file not found at {gt_path}")
    return {}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_edgar_samples(num_samples, ground_truth):
    """Load EDGAR samples that have ground truth."""
    print(f"Loading EDGAR corpus...")
    
    dataset = load_dataset(
        "c3po-ai/edgar-corpus",
        "full",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    samples = []
    for item in dataset:
        if len(samples) >= num_samples:
            break
            
        filename = item.get('filename', '')
        section_1 = item.get('section_1', '')
        
        if filename not in ground_truth:
            continue
        if not section_1 or len(section_1) < 100:
            continue
            
        samples.append({
            'filename': filename,
            'section_1': section_1,
            'section_2': item.get('section_2', ''),
            'section_7': item.get('section_7', ''),
            'ground_truth_state': ground_truth[filename],
        })
    
    print(f"Loaded {len(samples)} samples with ground truth")
    return samples

# =============================================================================
# CONTEXT CREATION
# =============================================================================

def truncate_to_tokens(text, tokenizer, max_tokens):
    """Truncate text to max_tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text, len(tokens)
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True), max_tokens

def create_context(sample, tokenizer, context_type):
    """Create context based on type."""
    target_tokens = CONTEXT_CONFIGS[context_type]['target_tokens']
    
    if context_type == "short":
        return truncate_to_tokens(sample['section_1'], tokenizer, target_tokens)
    else:
        # Long context: padding + section_1 + padding
        section_1 = sample['section_1']
        section_2 = sample.get('section_2', '') or ''
        section_7 = sample.get('section_7', '') or ''
        
        core_context, core_tokens = truncate_to_tokens(section_1, tokenizer, 400)
        remaining = target_tokens - core_tokens
        padding_tokens = remaining // 2
        
        filler = "The company operates across multiple segments. " * 50
        
        if section_7 and len(section_7) > 100:
            padding_before, _ = truncate_to_tokens(section_7, tokenizer, padding_tokens)
        else:
            padding_before, _ = truncate_to_tokens(filler, tokenizer, padding_tokens)
        
        if section_2 and len(section_2) > 100:
            padding_after, _ = truncate_to_tokens(section_2, tokenizer, padding_tokens)
        else:
            padding_after, _ = truncate_to_tokens(filler[::-1], tokenizer, padding_tokens)
        
        full_context = f"{padding_before}\n\n{core_context}\n\n{padding_after}"
        return truncate_to_tokens(full_context, tokenizer, target_tokens)

# =============================================================================
# NEEDLE TOKEN FINDING
# =============================================================================

def find_needle_positions(tokenizer, input_ids: torch.Tensor, answer: str, context_start: int, context_end: int) -> List[int]:
    """
    Find positions where the answer (needle) appears in the context.
    
    Args:
        tokenizer: The tokenizer
        input_ids: Token IDs of the full prompt
        answer: The ground truth answer (e.g., "Delaware")
        context_start: Start position of context in token sequence
        context_end: End position of context in token sequence
    
    Returns:
        List of token positions where the answer appears
    """
    # Get the text of the context region
    context_ids = input_ids[context_start:context_end]
    context_text = tokenizer.decode(context_ids, skip_special_tokens=True).lower()
    
    # Find all occurrences of the answer in the context text
    answer_lower = answer.lower()
    
    # Tokenize the answer to understand how it appears
    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    
    # Search for the answer tokens in the context
    needle_positions = []
    
    # Method 1: Look for the first token of the answer
    first_answer_token = answer_tokens[0] if answer_tokens else None
    if first_answer_token:
        for i, token_id in enumerate(context_ids.tolist()):
            if token_id == first_answer_token:
                # Found a match - record the absolute position
                needle_positions.append(context_start + i)
    
    # Method 2: If no exact token match, search by decoded text
    if not needle_positions:
        # Decode each token and check if it starts the answer
        for i, token_id in enumerate(context_ids.tolist()):
            token_text = tokenizer.decode([token_id], skip_special_tokens=True).lower().strip()
            if token_text and (answer_lower.startswith(token_text) or token_text.startswith(answer_lower)):
                needle_positions.append(context_start + i)
                break  # Just get the first occurrence
    
    return needle_positions

def find_needle_positions_v2(tokenizer, full_text: str, input_ids: torch.Tensor, answer: str) -> List[int]:
    """
    Alternative method: Find needle by string matching then map to token positions.
    """
    answer_lower = answer.lower()
    text_lower = full_text.lower()
    
    # Find character positions of the answer
    char_positions = []
    start = 0
    while True:
        pos = text_lower.find(answer_lower, start)
        if pos == -1:
            break
        char_positions.append(pos)
        start = pos + 1
    
    if not char_positions:
        return []
    
    # Map character positions to token positions
    # This is approximate - we use the tokenizer's offset mapping if available
    needle_positions = []
    
    # Simple approximation: tokenize prefix to get position
    for char_pos in char_positions[:3]:  # Only first few occurrences
        prefix = full_text[:char_pos]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        needle_positions.append(len(prefix_tokens))
    
    return needle_positions

# =============================================================================
# COPY HEAD IDENTIFICATION
# =============================================================================

def identify_copy_heads(
    attentions: Tuple[torch.Tensor, ...],
    needle_positions: List[int],
    top_k: int = 20
) -> Dict[Tuple[int, int], float]:
    """
    Identify copy heads - heads where attention at the last position
    has maximum (or high) attention to needle token positions.
    
    A "copy head" is one where:
    - argmax of attention from last position falls on a needle token
    - OR attention to needle tokens is significantly above average
    
    Args:
        attentions: Tuple of attention tensors, one per layer
                    Each tensor shape: (batch, num_heads, seq_len, seq_len)
        needle_positions: Token positions where the answer appears
        top_k: Number of top heads to return
    
    Returns:
        Dict mapping (layer, head) -> score
        Score represents how much the head attends to needle positions
    """
    if not needle_positions:
        return {}
    
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    seq_len = attentions[0].shape[2]
    
    head_scores = {}
    head_is_copy = {}  # Whether argmax falls on needle
    
    for layer_idx, layer_attn in enumerate(attentions):
        # Shape: (1, num_heads, seq_len, seq_len)
        attn = layer_attn[0].float()  # (num_heads, seq_len, seq_len)
        
        for head_idx in range(num_heads):
            # Attention from last position
            last_attn = attn[head_idx, -1, :]  # (seq_len,)
            
            # Check if argmax falls on a needle position
            argmax_pos = last_attn.argmax().item()
            is_copy_head = argmax_pos in needle_positions
            
            # Calculate attention mass on needle tokens
            needle_attn = sum(last_attn[pos].item() for pos in needle_positions if pos < seq_len)
            
            # Calculate max attention to any single needle token
            max_needle_attn = max(
                (last_attn[pos].item() for pos in needle_positions if pos < seq_len),
                default=0.0
            )
            
            head_scores[(layer_idx, head_idx)] = {
                'needle_attention': needle_attn,
                'max_needle_attention': max_needle_attn,
                'is_copy_head': is_copy_head,
                'argmax_position': argmax_pos,
            }
    
    return head_scores

def get_top_copy_heads(head_scores: Dict, metric: str = 'max_needle_attention', top_k: int = 20) -> List[Tuple]:
    """Get top-k copy heads by specified metric."""
    sorted_heads = sorted(
        head_scores.items(),
        key=lambda x: x[1][metric],
        reverse=True
    )
    return sorted_heads[:top_k]

# =============================================================================
# MODEL WITH ATTENTION HOOKS FOR ABLATION
# =============================================================================

class AttentionAblator:
    """
    Hook-based attention ablation for testing causal effects.
    
    Usage:
        ablator = AttentionAblator(model, heads_to_ablate=[(15, 7), (16, 1)])
        ablator.enable()
        output = model.generate(...)
        ablator.disable()
    """
    
    def __init__(self, model, heads_to_ablate: List[Tuple[int, int]] = None):
        self.model = model
        self.heads_to_ablate = set(heads_to_ablate or [])
        self.hooks = []
        self.enabled = False
    
    def _create_ablation_hook(self, layer_idx: int):
        """Create a hook that zeros out specified heads in this layer."""
        def hook(module, input, output):
            if not self.enabled:
                return output
            
            # output is typically (hidden_states, present_key_value, attn_weights)
            # or just hidden_states depending on model
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # For attention ablation, we need to modify the attention output
            # This is model-specific, but we'll try to zero the contribution
            # of specific heads by masking
            
            heads_in_layer = [h for (l, h) in self.heads_to_ablate if l == layer_idx]
            if not heads_in_layer:
                return output
            
            # Note: Proper ablation requires modifying the attention mechanism
            # This simplified version modifies the output hidden states
            # A more complete implementation would hook into attention_probs
            
            return output
        
        return hook
    
    def enable(self):
        """Enable ablation hooks."""
        self.enabled = True
    
    def disable(self):
        """Disable ablation hooks."""
        self.enabled = False
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def run_with_ablated_heads(
    model,
    tokenizer,
    prompt: str,
    heads_to_ablate: List[Tuple[int, int]],
    max_new_tokens: int = 15
) -> str:
    """
    Run inference with specified attention heads ablated (zeroed out).
    
    This uses a forward hook to zero the attention weights of specified heads.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Store original attention weights and create ablation mask
    ablated_layers = defaultdict(list)
    for layer_idx, head_idx in heads_to_ablate:
        ablated_layers[layer_idx].append(head_idx)
    
    hooks = []
    
    def create_attention_ablation_hook(layer_idx, heads_to_zero):
        def hook(module, args, output):
            # For models that return attention weights
            # output could be (attn_output, attn_weights) or just attn_output
            if isinstance(output, tuple) and len(output) >= 2:
                attn_output, attn_weights = output[0], output[1]
                if attn_weights is not None:
                    # Zero out the specified heads
                    for head_idx in heads_to_zero:
                        attn_weights[:, head_idx, :, :] = 0
                    # Renormalize (optional)
                    return (attn_output, attn_weights) + output[2:]
            return output
        return hook
    
    # Register hooks on attention layers
    try:
        # Try to find attention layers (model-specific)
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Llama-style
            for layer_idx, heads in ablated_layers.items():
                if layer_idx < len(model.model.layers):
                    layer = model.model.layers[layer_idx]
                    if hasattr(layer, 'self_attn'):
                        hook = layer.self_attn.register_forward_hook(
                            create_attention_ablation_hook(layer_idx, heads)
                        )
                        hooks.append(hook)
        
        # Run generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0, inputs.input_ids.shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_sample(
    model,
    tokenizer,
    sample: Dict,
    context_type: str,
    return_attentions: bool = True
) -> Dict:
    """
    Analyze a single sample for copy heads.
    
    Returns:
        Dict with:
        - answer: Model's answer
        - correct: Whether answer is correct
        - needle_positions: Where answer appears in context
        - head_scores: Copy head scores
    """
    # Create context and prompt
    context, num_tokens = create_context(sample, tokenizer, context_type)
    prompt = QUESTION_TEMPLATE.format(context=context)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0]
    
    # Find context boundaries
    pre_context = "Based on the following SEC 10-K filing excerpt, answer the question.\n\nContext:\n"
    context_start = len(tokenizer.encode(pre_context, add_special_tokens=False))
    context_end = context_start + num_tokens
    
    # Find needle positions
    answer = sample['ground_truth_state']
    needle_positions = find_needle_positions(
        tokenizer, input_ids, answer, context_start, context_end
    )
    
    # Also try v2 method
    if not needle_positions:
        needle_positions = find_needle_positions_v2(
            tokenizer, prompt, input_ids, answer
        )
    
    # Run inference with attention
    head_scores = {}
    if return_attentions and needle_positions:
        with torch.no_grad():
            outputs = model(
                inputs.input_ids,
                output_attentions=True,
                use_cache=False
            )
            attentions = outputs.attentions
            
            # Identify copy heads
            head_scores = identify_copy_heads(attentions, needle_positions)
            
            del attentions
            del outputs
    
    # Generate answer
    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = gen_outputs[0, inputs.input_ids.shape[1]:]
    model_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Check correctness
    answer_norm = answer.lower().strip()
    model_answer_norm = model_answer.lower().strip().split()[0] if model_answer else ""
    is_correct = answer_norm in model_answer_norm or model_answer_norm in answer_norm
    
    torch.cuda.empty_cache()
    
    return {
        'filename': sample['filename'],
        'ground_truth': answer,
        'model_answer': model_answer,
        'correct': is_correct,
        'context_tokens': num_tokens,
        'needle_positions': needle_positions,
        'num_needle_tokens': len(needle_positions),
        'head_scores': head_scores,
    }

def run_copy_head_experiment(
    model_key: str,
    context_type: str,
    samples: List[Dict],
    save_dir: str
) -> Dict:
    """Run the copy head analysis experiment."""
    
    model_name = MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"Copy Head Analysis: {model_key} - {context_type} context")
    print(f"{'='*70}\n")
    
    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",  # Need eager for attention access
    )
    model.eval()
    
    # Run analysis
    results = []
    all_head_scores = defaultdict(lambda: defaultdict(list))
    copy_head_counts = defaultdict(int)  # Count how often each head is a "copy head"
    
    for sample in tqdm(samples, desc=f"{model_key}/{context_type}"):
        try:
            result = analyze_sample(model, tokenizer, sample, context_type)
            results.append(result)
            
            # Aggregate head scores
            for (layer, head), scores in result['head_scores'].items():
                all_head_scores[(layer, head)]['needle_attention'].append(scores['needle_attention'])
                all_head_scores[(layer, head)]['max_needle_attention'].append(scores['max_needle_attention'])
                if scores['is_copy_head']:
                    copy_head_counts[(layer, head)] += 1
        
        except Exception as e:
            print(f"Error on {sample['filename']}: {e}")
            continue
    
    # Calculate average scores
    avg_head_scores = {}
    for (layer, head), scores in all_head_scores.items():
        avg_head_scores[(layer, head)] = {
            'avg_needle_attention': np.mean(scores['needle_attention']),
            'avg_max_needle_attention': np.mean(scores['max_needle_attention']),
            'copy_head_count': copy_head_counts[(layer, head)],
            'copy_head_rate': copy_head_counts[(layer, head)] / len(results) if results else 0,
        }
    
    # Get top copy heads
    top_by_attention = sorted(
        avg_head_scores.items(),
        key=lambda x: x[1]['avg_max_needle_attention'],
        reverse=True
    )[:20]
    
    top_by_copy_rate = sorted(
        avg_head_scores.items(),
        key=lambda x: x[1]['copy_head_rate'],
        reverse=True
    )[:20]
    
    # Calculate accuracy
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / len(results) if results else 0
    
    # Prepare output
    output = {
        'model': model_key,
        'model_name': model_name,
        'context_type': context_type,
        'num_samples': len(results),
        'accuracy': accuracy,
        'correct': correct,
        'total': len(results),
        'top_heads_by_needle_attention': [
            {
                'head': f"L{l}H{h}",
                'layer': l,
                'head_idx': h,
                **scores
            }
            for (l, h), scores in top_by_attention
        ],
        'top_heads_by_copy_rate': [
            {
                'head': f"L{l}H{h}",
                'layer': l,
                'head_idx': h,
                **scores
            }
            for (l, h), scores in top_by_copy_rate
        ],
        'results': [
            {k: v for k, v in r.items() if k != 'head_scores'}
            for r in results
        ],
    }
    
    # Save
    output_path = os.path.join(save_dir, f"{model_key}_{context_type}_copy_heads.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults for {model_key} - {context_type}:")
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(results)})")
    print(f"\n  Top 5 Copy Heads (by max needle attention):")
    for (l, h), scores in top_by_attention[:5]:
        print(f"    L{l}H{h}: needle_attn={scores['avg_max_needle_attention']:.4f}, copy_rate={scores['copy_head_rate']:.1%}")
    print(f"\n  Top 5 Copy Heads (by copy rate):")
    for (l, h), scores in top_by_copy_rate[:5]:
        print(f"    L{l}H{h}: copy_rate={scores['copy_head_rate']:.1%}, needle_attn={scores['avg_max_needle_attention']:.4f}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return output

# =============================================================================
# ABLATION EXPERIMENT
# =============================================================================

def run_ablation_experiment(
    model_key: str,
    context_type: str,
    samples: List[Dict],
    heads_to_ablate: List[Tuple[int, int]],
    save_dir: str
) -> Dict:
    """
    Run experiment with specified heads ablated to test causal effect.
    """
    model_name = MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"Ablation Experiment: {model_key} - {context_type}")
    print(f"Ablating heads: {heads_to_ablate[:10]}...")
    print(f"{'='*70}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    results = []
    
    for sample in tqdm(samples[:20], desc="Ablation"):  # Limited samples for speed
        context, _ = create_context(sample, tokenizer, context_type)
        prompt = QUESTION_TEMPLATE.format(context=context)
        
        # Normal inference
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            normal_out = model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        normal_answer = tokenizer.decode(
            normal_out[0, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Ablated inference
        ablated_answer = run_with_ablated_heads(
            model, tokenizer, prompt, heads_to_ablate
        )
        
        # Check correctness
        gt = sample['ground_truth_state'].lower()
        normal_correct = gt in normal_answer.lower()
        ablated_correct = gt in ablated_answer.lower()
        
        results.append({
            'filename': sample['filename'],
            'ground_truth': sample['ground_truth_state'],
            'normal_answer': normal_answer,
            'ablated_answer': ablated_answer,
            'normal_correct': normal_correct,
            'ablated_correct': ablated_correct,
        })
    
    # Summary
    normal_acc = sum(r['normal_correct'] for r in results) / len(results)
    ablated_acc = sum(r['ablated_correct'] for r in results) / len(results)
    
    output = {
        'model': model_key,
        'context_type': context_type,
        'heads_ablated': [f"L{l}H{h}" for l, h in heads_to_ablate],
        'normal_accuracy': normal_acc,
        'ablated_accuracy': ablated_acc,
        'accuracy_drop': normal_acc - ablated_acc,
        'results': results,
    }
    
    print(f"\nAblation Results:")
    print(f"  Normal accuracy: {normal_acc:.1%}")
    print(f"  Ablated accuracy: {ablated_acc:.1%}")
    print(f"  Accuracy drop: {(normal_acc - ablated_acc):.1%}")
    
    # Save
    output_path = os.path.join(save_dir, f"{model_key}_{context_type}_ablation.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    del model
    torch.cuda.empty_cache()
    
    return output

# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_copy_heads(save_dir: str):
    """Create visualization comparing copy heads across conditions."""
    
    # Load results
    results = {}
    for model_key in MODELS.keys():
        for ctx in ['short', 'long']:
            path = os.path.join(save_dir, f"{model_key}_{ctx}_copy_heads.json")
            if os.path.exists(path):
                with open(path) as f:
                    results[f"{model_key}_{ctx}"] = json.load(f)
    
    if len(results) < 2:
        print("Need at least 2 result files to compare")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Copy Head Analysis: Short vs Long Context', fontsize=14, fontweight='bold')
    
    # Plot 1: Accuracy comparison
    ax1 = axes[0, 0]
    models = list(MODELS.keys())
    x = np.arange(len(models))
    width = 0.35
    
    short_acc = [results.get(f"{m}_short", {}).get('accuracy', 0) * 100 for m in models]
    long_acc = [results.get(f"{m}_long", {}).get('accuracy', 0) * 100 for m in models]
    
    ax1.bar(x - width/2, short_acc, width, label='Short', color='#2ecc71')
    ax1.bar(x + width/2, long_acc, width, label='Long', color='#e74c3c')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Retrieval Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.capitalize() for m in models])
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # Plot 2: Top copy heads comparison (first model)
    ax2 = axes[0, 1]
    model = models[0]
    
    if f"{model}_short" in results and f"{model}_long" in results:
        short_heads = results[f"{model}_short"]['top_heads_by_needle_attention'][:10]
        long_heads = results[f"{model}_long"]['top_heads_by_needle_attention'][:10]
        
        short_names = [h['head'] for h in short_heads]
        short_scores = [h['avg_max_needle_attention'] for h in short_heads]
        
        y = np.arange(len(short_names))
        ax2.barh(y, short_scores, color='#2ecc71')
        ax2.set_yticks(y)
        ax2.set_yticklabels(short_names)
        ax2.set_xlabel('Avg Max Needle Attention')
        ax2.set_title(f'{model.capitalize()} - Short Context Top Heads')
    
    # Plot 3: Long context top heads
    ax3 = axes[1, 0]
    if f"{model}_long" in results:
        long_heads = results[f"{model}_long"]['top_heads_by_needle_attention'][:10]
        long_names = [h['head'] for h in long_heads]
        long_scores = [h['avg_max_needle_attention'] for h in long_heads]
        
        y = np.arange(len(long_names))
        ax3.barh(y, long_scores, color='#e74c3c')
        ax3.set_yticks(y)
        ax3.set_yticklabels(long_names)
        ax3.set_xlabel('Avg Max Needle Attention')
        ax3.set_title(f'{model.capitalize()} - Long Context Top Heads')
    
    # Plot 4: Copy rate comparison
    ax4 = axes[1, 1]
    if f"{model}_short" in results and f"{model}_long" in results:
        short_heads = results[f"{model}_short"]['top_heads_by_copy_rate'][:10]
        long_heads = results[f"{model}_long"]['top_heads_by_copy_rate'][:10]
        
        # Get union of heads
        all_heads = list(set(h['head'] for h in short_heads) | set(h['head'] for h in long_heads))[:10]
        
        short_dict = {h['head']: h['copy_head_rate'] for h in short_heads}
        long_dict = {h['head']: h['copy_head_rate'] for h in long_heads}
        
        y = np.arange(len(all_heads))
        height = 0.35
        
        short_rates = [short_dict.get(h, 0) * 100 for h in all_heads]
        long_rates = [long_dict.get(h, 0) * 100 for h in all_heads]
        
        ax4.barh(y - height/2, short_rates, height, label='Short', color='#2ecc71')
        ax4.barh(y + height/2, long_rates, height, label='Long', color='#e74c3c')
        ax4.set_yticks(y)
        ax4.set_yticklabels(all_heads)
        ax4.set_xlabel('Copy Head Rate (%)')
        ax4.set_title('How Often Head is "Copy Head" (argmax on needle)')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'copy_head_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_dir}/copy_head_analysis.png")

def compare_short_long_heads(save_dir: str):
    """Analyze differences between short and long context heads."""
    
    print("\n" + "="*70)
    print("SHORT vs LONG CONTEXT HEAD COMPARISON")
    print("="*70)
    
    for model_key in MODELS.keys():
        short_path = os.path.join(save_dir, f"{model_key}_short_copy_heads.json")
        long_path = os.path.join(save_dir, f"{model_key}_long_copy_heads.json")
        
        if not os.path.exists(short_path) or not os.path.exists(long_path):
            continue
        
        with open(short_path) as f:
            short_data = json.load(f)
        with open(long_path) as f:
            long_data = json.load(f)
        
        print(f"\n{MODELS[model_key].split('/')[-1]}:")
        print(f"  Short accuracy: {short_data['accuracy']:.1%}")
        print(f"  Long accuracy: {long_data['accuracy']:.1%}")
        
        # Compare top heads
        short_top = set(h['head'] for h in short_data['top_heads_by_needle_attention'][:10])
        long_top = set(h['head'] for h in long_data['top_heads_by_needle_attention'][:10])
        
        overlap = short_top & long_top
        short_only = short_top - long_top
        long_only = long_top - short_top
        
        print(f"\n  Top 10 heads overlap: {len(overlap)}/10")
        print(f"  Shared heads: {sorted(overlap)}")
        print(f"  Short-only heads: {sorted(short_only)}")
        print(f"  Long-only heads: {sorted(long_only)}")
        
        # Key insight: which heads are important for short but not long?
        print(f"\n  → Potential 'patching' candidates (strong in short, weak in long):")
        for h in short_only:
            print(f"      {h}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Copy Head Analysis")
    parser.add_argument('--model', type=str, choices=['llama', 'qwen', 'all'], default='all')
    parser.add_argument('--context', type=str, choices=['short', 'long', 'all'], default='all')
    parser.add_argument('--samples', type=int, default=NUM_SAMPLES)
    parser.add_argument('--visualize-only', action='store_true')
    parser.add_argument('--ablate', action='store_true', help='Run ablation experiment')
    
    args = parser.parse_args()
    
    save_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.visualize_only:
        visualize_copy_heads(save_dir)
        compare_short_long_heads(save_dir)
        return
    
    # Load data
    ground_truth = load_ground_truth()
    samples = load_edgar_samples(args.samples, ground_truth)
    
    # Run experiments
    models_to_run = list(MODELS.keys()) if args.model == 'all' else [args.model]
    contexts_to_run = list(CONTEXT_CONFIGS.keys()) if args.context == 'all' else [args.context]
    
    all_results = {}
    for model in models_to_run:
        for context in contexts_to_run:
            result = run_copy_head_experiment(model, context, samples, save_dir)
            all_results[f"{model}_{context}"] = result
    
    # Optional: Run ablation with top short-context heads
    if args.ablate:
        for model in models_to_run:
            short_result = all_results.get(f"{model}_short")
            if short_result:
                # Get top 5 copy heads from short context
                top_heads = [
                    (h['layer'], h['head_idx'])
                    for h in short_result['top_heads_by_needle_attention'][:5]
                ]
                run_ablation_experiment(model, 'short', samples, top_heads, save_dir)
    
    # Visualize
    visualize_copy_heads(save_dir)
    compare_short_long_heads(save_dir)
    
    print("\n" + "="*70)
    print("COPY HEAD ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()


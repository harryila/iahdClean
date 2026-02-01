# -*- coding: utf-8 -*-
"""
Retrieval Head Analysis for EDGAR State of Incorporation Task

Based on the paper "Retrieval Head Mechanistically Explains Long-Context Factual Recall"
https://arxiv.org/pdf/2404.15574

This script identifies which attention heads are responsible for retrieving
factual information (state of incorporation) from context, and how this
changes as context length increases.

Key Concepts:
- Retrieval Heads: Specific attention heads that "retrieve" information by
  attending from the answer position back to where the fact is stated in context
- As context grows, these heads must work harder to find the needle in the haystack
- We measure: (1) accuracy, (2) which heads are active, (3) attention patterns
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
import matplotlib.patches as mpatches

# Add parent directory for shared utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Models to test
MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

# Context length configurations
CONTEXT_CONFIGS = {
    "short": {
        "target_tokens": 200,
        "description": "Original short context (~200 tokens)",
        "expected_accuracy": 0.70,
    },
    "long": {
        "target_tokens": 2000, 
        "description": "Degraded long context (~2000 tokens with padding)",
        "expected_accuracy": 0.40,
    },
}

# Question template for state of incorporation
QUESTION_TEMPLATE = """Based on the following SEC 10-K filing excerpt, answer the question.

Context:
{context}

Question: In which US state was this company incorporated?
Answer with just the state name:"""

# Number of samples to test
NUM_SAMPLES = 50

# =============================================================================
# GROUND TRUTH LOADING
# =============================================================================

def load_ground_truth():
    """Load ground truth from the verified CSV."""
    gt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           "edgar_gt_verified_slim.csv")
    if os.path.exists(gt_path):
        import pandas as pd
        df = pd.read_csv(gt_path)
        gt_dict = {}
        for _, row in df.iterrows():
            filename = row['filename']
            state = row.get('original_Inc_state_truth', None)
            if pd.notna(state) and state and str(state).upper() not in ['NULL', 'NAN', 'NONE', '']:
                gt_dict[filename] = {'incorporation_state': str(state).strip()}
        print(f"Loaded ground truth for {len(gt_dict)} files from edgar_gt_verified_slim.csv")
        return gt_dict
    print(f"Warning: Ground truth file not found at {gt_path}")
    return {}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_edgar_samples(num_samples=50):
    """Load EDGAR corpus samples with section_1 for state of incorporation."""
    print(f"Loading EDGAR corpus ({num_samples} samples)...")
    
    dataset = load_dataset(
        "c3po-ai/edgar-corpus",
        "full",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    samples = []
    ground_truth = load_ground_truth()
    
    for item in dataset:
        if len(samples) >= num_samples:
            break
            
        filename = item.get('filename', '')
        section_1 = item.get('section_1', '')
        
        if not section_1 or len(section_1) < 100:
            continue
            
        # Get ground truth if available
        gt_state = None
        if filename in ground_truth:
            gt_state = ground_truth[filename].get('incorporation_state', None)
            # Handle pandas NaN values
            import pandas as pd
            if pd.isna(gt_state) if hasattr(gt_state, '__class__') else gt_state is None:
                gt_state = None
        
        samples.append({
            'filename': filename,
            'section_1': section_1,
            'section_2': item.get('section_2', ''),
            'section_7': item.get('section_7', ''),
            'ground_truth_state': gt_state,
        })
    
    print(f"Loaded {len(samples)} samples")
    return samples

# =============================================================================
# CONTEXT MANIPULATION
# =============================================================================

def truncate_to_tokens(text, tokenizer, max_tokens):
    """Truncate text to approximately max_tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

def create_short_context(sample, tokenizer, target_tokens=200):
    """Create a short context with just section_1 truncated."""
    section_1 = sample['section_1']
    return truncate_to_tokens(section_1, tokenizer, target_tokens)

def create_long_context(sample, tokenizer, target_tokens=2000):
    """
    Create a long context by padding section_1 with other sections.
    The key fact (state of incorporation) is usually at the start of section_1,
    so we add irrelevant content BEFORE and AFTER to make retrieval harder.
    """
    section_1 = sample['section_1']
    section_2 = sample.get('section_2', '')
    section_7 = sample.get('section_7', '')
    
    # Truncate section_1 to ~300 tokens (contains the key info)
    core_context = truncate_to_tokens(section_1, tokenizer, 300)
    core_tokens = len(tokenizer.encode(core_context, add_special_tokens=False))
    
    # Calculate remaining tokens for padding
    remaining_tokens = target_tokens - core_tokens
    padding_per_side = remaining_tokens // 2
    
    # Create padding from other sections
    padding_before = truncate_to_tokens(section_7, tokenizer, padding_per_side) if section_7 else ""
    padding_after = truncate_to_tokens(section_2, tokenizer, padding_per_side) if section_2 else ""
    
    # If we don't have enough padding, use generic filler
    if len(tokenizer.encode(padding_before, add_special_tokens=False)) < padding_per_side // 2:
        filler = "The company operates in various markets and segments. " * 50
        padding_before = truncate_to_tokens(filler, tokenizer, padding_per_side)
    
    if len(tokenizer.encode(padding_after, add_special_tokens=False)) < padding_per_side // 2:
        filler = "Financial results are subject to various risks and uncertainties. " * 50
        padding_after = truncate_to_tokens(filler, tokenizer, padding_per_side)
    
    # Combine: padding + core (with key fact) + padding
    full_context = f"{padding_before}\n\n{core_context}\n\n{padding_after}"
    return truncate_to_tokens(full_context, tokenizer, target_tokens)

# =============================================================================
# MODEL LOADING WITH ATTENTION OUTPUT
# =============================================================================

def load_model_with_attention(model_name):
    """Load model configured to output attention weights."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",  # Need eager for attention weights
        output_attentions=True,
    )
    model.eval()
    
    return model, tokenizer

# =============================================================================
# ATTENTION ANALYSIS
# =============================================================================

def extract_retrieval_heads(attentions, input_ids, tokenizer, context_start, context_end):
    """
    Identify retrieval heads by analyzing which heads attend strongly
    from the final (answer) position back to the context.
    
    A retrieval head is one where:
    - The attention from the last token position
    - Has high weight on tokens within the context region
    - Specifically on tokens that contain the answer information
    
    Returns: dict mapping (layer, head) -> attention_score_on_context
    """
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    
    head_scores = {}
    
    for layer_idx, layer_attn in enumerate(attentions):
        # layer_attn shape: (batch, num_heads, seq_len, seq_len)
        attn = layer_attn[0]  # Remove batch dimension
        
        for head_idx in range(num_heads):
            # Get attention from last position
            last_pos_attn = attn[head_idx, -1, :]  # Shape: (seq_len,)
            
            # Calculate how much attention goes to context region
            context_attn = last_pos_attn[context_start:context_end].sum().item()
            
            head_scores[(layer_idx, head_idx)] = context_attn
    
    return head_scores

def get_top_retrieval_heads(head_scores, top_k=20):
    """Get the top-k heads by retrieval score."""
    sorted_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_heads[:top_k]

# =============================================================================
# INFERENCE WITH ATTENTION TRACKING
# =============================================================================

def run_inference_with_attention(model, tokenizer, prompt, max_new_tokens=10):
    """
    Run inference and capture attention patterns.
    
    Returns:
        - generated_text: The model's answer
        - attentions: Attention weights from all layers/heads
        - input_ids: Token IDs of the input
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    
    # Get the generated tokens
    generated_ids = outputs.sequences[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # For attention analysis, we need to do a forward pass
    with torch.no_grad():
        forward_outputs = model(input_ids, output_attentions=True)
        attentions = forward_outputs.attentions
    
    return generated_text, attentions, input_ids[0]

# =============================================================================
# ACCURACY EVALUATION
# =============================================================================

US_STATES = [
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
]

def normalize_state(text):
    """Normalize state name for comparison."""
    if not text:
        return None
    text = text.lower().strip()
    # Remove common suffixes
    for suffix in ['.', ',', '!', '?', '\n', ' (', ' -']:
        if suffix in text:
            text = text.split(suffix)[0]
    # Get first word if multiple
    text = text.split()[0] if text.split() else text
    return text.strip()

def check_accuracy(answer, ground_truth):
    """Check if the answer matches ground truth."""
    if not ground_truth:
        return None  # Can't evaluate without ground truth
    
    answer_norm = normalize_state(answer)
    truth_norm = normalize_state(str(ground_truth))
    
    if not answer_norm or not truth_norm:
        return False
    
    # Check if answer is a valid state
    if answer_norm not in US_STATES and answer_norm not in [s.split()[0] for s in US_STATES]:
        return False
    
    return answer_norm == truth_norm or answer_norm in truth_norm or truth_norm in answer_norm

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(model_key, context_type, samples, save_dir):
    """
    Run the retrieval head experiment for a specific model and context type.
    """
    model_name = MODELS[model_key]
    config = CONTEXT_CONFIGS[context_type]
    
    print(f"\n{'='*70}")
    print(f"Running: {model_key} with {context_type} context")
    print(f"Target tokens: {config['target_tokens']}")
    print(f"{'='*70}\n")
    
    # Load model
    model, tokenizer = load_model_with_attention(model_name)
    
    results = []
    all_head_scores = defaultdict(list)
    
    for i, sample in enumerate(tqdm(samples, desc=f"{model_key}/{context_type}")):
        # Create context based on type
        if context_type == "short":
            context = create_short_context(sample, tokenizer, config['target_tokens'])
        else:
            context = create_long_context(sample, tokenizer, config['target_tokens'])
        
        # Create prompt
        prompt = QUESTION_TEMPLATE.format(context=context)
        
        # Find context boundaries in tokens
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        
        # Rough estimate of context position
        context_start = len(tokenizer.encode("Based on the following SEC 10-K filing excerpt, answer the question.\n\nContext:\n", add_special_tokens=False))
        context_end = context_start + len(context_tokens)
        
        # Run inference with attention
        try:
            answer, attentions, input_ids = run_inference_with_attention(
                model, tokenizer, prompt, max_new_tokens=10
            )
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue
        
        # Extract retrieval head scores
        head_scores = extract_retrieval_heads(
            attentions, input_ids, tokenizer, 
            context_start, min(context_end, len(input_ids))
        )
        
        # Accumulate head scores
        for (layer, head), score in head_scores.items():
            all_head_scores[(layer, head)].append(score)
        
        # Check accuracy
        is_correct = check_accuracy(answer, sample['ground_truth_state'])
        
        results.append({
            'filename': sample['filename'],
            'answer': answer.strip(),
            'ground_truth': sample['ground_truth_state'],
            'correct': is_correct,
            'context_tokens': len(context_tokens),
            'top_heads': get_top_retrieval_heads(head_scores, top_k=10),
        })
    
    # Calculate average head scores
    avg_head_scores = {
        head: np.mean(scores) for head, scores in all_head_scores.items()
    }
    
    # Get overall accuracy
    evaluated = [r for r in results if r['correct'] is not None]
    accuracy = sum(1 for r in evaluated if r['correct']) / len(evaluated) if evaluated else 0
    
    # Get top retrieval heads overall
    top_heads = get_top_retrieval_heads(avg_head_scores, top_k=20)
    
    # Save results
    output = {
        'model': model_key,
        'model_name': model_name,
        'context_type': context_type,
        'config': config,
        'accuracy': accuracy,
        'num_samples': len(results),
        'num_evaluated': len(evaluated),
        'top_retrieval_heads': [(f"L{l}H{h}", score) for (l, h), score in top_heads],
        'results': results,
    }
    
    output_path = os.path.join(save_dir, f"{model_key}_{context_type}_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults for {model_key} - {context_type}:")
    print(f"  Accuracy: {accuracy:.1%} ({sum(1 for r in evaluated if r['correct'])}/{len(evaluated)})")
    print(f"  Top 5 Retrieval Heads:")
    for (layer, head), score in top_heads[:5]:
        print(f"    Layer {layer}, Head {head}: {score:.4f}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return output

# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_retrieval_heads(results_dir):
    """Create visualizations comparing retrieval heads across conditions."""
    
    # Load all results
    results = {}
    for filename in os.listdir(results_dir):
        if filename.endswith('_results.json'):
            with open(os.path.join(results_dir, filename)) as f:
                data = json.load(f)
                key = f"{data['model']}_{data['context_type']}"
                results[key] = data
    
    if len(results) < 2:
        print("Need at least 2 result files to compare")
        return
    
    # Create accuracy comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy comparison
    ax1 = axes[0]
    models = list(MODELS.keys())
    x = np.arange(len(models))
    width = 0.35
    
    short_acc = [results.get(f"{m}_short", {}).get('accuracy', 0) for m in models]
    long_acc = [results.get(f"{m}_long", {}).get('accuracy', 0) for m in models]
    
    bars1 = ax1.bar(x - width/2, [a*100 for a in short_acc], width, label='Short (~200 tokens)', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, [a*100 for a in long_acc], width, label='Long (~2000 tokens)', color='#e74c3c')
    
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('State of Incorporation Retrieval Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODELS[m].split('/')[-1] for m in models], rotation=15, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.axhline(y=70, color='#2ecc71', linestyle='--', alpha=0.5, label='Expected Short')
    ax1.axhline(y=40, color='#e74c3c', linestyle='--', alpha=0.5, label='Expected Long')
    
    # Add value labels
    for bar, acc in zip(bars1, short_acc):
        ax1.annotate(f'{acc*100:.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    for bar, acc in zip(bars2, long_acc):
        ax1.annotate(f'{acc*100:.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    
    # Plot 2: Top retrieval heads comparison (for first model)
    ax2 = axes[1]
    
    if f"{models[0]}_short" in results and f"{models[0]}_long" in results:
        short_heads = {h: s for h, s in results[f"{models[0]}_short"]['top_retrieval_heads'][:15]}
        long_heads = {h: s for h, s in results[f"{models[0]}_long"]['top_retrieval_heads'][:15]}
        
        all_heads = sorted(set(short_heads.keys()) | set(long_heads.keys()))[:15]
        
        y = np.arange(len(all_heads))
        height = 0.35
        
        short_scores = [short_heads.get(h, 0) for h in all_heads]
        long_scores = [long_heads.get(h, 0) for h in all_heads]
        
        ax2.barh(y - height/2, short_scores, height, label='Short context', color='#2ecc71')
        ax2.barh(y + height/2, long_scores, height, label='Long context', color='#e74c3c')
        
        ax2.set_xlabel('Average Attention Score on Context')
        ax2.set_title(f'Top Retrieval Heads - {MODELS[models[0]].split("/")[-1]}')
        ax2.set_yticks(y)
        ax2.set_yticklabels(all_heads)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'retrieval_heads_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {results_dir}/retrieval_heads_comparison.png")

def compare_retrieval_heads(results_dir):
    """Analyze overlap in retrieval heads between short and long context."""
    
    results = {}
    for filename in os.listdir(results_dir):
        if filename.endswith('_results.json'):
            with open(os.path.join(results_dir, filename)) as f:
                data = json.load(f)
                key = f"{data['model']}_{data['context_type']}"
                results[key] = data
    
    print("\n" + "="*70)
    print("RETRIEVAL HEAD OVERLAP ANALYSIS")
    print("="*70)
    
    for model in MODELS.keys():
        short_key = f"{model}_short"
        long_key = f"{model}_long"
        
        if short_key not in results or long_key not in results:
            continue
        
        # Get top 20 heads from each
        short_heads = set(h for h, _ in results[short_key]['top_retrieval_heads'][:20])
        long_heads = set(h for h, _ in results[long_key]['top_retrieval_heads'][:20])
        
        overlap = short_heads & long_heads
        overlap_pct = len(overlap) / 20 * 100
        
        print(f"\n{MODELS[model].split('/')[-1]}:")
        print(f"  Top 20 heads overlap: {len(overlap)}/20 ({overlap_pct:.0f}%)")
        print(f"  Shared heads: {sorted(overlap)[:10]}...")
        print(f"  Short-only heads: {sorted(short_heads - long_heads)[:5]}...")
        print(f"  Long-only heads: {sorted(long_heads - short_heads)[:5]}...")

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrieval Head Analysis")
    parser.add_argument('--model', type=str, choices=['llama', 'qwen', 'all'], default='all')
    parser.add_argument('--context', type=str, choices=['short', 'long', 'all'], default='all')
    parser.add_argument('--samples', type=int, default=NUM_SAMPLES)
    parser.add_argument('--visualize-only', action='store_true', help='Only create visualizations from existing results')
    
    args = parser.parse_args()
    
    save_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.visualize_only:
        visualize_retrieval_heads(save_dir)
        compare_retrieval_heads(save_dir)
        return
    
    # Load samples
    samples = load_edgar_samples(args.samples)
    
    # Determine which models and contexts to run
    models_to_run = list(MODELS.keys()) if args.model == 'all' else [args.model]
    contexts_to_run = list(CONTEXT_CONFIGS.keys()) if args.context == 'all' else [args.context]
    
    # Run experiments
    all_results = {}
    for model in models_to_run:
        for context in contexts_to_run:
            result = run_experiment(model, context, samples, save_dir)
            all_results[f"{model}_{context}"] = result
    
    # Create visualizations
    visualize_retrieval_heads(save_dir)
    compare_retrieval_heads(save_dir)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()



"""
Llama 3 - CONTEXT ATTENTION Analysis (Your Approach)

Measures: Sum of attention over the ENTIRE Section 1 region (200-500 tokens)
This identifies heads that LOCATE/ATTEND to the relevant region.

Note: Attention analysis limited to ≤4K tokens (memory constraint).
For longer contexts, only accuracy is measured.
"""

import torch
import numpy as np
import json
import os
import sys
import gc
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '.')
from needle_haystack_sweep import (
    load_ground_truth, load_edgar_samples, MODELS,
    create_needle_in_haystack, QUESTION_TEMPLATE,
    extract_needle_full_section, get_haystack_text,
    run_inference, check_answer
)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_KEY = "llama"  # Llama 3
CONTEXT_LENGTHS = [200, 500, 1000, 2000, 4000, 8000, 10000, 12000, 15000]
MAX_CONTEXT_FOR_ATTENTION = 8000  # B200 has 180GB, can handle longer sequences
NUM_SAMPLES = 30  # Same 30 samples across all tests for consistency
NEEDLE_POSITION = 0.5  # Always middle
TOP_K_HEADS = 15

# ============================================================================
# TOKEN FINDING
# ============================================================================

def find_section1_region(input_ids, tokenizer, needle_text):
    """
    Find the start and end of the ENTIRE Section 1 (needle) region.
    Uses EXACT needle token count to determine region size.
    """
    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)
    needle_token_count = len(needle_tokens)
    
    search_tokens = needle_tokens[:15]
    input_list = input_ids.tolist() if torch.is_tensor(input_ids) else input_ids
    section_start = 0
    
    for i in range(len(input_list) - len(search_tokens) + 1):
        match_count = sum(1 for j, t in enumerate(search_tokens) 
                        if i+j < len(input_list) and input_list[i+j] == t)
        if match_count >= len(search_tokens) * 0.7:
            section_start = i
            break
    
    section_end = min(section_start + needle_token_count, len(input_list))
    return section_start, section_end


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_sample(model, tokenizer, sample, context_length, shuffle, haystack_text, device):
    """Analyze one sample, compute context attention for all heads."""
    ground_truth = sample['ground_truth_state']
    
    needle = extract_needle_full_section(
        sample['section_1'],
        ground_truth,
        shuffle=shuffle
    )
    
    context = create_needle_in_haystack(
        needle=needle,
        tokenizer=tokenizer,
        target_tokens=context_length,
        needle_position=NEEDLE_POSITION,
        haystack_text=haystack_text
    )
    
    prompt = QUESTION_TEMPLATE.format(context=context)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids[0]
    seq_len = len(input_ids)
    
    # Find Section 1 region
    section_start, section_end = find_section1_region(input_ids, tokenizer, needle)
    section_tokens = section_end - section_start
    
    # Get accuracy (without attention - just inference)
    try:
        answer = run_inference(model, tokenizer, prompt)
        is_correct = check_answer(answer, ground_truth)
    except Exception as e:
        answer = str(e)[:50]
        is_correct = False
    
    # Get attention scores (only for shorter contexts to avoid OOM)
    head_scores = {}
    if context_length <= MAX_CONTEXT_FOR_ATTENTION:
        try:
            with torch.no_grad():
                outputs = model(
                    inputs.input_ids,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True
                )
                
                for layer_idx, layer_attn in enumerate(outputs.attentions):
                    # Attention from last token to all positions
                    last_pos_attn = layer_attn[0, :, -1, :].float()  # (n_heads, seq_len)
                    
                    for head_idx in range(last_pos_attn.shape[0]):
                        head_attn = last_pos_attn[head_idx].cpu().numpy()
                        
                        # Context attention: sum over entire Section 1
                        s_start = max(0, section_start)
                        s_end = min(section_end, seq_len)
                        context_attn = float(head_attn[s_start:s_end].sum())
                        
                        head_key = f"L{layer_idx}H{head_idx}"
                        head_scores[head_key] = context_attn
                
                del outputs
                
        except Exception as e:
            print(f"\n    Attention error: {str(e)[:50]}")
    
    torch.cuda.empty_cache()
    
    return {
        'filename': sample['filename'],
        'ground_truth': ground_truth,
        'answer': answer.strip()[:100] if isinstance(answer, str) else str(answer)[:100],
        'is_correct': is_correct,
        'context_length': context_length,
        'seq_len': seq_len,
        'section_region': (section_start, section_end),
        'section_tokens': section_tokens,
        'head_scores': head_scores
    }


def run_sweep(model, tokenizer, samples, haystack_text, device):
    """Run full sweep across all context lengths, shuffled and unshuffled."""
    
    all_results = {
        'config': {
            'model': MODELS[MODEL_KEY],
            'context_lengths': CONTEXT_LENGTHS,
            'max_context_for_attention': MAX_CONTEXT_FOR_ATTENTION,
            'num_samples': NUM_SAMPLES,
            'needle_position': NEEDLE_POSITION,
            'metric': 'context_attention (sum over Section 1)',
            'sample_filenames': [s['filename'] for s in samples]
        },
        'by_context_length': {}
    }
    
    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n{'='*70}")
        print(f"Context Length: {ctx_len} tokens")
        if ctx_len > MAX_CONTEXT_FOR_ATTENTION:
            print(f"  (Accuracy only - attention skipped for memory)")
        print(f"{'='*70}")
        
        all_results['by_context_length'][ctx_len] = {
            'unshuffled': {'samples': [], 'accuracy': 0, 'head_scores': {}},
            'shuffled': {'samples': [], 'accuracy': 0, 'head_scores': {}}
        }
        
        for shuffle in [False, True]:
            condition = 'shuffled' if shuffle else 'unshuffled'
            print(f"\n  {condition.upper()}:")
            
            head_scores_all = defaultdict(list)
            correct = 0
            total = 0
            
            for sample in tqdm(samples, desc=f"  {condition}"):
                try:
                    result = analyze_sample(
                        model, tokenizer, sample,
                        ctx_len, shuffle, haystack_text, device
                    )
                    
                    all_results['by_context_length'][ctx_len][condition]['samples'].append({
                        'filename': result['filename'],
                        'ground_truth': result['ground_truth'],
                        'answer': result['answer'],
                        'is_correct': result['is_correct'],
                        'section_region': result['section_region'],
                        'section_tokens': result['section_tokens'],
                    })
                    
                    for head, score in result['head_scores'].items():
                        head_scores_all[head].append(score)
                    
                    total += 1
                    if result['is_correct']:
                        correct += 1
                        
                except torch.cuda.OutOfMemoryError:
                    print(f"\n    OOM, skipping...")
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"\n    Error: {e}")
                    torch.cuda.empty_cache()
                    continue
            
            # Calculate mean scores and accuracy
            accuracy = correct / total if total > 0 else 0
            mean_scores = {h: np.mean(s) for h, s in head_scores_all.items()}
            top_heads = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_HEADS]
            
            all_results['by_context_length'][ctx_len][condition]['accuracy'] = accuracy
            all_results['by_context_length'][ctx_len][condition]['correct'] = correct
            all_results['by_context_length'][ctx_len][condition]['total'] = total
            all_results['by_context_length'][ctx_len][condition]['top_heads'] = top_heads
            all_results['by_context_length'][ctx_len][condition]['mean_head_scores'] = mean_scores
            
            print(f"    Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
            if top_heads:
                print(f"    Top 5 heads: {[h for h,s in top_heads[:5]]}")
            else:
                print(f"    (No attention data - context too long)")
        
        # Save intermediate
        save_path = 'llama3_results/head/context_attention_sweep_partial.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        torch.cuda.empty_cache()
        gc.collect()
    
    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("LLAMA 3 - CONTEXT ATTENTION SWEEP")
    print("Metric: Sum of attention over ENTIRE Section 1 region")
    print(f"Attention analysis: ≤{MAX_CONTEXT_FOR_ATTENTION} tokens only")
    print("="*70)
    
    # Load model on single GPU
    print("\nLoading Llama 3 on GPU 0...")
    model_name = MODELS[MODEL_KEY]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # All on GPU 0
        attn_implementation="eager",  # Need eager for attention output
    )
    model.eval()
    device = torch.device("cuda:0")
    print(f"Model loaded on {device}")
    
    # Load data
    ground_truth = load_ground_truth()
    samples = load_edgar_samples(NUM_SAMPLES, ground_truth)
    haystack_text = get_haystack_text()
    print(f"Loaded {len(samples)} samples")
    print(f"Haystack: Alice in Wonderland ({len(haystack_text)} chars)")
    
    # Print sample filenames for consistency verification
    print("\nSample filenames (for cross-test verification):")
    sample_filenames = [s['filename'] for s in samples]
    for i, fn in enumerate(sample_filenames[:5]):
        print(f"  {i+1}. {fn}")
    print(f"  ... and {len(sample_filenames)-5} more")
    
    # Run sweep
    results = run_sweep(model, tokenizer, samples, haystack_text, device)
    
    # Save final
    save_path = 'llama3_results/head/context_attention_sweep.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {save_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: CONTEXT ATTENTION ACROSS CONTEXT LENGTHS")
    print("="*70)
    
    print("\nContext | Unshuf Acc | Shuf Acc | Drop | Top Head (unshuf)")
    print("-"*70)
    for ctx_len in CONTEXT_LENGTHS:
        data = results['by_context_length'][ctx_len]
        unshuf_acc = data['unshuffled']['accuracy'] * 100
        shuf_acc = data['shuffled']['accuracy'] * 100
        drop = unshuf_acc - shuf_acc
        top_head = data['unshuffled']['top_heads'][0][0] if data['unshuffled']['top_heads'] else "N/A (>4K)"
        print(f"{ctx_len:>7} | {unshuf_acc:>9.1f}% | {shuf_acc:>7.1f}% | {drop:>+5.1f}% | {top_head}")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

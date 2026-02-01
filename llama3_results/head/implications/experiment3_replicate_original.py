"""
EXPERIMENT 3: Replicate Original Finding (500-token window)

Original finding to replicate:
- UNSHUFFLED: Later layers (L16-24) do retrieval → comprehension heads
- SHUFFLED: Early layers (L2-5) do retrieval → keyword matching heads

This uses the SAME method as the original llama3_retrieval_head_analysis.py:
- 500-token window for context attention (not full Section 1)
- Same samples for both conditions

This should replicate the layer distribution difference observed before.
"""

import torch
import numpy as np
import json
import os
import sys
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/ubuntu/iahd/retrieval_heads')
from needle_haystack_sweep import (
    load_ground_truth, load_edgar_samples, MODELS,
    create_needle_in_haystack, QUESTION_TEMPLATE,
    extract_needle_full_section, get_haystack_text
)

# Configuration - SAME as original
MODEL_KEY = "llama"
CONTEXT_LENGTH = 2000  # Same as original "short" condition
NUM_SAMPLES = 20  # Same as original
NEEDLE_POSITION = 0.5
CONTEXT_WINDOW = 500  # KEY DIFFERENCE: 500-token window, not full Section 1


def get_attention_500_window(
    model,
    tokenizer,
    prompt: str,
    context_start_text: str,
    device: str = "cuda"
) -> np.ndarray:
    """
    Compute attention from last token to a 500-token context window.
    This is the ORIGINAL method that showed L16-24 vs L2-5 difference.
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs.input_ids[0]
    
    # Find context region - match first 50 tokens
    context_ids = tokenizer.encode(context_start_text[:500], add_special_tokens=False)
    context_tensor = torch.tensor(context_ids[:50], device=device)
    
    start_idx = 0
    for i in range(len(input_ids) - len(context_tensor) + 1):
        if torch.equal(input_ids[i:i+len(context_tensor)], context_tensor):
            start_idx = i
            break
    
    # 500-token window (KEY: not full Section 1)
    end_idx = min(start_idx + CONTEXT_WINDOW, len(input_ids) - 50)
    
    # Run model
    with torch.no_grad():
        outputs = model(
            inputs.input_ids,
            output_attentions=True,
            use_cache=False
        )
    
    # Extract attention scores
    n_layers = len(outputs.attentions)
    n_heads = outputs.attentions[0].shape[1]
    attention_scores = np.zeros((n_layers, n_heads))
    
    for layer_idx, layer_attn in enumerate(outputs.attentions):
        last_token_attn = layer_attn[0, :, -1, :].float().cpu().numpy()
        context_attn = last_token_attn[:, start_idx:end_idx].sum(axis=1)
        attention_scores[layer_idx] = context_attn
    
    return attention_scores, (start_idx, end_idx)


def identify_top_heads(attention_scores: np.ndarray, top_k: int = 15) -> list:
    """Get top-k heads sorted by attention score"""
    n_layers, n_heads = attention_scores.shape
    head_scores = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            head_scores.append((layer, head, attention_scores[layer, head]))
    
    head_scores.sort(key=lambda x: x[2], reverse=True)
    return head_scores[:top_k]


def main():
    print("=" * 70)
    print("EXPERIMENT 3: REPLICATE ORIGINAL FINDING")
    print("500-token window analysis (same as original)")
    print("=" * 70)
    print()
    print("Expected results:")
    print("  UNSHUFFLED: Later layers (L16-24)")
    print("  SHUFFLED: Early layers (L2-5)")
    print()
    
    # Load model
    print("Loading model...")
    model_name = MODELS[MODEL_KEY]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    
    # Load data
    ground_truth = load_ground_truth()
    samples = load_edgar_samples(NUM_SAMPLES, ground_truth)
    haystack_text = get_haystack_text()
    print(f"Loaded {len(samples)} samples")
    
    results = {
        "config": {
            "context_window": CONTEXT_WINDOW,
            "context_length": CONTEXT_LENGTH,
            "num_samples": NUM_SAMPLES,
            "method": "500-token window (original method)"
        },
        "unshuffled": {},
        "shuffled": {}
    }
    
    for shuffle in [False, True]:
        condition = "shuffled" if shuffle else "unshuffled"
        print(f"\n{'=' * 70}")
        print(f"Running {condition.upper()} condition...")
        print("=" * 70)
        
        all_attention_scores = []
        
        for sample in tqdm(samples, desc=condition):
            needle = extract_needle_full_section(
                sample['section_1'],
                sample['ground_truth_state'],
                shuffle=shuffle
            )
            
            context = create_needle_in_haystack(
                needle=needle,
                tokenizer=tokenizer,
                target_tokens=CONTEXT_LENGTH,
                needle_position=NEEDLE_POSITION,
                haystack_text=haystack_text
            )
            
            prompt = QUESTION_TEMPLATE.format(context=context)
            
            try:
                attn_scores, _ = get_attention_500_window(
                    model, tokenizer, prompt,
                    sample['section_1'][:500],
                    device
                )
                all_attention_scores.append(attn_scores)
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            # Clear memory
            del prompt
            torch.cuda.empty_cache()
            gc.collect()
        
        if all_attention_scores:
            mean_scores = np.mean(all_attention_scores, axis=0)
            top_heads = identify_top_heads(mean_scores)
            
            results[condition] = {
                "n_samples": len(all_attention_scores),
                "mean_attention": mean_scores.tolist(),
                "top_heads": [(int(l), int(h), float(s)) for l, h, s in top_heads]
            }
            
            # Print top heads
            print(f"\nTop 10 heads ({condition}):")
            layers = []
            for i, (layer, head, score) in enumerate(top_heads[:10]):
                print(f"  {i+1}. L{layer}H{head}: {score:.4f}")
                layers.append(layer)
            
            avg_layer = np.mean(layers)
            print(f"\n  Average layer: {avg_layer:.1f}")
            results[condition]["avg_layer_top10"] = avg_layer
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON: UNSHUFFLED vs SHUFFLED")
    print("=" * 70)
    
    unshuf_layers = [h[0] for h in results["unshuffled"]["top_heads"][:10]]
    shuf_layers = [h[0] for h in results["shuffled"]["top_heads"][:10]]
    
    print(f"\nUNSHUFFLED top 10 layers: {unshuf_layers}")
    print(f"  Average: {np.mean(unshuf_layers):.1f}")
    
    print(f"\nSHUFFLED top 10 layers: {shuf_layers}")
    print(f"  Average: {np.mean(shuf_layers):.1f}")
    
    print(f"\nDifference: {np.mean(unshuf_layers) - np.mean(shuf_layers):.1f} layers")
    
    # Check if original finding replicated
    unshuf_avg = np.mean(unshuf_layers)
    shuf_avg = np.mean(shuf_layers)
    
    print("\n" + "=" * 70)
    print("REPLICATION CHECK")
    print("=" * 70)
    
    if unshuf_avg > 15 and shuf_avg < 10:
        print("✓ ORIGINAL FINDING REPLICATED!")
        print(f"  Unshuffled uses later layers (avg: {unshuf_avg:.1f})")
        print(f"  Shuffled uses earlier layers (avg: {shuf_avg:.1f})")
    elif unshuf_avg > shuf_avg:
        print("~ PARTIAL REPLICATION")
        print(f"  Unshuffled uses later layers than shuffled")
        print(f"  But difference less dramatic than original")
    else:
        print("✗ ORIGINAL FINDING NOT REPLICATED")
        print(f"  Unshuffled avg: {unshuf_avg:.1f}")
        print(f"  Shuffled avg: {shuf_avg:.1f}")
    
    results["summary"] = {
        "unshuffled_avg_layer": float(unshuf_avg),
        "shuffled_avg_layer": float(shuf_avg),
        "layer_difference": float(unshuf_avg - shuf_avg),
        "replicated": bool(unshuf_avg > 15 and shuf_avg < 10)
    }
    
    # Save results
    with open("experiment3_replication_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nSaved: experiment3_replication_results.json")
    
    # Create visualization
    create_visualization(results)


def create_visualization(results):
    """Create comparison visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Layer distribution comparison
    ax1 = axes[0]
    unshuf_layers = [h[0] for h in results["unshuffled"]["top_heads"][:10]]
    shuf_layers = [h[0] for h in results["shuffled"]["top_heads"][:10]]
    
    x = np.arange(10)
    width = 0.35
    
    ax1.bar(x - width/2, unshuf_layers, width, label='Unshuffled', color='#2563eb')
    ax1.bar(x + width/2, shuf_layers, width, label='Shuffled', color='#dc2626')
    
    ax1.set_xlabel('Rank', fontsize=12)
    ax1.set_ylabel('Layer Index', fontsize=12)
    ax1.set_title('Layer of Top 10 Heads\n(500-token window)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i+1) for i in range(10)])
    ax1.axhline(y=16, color='gray', linestyle='--', alpha=0.5)
    ax1.text(9, 17, 'Late layers', fontsize=9, color='gray')
    ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
    ax1.text(9, 6, 'Early layers', fontsize=9, color='gray')
    
    # Plot 2: Average layer comparison
    ax2 = axes[1]
    conditions = ['Unshuffled', 'Shuffled']
    avg_layers = [np.mean(unshuf_layers), np.mean(shuf_layers)]
    colors = ['#2563eb', '#dc2626']
    
    bars = ax2.bar(conditions, avg_layers, color=colors, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Average Layer', fontsize=12)
    ax2.set_title('Average Layer of Top 10 Heads\n(500-token window)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 25)
    
    for bar, val in zip(bars, avg_layers):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=14, fontweight='bold')
    
    ax2.axhline(y=16, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('experiment3_replication_visual.png', dpi=150, facecolor='white')
    print("Saved: experiment3_replication_visual.png")
    plt.close()


if __name__ == "__main__":
    main()


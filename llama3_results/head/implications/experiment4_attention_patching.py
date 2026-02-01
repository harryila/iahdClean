"""
EXPERIMENT 4: Attention Pattern Patching (v2 - Fixed)

Goal: Restore accuracy in degraded conditions (shuffled) by copying
attention statistics from good conditions (unshuffled).

Issue with v1: Sequence lengths differ between shuffled/unshuffled,
so we can't directly copy attention matrices.

New approach: 
- Measure which POSITIONS the model attends to in good condition
- Apply a bias to encourage similar attention in bad condition
- Focus on attention to the NEEDLE REGION specifically

This is "soft patching" - encouraging similar behavior rather than forcing it.
"""

import torch
import numpy as np
import json
import os
import sys
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/home/ubuntu/iahd/retrieval_heads')
from needle_haystack_sweep import (
    load_ground_truth, load_edgar_samples, MODELS,
    create_needle_in_haystack, QUESTION_TEMPLATE,
    extract_needle_full_section, get_haystack_text,
    check_answer
)

# Configuration
MODEL_KEY = "llama"
CONTEXT_LENGTH = 2000
NUM_SAMPLES = 30
NEEDLE_POSITION = 0.5

# Test different patching strategies
PATCH_CONFIGS = {
    "baseline_unshuffled": {"shuffle": False, "patch": None},
    "baseline_shuffled": {"shuffle": True, "patch": None},
    "patch_early_L0-2": {"shuffle": True, "patch": {"layers": [0, 1, 2], "boost": 3.0}},
    "patch_critical_L2": {"shuffle": True, "patch": {"layers": [2], "boost": 5.0}},
    "patch_copy_L17-20": {"shuffle": True, "patch": {"layers": [17, 18, 19, 20], "boost": 3.0}},
}


def find_needle_region(tokenizer, prompt):
    """Find token positions of needle region in prompt"""
    # Look for Section 1 marker
    if "Section 1" in prompt:
        before = prompt.split("Section 1")[0]
        tokens_before = len(tokenizer.encode(before, add_special_tokens=False))
        needle_start = tokens_before
        needle_end = min(needle_start + 500, len(tokenizer.encode(prompt, add_special_tokens=False)) - 100)
    else:
        total_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        needle_start = total_len // 4
        needle_end = total_len // 2
    
    return needle_start, needle_end


def run_with_attention_bias(model, tokenizer, prompt, patch_config, device="cuda"):
    """
    Run inference with optional attention bias to needle region.
    
    Instead of copying exact patterns, we add a bias to attention scores
    to encourage attending to the needle region in specified layers.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]
    
    if patch_config is None:
        # No patching - standard inference
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Get needle region
    needle_start, needle_end = find_needle_region(tokenizer, prompt)
    layers_to_patch = patch_config["layers"]
    boost = patch_config["boost"]
    
    # Create attention bias
    # Shape: [1, 1, 1, seq_len] - broadcasts to all heads
    attn_bias = torch.zeros(1, 1, 1, seq_len, device=device, dtype=torch.bfloat16)
    actual_end = min(needle_end, seq_len)
    actual_start = min(needle_start, actual_end)
    attn_bias[0, 0, 0, actual_start:actual_end] = boost
    
    original_forwards = {}
    
    def create_biased_forward(layer_idx, original_forward, bias):
        def biased_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs
        ):
            # Add bias to attention mask
            if attention_mask is not None:
                # Expand bias to match current sequence length
                curr_len = attention_mask.shape[-1]
                if curr_len <= bias.shape[-1]:
                    curr_bias = bias[:, :, :, :curr_len]
                else:
                    curr_bias = torch.zeros(1, 1, 1, curr_len, device=bias.device, dtype=bias.dtype)
                    curr_bias[:, :, :, :bias.shape[-1]] = bias
                
                attention_mask = attention_mask + curr_bias.expand(-1, -1, attention_mask.shape[2], -1)
            
            return original_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs
            )
        return biased_forward
    
    # Patch layers
    for layer_idx in layers_to_patch:
        if layer_idx < len(model.model.layers):
            attn = model.model.layers[layer_idx].self_attn
            original_forwards[layer_idx] = attn.forward
            attn.forward = create_biased_forward(layer_idx, attn.forward, attn_bias)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    finally:
        for layer_idx, original in original_forwards.items():
            model.model.layers[layer_idx].self_attn.forward = original
    
    return response


def main():
    print("=" * 70)
    print("EXPERIMENT 4: ATTENTION BIAS PATCHING (v2)")
    print("=" * 70)
    print("Testing if biasing attention toward needle region helps shuffled accuracy")
    print()
    
    # Load model
    print("Loading model...")
    model_name = MODELS[MODEL_KEY]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()
    device = next(model.parameters()).device
    
    # Load data
    ground_truth = load_ground_truth()
    samples = load_edgar_samples(NUM_SAMPLES, ground_truth)
    haystack_text = get_haystack_text()
    print(f"Loaded {len(samples)} samples")
    
    # Prepare prompts
    prompts_data = []
    for sample in samples:
        # Unshuffled needle
        needle_unshuf = extract_needle_full_section(
            sample['section_1'], sample['ground_truth_state'], shuffle=False
        )
        context_unshuf = create_needle_in_haystack(
            needle=needle_unshuf, tokenizer=tokenizer,
            target_tokens=CONTEXT_LENGTH, needle_position=NEEDLE_POSITION,
            haystack_text=haystack_text
        )
        
        # Shuffled needle
        needle_shuf = extract_needle_full_section(
            sample['section_1'], sample['ground_truth_state'], shuffle=True
        )
        context_shuf = create_needle_in_haystack(
            needle=needle_shuf, tokenizer=tokenizer,
            target_tokens=CONTEXT_LENGTH, needle_position=NEEDLE_POSITION,
            haystack_text=haystack_text
        )
        
        prompts_data.append({
            "filename": sample['filename'],
            "gt": sample['ground_truth_state'],
            "prompt_unshuf": QUESTION_TEMPLATE.format(context=context_unshuf),
            "prompt_shuf": QUESTION_TEMPLATE.format(context=context_shuf),
        })
    
    results = {"config": PATCH_CONFIGS, "conditions": {}}
    
    # Test each configuration
    for config_name, config in PATCH_CONFIGS.items():
        print(f"\n{'=' * 70}")
        print(f"Testing: {config_name}")
        print(f"Shuffle: {config['shuffle']}, Patch: {config['patch']}")
        print("=" * 70)
        
        correct = 0
        total = 0
        sample_results = []
        
        for data in tqdm(prompts_data, desc=config_name):
            prompt = data["prompt_shuf"] if config["shuffle"] else data["prompt_unshuf"]
            
            try:
                answer = run_with_attention_bias(model, tokenizer, prompt, config["patch"], device)
                is_correct = check_answer(answer, data["gt"])
            except Exception as e:
                print(f"\nError: {e}")
                answer = "ERROR"
                is_correct = False
            
            if is_correct:
                correct += 1
            total += 1
            
            sample_results.append({
                "filename": data["filename"],
                "gt": data["gt"],
                "answer": answer,
                "correct": is_correct
            })
            
            torch.cuda.empty_cache()
            gc.collect()
        
        accuracy = correct / total * 100
        results["conditions"][config_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "samples": sample_results
        }
        print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    # Summary
    unshuf_acc = results["conditions"]["baseline_unshuffled"]["accuracy"]
    shuf_acc = results["conditions"]["baseline_shuffled"]["accuracy"]
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<25} | {'Accuracy':>10} | {'vs Shuffled':>12}")
    print("-" * 55)
    
    for config_name, data in results["conditions"].items():
        change = data["accuracy"] - shuf_acc
        print(f"{config_name:<25} | {data['accuracy']:>9.1f}% | {change:>+11.1f}%")
    
    # Find best patching config
    best_patch = None
    best_improvement = 0
    for config_name, data in results["conditions"].items():
        if "patch" in config_name:
            improvement = data["accuracy"] - shuf_acc
            if improvement > best_improvement:
                best_improvement = improvement
                best_patch = config_name
    
    results["summary"] = {
        "unshuffled_baseline": unshuf_acc,
        "shuffled_baseline": shuf_acc,
        "degradation": unshuf_acc - shuf_acc,
        "best_patch": best_patch,
        "best_improvement": best_improvement
    }
    
    # Save
    with open("experiment4_patching_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print()
    if best_improvement > 5:
        recovery = best_improvement / (unshuf_acc - shuf_acc) * 100 if unshuf_acc > shuf_acc else 0
        print(f"✓ Best patch: {best_patch} (+{best_improvement:.1f}%)")
        print(f"  Recovery: {recovery:.1f}% of lost accuracy")
    elif best_improvement > 0:
        print(f"~ Small improvement: {best_patch} (+{best_improvement:.1f}%)")
    else:
        print("✗ No patching configuration improved accuracy")
    
    print("\nSaved: experiment4_patching_results.json")


if __name__ == "__main__":
    main()

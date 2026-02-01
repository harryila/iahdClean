"""
EXPERIMENT 1: Exact Attention Boost at 8K Context

Goal: Test if boosting specific heads by their EXACT attention scores
from 8K unshuffled can help 8K shuffled achieve similar accuracy.

Data from contextAlg 8K analysis:
- Unshuffled accuracy: 66.7%
- Shuffled accuracy: 33.3%

Top heads from 8K UNSHUFFLED (these are the exact values we'll use):
- L17H24: 0.7144
- L16H1:  0.7132
- L18H16: 0.6433
- L17H25: 0.5909
- L20H1:  0.5871

Method: Boost head outputs BEFORE o_proj by the exact attention score value.
This is done correctly by patching the forward method.
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

# Configuration - FIXED at 8K context
MODEL_KEY = "llama"
CONTEXT_LENGTH = 8000  # Fixed at 8K
NUM_SAMPLES = 30
NEEDLE_POSITION = 0.5

# EXACT attention scores from 8K UNSHUFFLED top heads
# Format: (layer, head, exact_attention_score)
HEADS_FROM_8K_UNSHUFFLED = [
    (17, 24, 0.7144),
    (16, 1,  0.7132),
    (18, 16, 0.6433),
    (17, 25, 0.5909),
    (20, 1,  0.5871),
    (18, 20, 0.5843),
    (20, 14, 0.5831),
    (17, 26, 0.5765),
    (27, 6,  0.5470),
    (18, 22, 0.5390),
]

# Test configurations
BOOST_CONFIGS = {
    "baseline_unshuffled_8k": {"condition": "unshuffled", "heads": []},
    "baseline_shuffled_8k": {"condition": "shuffled", "heads": []},
    # Boost by exact attention values
    "top5_exact_boost": {
        "condition": "shuffled",
        "heads": HEADS_FROM_8K_UNSHUFFLED[:5]
    },
    "top10_exact_boost": {
        "condition": "shuffled",
        "heads": HEADS_FROM_8K_UNSHUFFLED[:10]
    },
    # Boost by 2x the exact attention values
    "top5_2x_exact": {
        "condition": "shuffled",
        "heads": [(l, h, s*2) for l, h, s in HEADS_FROM_8K_UNSHUFFLED[:5]]
    },
}


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class ExactAttentionBooster:
    """
    Boost specific heads by their exact attention score values BEFORE o_proj.
    """
    
    def __init__(self, model, heads_with_scores):
        """
        heads_with_scores: list of (layer, head, exact_score) tuples
        """
        self.model = model
        self.original_forwards = {}
        
        # Group by layer: {layer: [(head, score), ...]}
        self.layers_to_heads = {}
        for layer, head, score in heads_with_scores:
            if layer not in self.layers_to_heads:
                self.layers_to_heads[layer] = []
            self.layers_to_heads[layer].append((head, score))
    
    def _create_patched_forward(self, layer_idx, heads_with_scores):
        """Create a patched forward that boosts specific heads by exact scores BEFORE o_proj"""
        model = self.model
        
        def patched_forward(
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
            attn = model.model.layers[layer_idx].self_attn
            bsz, q_len, _ = hidden_states.size()
            
            # Project to Q, K, V
            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)
            value_states = attn.v_proj(hidden_states)
            
            # Reshape
            query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            
            # Rotary embeddings
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = attn.rotary_emb(value_states, position_ids)
            
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
            # Repeat KV for GQA
            n_rep = attn.num_heads // attn.num_key_value_heads
            if n_rep > 1:
                key_states = key_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, attn.num_heads, -1, attn.head_dim)
                value_states = value_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bsz, attn.num_heads, -1, attn.head_dim)
            
            # Compute attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (attn.head_dim ** 0.5)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            attn_output = torch.matmul(attn_weights, value_states)  # [bsz, heads, seq, head_dim]
            
            # *** BOOST SPECIFIC HEADS BY EXACT ATTENTION SCORE BEFORE o_proj ***
            for head_idx, exact_score in heads_with_scores:
                # Use the exact attention score as a multiplicative boost factor
                # (adding 1 so score of 0.7 means 1.7x boost)
                boost_factor = 1.0 + exact_score
                attn_output[:, head_idx, :, :] = attn_output[:, head_idx, :, :] * boost_factor
            
            # Reshape and project
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            attn_output = attn.o_proj(attn_output)
            
            if not output_attentions:
                attn_weights = None
            
            return attn_output, attn_weights, past_key_value
        
        return patched_forward
    
    def enable(self):
        """Patch forward methods for layers with heads to boost"""
        for layer_idx, heads in self.layers_to_heads.items():
            attn = self.model.model.layers[layer_idx].self_attn
            self.original_forwards[layer_idx] = attn.forward
            attn.forward = self._create_patched_forward(layer_idx, heads)
    
    def disable(self):
        """Restore original forward methods"""
        for layer_idx, original in self.original_forwards.items():
            self.model.model.layers[layer_idx].self_attn.forward = original
        self.original_forwards = {}


def run_inference(model, tokenizer, prompt, heads_with_scores=None, device="cuda"):
    """Run inference with optional exact attention boosting"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    booster = None
    if heads_with_scores:
        booster = ExactAttentionBooster(model, heads_with_scores)
        booster.enable()
    
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
        if booster:
            booster.disable()
    
    return response


def main():
    print("=" * 70)
    print("EXPERIMENT 1: EXACT ATTENTION BOOST AT 8K CONTEXT")
    print("=" * 70)
    print(f"Context length: {CONTEXT_LENGTH} tokens")
    print(f"Samples: {NUM_SAMPLES}")
    print()
    print("Top 5 heads from 8K UNSHUFFLED (exact values):")
    for layer, head, score in HEADS_FROM_8K_UNSHUFFLED[:5]:
        print(f"  L{layer}H{head}: {score:.4f}")
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
    
    # Prepare prompts for 8K context
    prompts = {"shuffled": [], "unshuffled": []}
    
    for sample in samples:
        for condition in ["shuffled", "unshuffled"]:
            needle = extract_needle_full_section(
                sample['section_1'], sample['ground_truth_state'], 
                shuffle=(condition == "shuffled")
            )
            context = create_needle_in_haystack(
                needle=needle, tokenizer=tokenizer,
                target_tokens=CONTEXT_LENGTH, needle_position=NEEDLE_POSITION,
                haystack_text=haystack_text
            )
            prompts[condition].append({
                "prompt": QUESTION_TEMPLATE.format(context=context),
                "gt": sample['ground_truth_state'],
                "filename": sample['filename']
            })
    
    results = {
        "context_length": CONTEXT_LENGTH,
        "num_samples": NUM_SAMPLES,
        "heads_used": HEADS_FROM_8K_UNSHUFFLED[:10],
        "conditions": {}
    }
    
    # Run each configuration
    for config_name, config in BOOST_CONFIGS.items():
        print(f"\n{'=' * 70}")
        print(f"Testing: {config_name}")
        
        condition = config["condition"]
        heads = config.get("heads", [])
        
        if heads:
            print(f"Boosting {len(heads)} heads by exact scores:")
            for layer, head, score in heads[:5]:
                print(f"  L{layer}H{head}: boost = 1 + {score:.4f} = {1+score:.4f}x")
        else:
            print(f"No boost (baseline {condition})")
        print("=" * 70)
        
        correct = 0
        sample_results = []
        
        prompt_list = prompts[condition]
        
        for data in tqdm(prompt_list, desc=config_name):
            try:
                answer = run_inference(
                    model, tokenizer, data["prompt"],
                    heads_with_scores=heads if heads else None,
                    device=device
                )
                is_correct = check_answer(answer, data["gt"])
            except Exception as e:
                print(f"\nError: {e}")
                answer = "ERROR"
                is_correct = False
            
            if is_correct:
                correct += 1
            sample_results.append({
                "gt": data["gt"], 
                "answer": answer, 
                "correct": bool(is_correct)
            })
            
            torch.cuda.empty_cache()
            gc.collect()
        
        accuracy = correct / len(prompt_list) * 100
        results["conditions"][config_name] = {
            "condition": condition,
            "heads_boosted": len(heads),
            "accuracy": accuracy,
            "correct": correct,
            "total": len(prompt_list),
            "samples": sample_results
        }
        print(f"Accuracy: {accuracy:.1f}% ({correct}/{len(prompt_list)})")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - 8K CONTEXT")
    print("=" * 70)
    
    baseline_shuf = results["conditions"]["baseline_shuffled_8k"]["accuracy"]
    baseline_unshuf = results["conditions"]["baseline_unshuffled_8k"]["accuracy"]
    
    print(f"{'Configuration':<25} | {'Accuracy':>10} | {'vs Shuffled':>12} | {'vs Target':>12}")
    print("-" * 70)
    
    for config_name, data in results["conditions"].items():
        change_vs_shuf = data["accuracy"] - baseline_shuf
        change_vs_unshuf = data["accuracy"] - baseline_unshuf
        
        if config_name == "baseline_unshuffled_8k":
            label = " (TARGET)"
        elif config_name == "baseline_shuffled_8k":
            label = " (START)"
        else:
            label = ""
        
        print(f"{config_name:<25} | {data['accuracy']:>9.1f}% | {change_vs_shuf:>+11.1f}% | {change_vs_unshuf:>+11.1f}%{label}")
    
    # Save
    with open("experiment1_exact_boost_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Check if any boost helped
    print()
    boost_configs = [k for k in results["conditions"] if "boost" in k]
    best = max([(k, results["conditions"][k]["accuracy"]) for k in boost_configs], key=lambda x: x[1])
    
    if best[1] > baseline_shuf + 5:
        print(f"✓ IMPROVEMENT! {best[0]}: {best[1]:.1f}% (was {baseline_shuf:.1f}%)")
    elif best[1] > baseline_shuf:
        print(f"~ Small improvement: {best[0]}: {best[1]:.1f}% (+{best[1]-baseline_shuf:.1f}%)")
    else:
        print(f"✗ No improvement from boosting. Shuffled already has higher attention to these heads.")
    
    print("\nSaved: experiment1_exact_boost_results.json")


if __name__ == "__main__":
    main()


"""
EXPERIMENT 2: Ablation Study - Disabling Top Copy Heads

Hypothesis: Disabling L17H24, L20H14, L24H27 should hurt accuracy the most
because these are the primary "copy heads" that extract the answer.

Method:
1. Run baseline on unshuffled text → get accuracy
2. Disable top copy heads (zero out their output) → measure accuracy drop
3. Compare which heads cause the biggest drop

Implementation:
- We patch the attention layer's forward method to zero out specific heads' outputs
- This is done DURING the computation, not after
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

# Heads to ablate (from our analysis - top copy heads)
ABLATION_CONFIGS = {
    "baseline": [],  # No ablation
    "L17H24": [(17, 24)],
    "L20H14": [(20, 14)],
    "L24H27": [(24, 27)],
    "top3_copy": [(17, 24), (20, 14), (24, 27)],
    "top5_copy": [(17, 24), (20, 14), (24, 27), (20, 1), (25, 15)],
    "early_L2": [(2, 21), (2, 22), (2, 23)],  # Early layer heads for comparison
}


class HeadAblator:
    """
    Properly ablates attention heads by modifying the forward pass.
    
    In LLaMA, attention output has shape [batch, seq, num_heads, head_dim].
    Before the output projection, we can zero out specific heads.
    """
    
    def __init__(self, model, heads_to_ablate):
        self.model = model
        self.heads_to_ablate = heads_to_ablate
        self.original_forwards = {}
        self.active = False
        
        # Group by layer
        self.layers_to_ablate = {}
        for layer, head in heads_to_ablate:
            if layer not in self.layers_to_ablate:
                self.layers_to_ablate[layer] = []
            self.layers_to_ablate[layer].append(head)
    
    def _create_ablated_forward(self, layer_idx, original_forward, heads_to_zero):
        """Create a forward function that zeros out specific heads"""
        
        def ablated_forward(
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
            # Get the attention module
            attn = self.model.model.layers[layer_idx].self_attn
            
            bsz, q_len, _ = hidden_states.size()
            
            # Project to Q, K, V
            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)
            value_states = attn.v_proj(hidden_states)
            
            # Reshape for multi-head attention
            query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            
            # Apply rotary embeddings
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = attn.rotary_emb(value_states, position_ids)
            
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
            # Handle GQA (grouped query attention)
            key_states = repeat_kv(key_states, attn.num_key_value_groups)
            value_states = repeat_kv(value_states, attn.num_key_value_groups)
            
            # Compute attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (attn.head_dim ** 0.5)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            # *** ABLATION: Zero out attention weights for specific heads ***
            for head_idx in heads_to_zero:
                attn_weights[:, head_idx, :, :] = 0
            
            attn_output = torch.matmul(attn_weights, value_states)
            
            # Reshape and project
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            attn_output = attn.o_proj(attn_output)
            
            if not output_attentions:
                attn_weights = None
            
            return attn_output, attn_weights, past_key_value
        
        return ablated_forward
    
    def enable(self):
        """Enable ablation by patching forward methods"""
        if self.active:
            return
        
        for layer_idx, heads in self.layers_to_ablate.items():
            attn = self.model.model.layers[layer_idx].self_attn
            self.original_forwards[layer_idx] = attn.forward
            attn.forward = self._create_ablated_forward(layer_idx, attn.forward, heads)
        
        self.active = True
    
    def disable(self):
        """Restore original forward methods"""
        if not self.active:
            return
        
        for layer_idx, original in self.original_forwards.items():
            self.model.model.layers[layer_idx].self_attn.forward = original
        
        self.original_forwards = {}
        self.active = False


# Helper functions for rotary embeddings and KV repetition
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    """Repeat KV heads for GQA"""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def run_with_ablation(model, tokenizer, prompt, heads_to_ablate, device="cuda"):
    """Run inference with specific heads ablated"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    ablator = None
    if heads_to_ablate:
        ablator = HeadAblator(model, heads_to_ablate)
        ablator.enable()
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        if ablator:
            ablator.disable()
    
    return response.strip()


def main():
    print("=" * 70)
    print("EXPERIMENT 2: ABLATION STUDY - DISABLING COPY HEADS")
    print("=" * 70)
    print(f"Testing {len(ABLATION_CONFIGS)} configurations")
    print()
    
    # Load model
    print("Loading model...")
    model_name = MODELS[MODEL_KEY]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # Single GPU
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
            "ablation_configs": {k: v for k, v in ABLATION_CONFIGS.items()},
            "context_length": CONTEXT_LENGTH,
            "num_samples": NUM_SAMPLES
        },
        "conditions": {}
    }
    
    # Prepare prompts (unshuffled)
    prompts_and_gts = []
    print("\nPreparing prompts...")
    for sample in samples:
        needle = extract_needle_full_section(
            sample['section_1'],
            sample['ground_truth_state'],
            shuffle=False  # Unshuffled for ablation
        )
        
        context = create_needle_in_haystack(
            needle=needle,
            tokenizer=tokenizer,
            target_tokens=CONTEXT_LENGTH,
            needle_position=NEEDLE_POSITION,
            haystack_text=haystack_text
        )
        
        prompt = QUESTION_TEMPLATE.format(context=context)
        prompts_and_gts.append((prompt, sample['ground_truth_state'], sample['filename']))
    
    # Run each ablation configuration
    for config_name, heads_to_ablate in ABLATION_CONFIGS.items():
        print(f"\n{'=' * 70}")
        print(f"Testing: {config_name}")
        print(f"Ablating: {heads_to_ablate if heads_to_ablate else 'None (baseline)'}")
        print("=" * 70)
        
        correct = 0
        total = 0
        sample_results = []
        
        for prompt, gt, filename in tqdm(prompts_and_gts, desc=config_name):
            try:
                answer = run_with_ablation(model, tokenizer, prompt, heads_to_ablate, device)
                is_correct = check_answer(answer, gt)
            except Exception as e:
                print(f"\nError: {e}")
                answer = "ERROR"
                is_correct = False
            
            if is_correct:
                correct += 1
            total += 1
            
            sample_results.append({
                "filename": filename,
                "gt": gt,
                "answer": answer,
                "correct": is_correct
            })
            
            torch.cuda.empty_cache()
            gc.collect()
        
        accuracy = correct / total * 100
        results["conditions"][config_name] = {
            "heads_ablated": heads_to_ablate,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "samples": sample_results
        }
        
        print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    # Calculate drops from baseline
    baseline_acc = results["conditions"]["baseline"]["accuracy"]
    
    print("\n" + "=" * 70)
    print("SUMMARY: ACCURACY DROP FROM BASELINE")
    print("=" * 70)
    print(f"{'Configuration':<20} | {'Accuracy':>10} | {'Drop':>10}")
    print("-" * 50)
    
    summary = []
    for config_name, data in results["conditions"].items():
        drop = baseline_acc - data["accuracy"]
        print(f"{config_name:<20} | {data['accuracy']:>9.1f}% | {drop:>+9.1f}%")
        summary.append({"config": config_name, "accuracy": data["accuracy"], "drop": drop})
    
    results["summary"] = summary
    
    # Save results
    with open("experiment2_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nSaved: experiment2_ablation_results.json")
    
    # Find which ablation hurt most
    non_baseline = [s for s in summary if s["config"] != "baseline"]
    if non_baseline:
        sorted_by_drop = sorted(non_baseline, key=lambda x: x["drop"], reverse=True)
        print(f"\nMost impactful ablation: {sorted_by_drop[0]['config']} (drop: {sorted_by_drop[0]['drop']:.1f}%)")
        
        if sorted_by_drop[0]['drop'] > 5:
            print("✓ HYPOTHESIS SUPPORTED: Ablating copy heads significantly hurts accuracy!")
        else:
            print("~ INCONCLUSIVE: Ablation effect was small.")


if __name__ == "__main__":
    main()

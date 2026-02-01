"""
EXPERIMENT 6: CORRECT Head Boosting (Fixed Implementation)

Previous experiments had a critical bug:
- We were modifying attention outputs AFTER o_proj
- At that point, all head outputs are already mixed together
- Scaling doesn't separate individual heads

CORRECT approach:
- Patch the forward method to intervene BEFORE o_proj
- Scale specific head outputs in the [batch, heads, seq, head_dim] space
- Then pass through o_proj

This should actually affect model behavior!
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

# Heads that are STRONGER in unshuffled (from our data)
HEADS_TO_BOOST = [
    (29, 11),  # +0.26 diff
    (5, 7),    # +0.25 diff
    (24, 27),  # +0.24 diff
    (29, 9),   # +0.23 diff
    (6, 24),   # +0.20 diff
    (0, 14),   # +0.20 diff
    (5, 5),    # +0.19 diff
    (5, 24),   # +0.19 diff
]

# Test configs
BOOST_CONFIGS = {
    "baseline_shuffled": {"heads": [], "scale": 1.0},
    "top5_heads_1.5x": {"heads": HEADS_TO_BOOST[:5], "scale": 1.5},
    "top5_heads_2x": {"heads": HEADS_TO_BOOST[:5], "scale": 2.0},
    "top8_heads_1.5x": {"heads": HEADS_TO_BOOST[:8], "scale": 1.5},
}


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class CorrectHeadBooster:
    """
    Correctly boost specific heads by patching forward methods
    to intervene BEFORE o_proj.
    """
    
    def __init__(self, model, heads_to_boost, scale_factor):
        self.model = model
        self.scale_factor = scale_factor
        self.original_forwards = {}
        
        # Group by layer
        self.layers_to_heads = {}
        for layer, head in heads_to_boost:
            if layer not in self.layers_to_heads:
                self.layers_to_heads[layer] = []
            self.layers_to_heads[layer].append(head)
    
    def _create_patched_forward(self, layer_idx, heads_to_scale):
        """Create a patched forward that scales specific heads BEFORE o_proj"""
        model = self.model
        scale = self.scale_factor
        
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
            
            # *** SCALE SPECIFIC HEADS HERE (BEFORE o_proj) ***
            for head_idx in heads_to_scale:
                attn_output[:, head_idx, :, :] = attn_output[:, head_idx, :, :] * scale
            
            # Reshape and project
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            attn_output = attn.o_proj(attn_output)
            
            if not output_attentions:
                attn_weights = None
            
            return attn_output, attn_weights, past_key_value
        
        return patched_forward
    
    def enable(self):
        """Patch forward methods for layers with heads to scale"""
        for layer_idx, heads in self.layers_to_heads.items():
            attn = self.model.model.layers[layer_idx].self_attn
            self.original_forwards[layer_idx] = attn.forward
            attn.forward = self._create_patched_forward(layer_idx, heads)
    
    def disable(self):
        """Restore original forward methods"""
        for layer_idx, original in self.original_forwards.items():
            self.model.model.layers[layer_idx].self_attn.forward = original
        self.original_forwards = {}


def run_with_boost(model, tokenizer, prompt, heads, scale, device="cuda"):
    """Run inference with correct head boosting"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    booster = None
    if heads and scale != 1.0:
        booster = CorrectHeadBooster(model, heads, scale)
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
    print("EXPERIMENT 6: CORRECT HEAD BOOSTING")
    print("=" * 70)
    print("Intervening BEFORE o_proj to actually scale head outputs")
    print()
    print("Heads to boost (strongest in unshuffled):")
    for layer, head in HEADS_TO_BOOST[:5]:
        print(f"  L{layer}H{head}")
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
    shuf_prompts = []
    unshuf_prompts = []
    
    for sample in samples:
        # Shuffled
        needle = extract_needle_full_section(
            sample['section_1'], sample['ground_truth_state'], shuffle=True
        )
        context = create_needle_in_haystack(
            needle=needle, tokenizer=tokenizer,
            target_tokens=CONTEXT_LENGTH, needle_position=NEEDLE_POSITION,
            haystack_text=haystack_text
        )
        shuf_prompts.append({
            "prompt": QUESTION_TEMPLATE.format(context=context),
            "gt": sample['ground_truth_state'],
            "filename": sample['filename']
        })
        
        # Unshuffled
        needle = extract_needle_full_section(
            sample['section_1'], sample['ground_truth_state'], shuffle=False
        )
        context = create_needle_in_haystack(
            needle=needle, tokenizer=tokenizer,
            target_tokens=CONTEXT_LENGTH, needle_position=NEEDLE_POSITION,
            haystack_text=haystack_text
        )
        unshuf_prompts.append({
            "prompt": QUESTION_TEMPLATE.format(context=context),
            "gt": sample['ground_truth_state'],
            "filename": sample['filename']
        })
    
    results = {"config": BOOST_CONFIGS, "conditions": {}}
    
    # Unshuffled baseline
    print(f"\n{'=' * 70}")
    print("Testing: baseline_unshuffled")
    print("=" * 70)
    correct = 0
    for data in tqdm(unshuf_prompts, desc="unshuffled"):
        answer = run_with_boost(model, tokenizer, data["prompt"], [], 1.0, device)
        if check_answer(answer, data["gt"]):
            correct += 1
        torch.cuda.empty_cache()
    unshuf_acc = correct / len(unshuf_prompts) * 100
    results["conditions"]["baseline_unshuffled"] = {"accuracy": unshuf_acc}
    print(f"Accuracy: {unshuf_acc:.1f}%")
    
    # Test each config on shuffled
    for config_name, config in BOOST_CONFIGS.items():
        print(f"\n{'=' * 70}")
        print(f"Testing: {config_name}")
        if config["heads"]:
            print(f"Heads: {config['heads']}")
            print(f"Scale: {config['scale']}x")
        print("=" * 70)
        
        correct = 0
        sample_results = []
        
        for data in tqdm(shuf_prompts, desc=config_name):
            try:
                answer = run_with_boost(
                    model, tokenizer, data["prompt"],
                    config["heads"], config["scale"], device
                )
                is_correct = check_answer(answer, data["gt"])
            except Exception as e:
                print(f"\nError: {e}")
                answer = "ERROR"
                is_correct = False
            
            if is_correct:
                correct += 1
            sample_results.append({"gt": data["gt"], "answer": answer, "correct": is_correct})
            torch.cuda.empty_cache()
            gc.collect()
        
        accuracy = correct / len(shuf_prompts) * 100
        results["conditions"][config_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(shuf_prompts),
            "samples": sample_results
        }
        print(f"Accuracy: {accuracy:.1f}%")
    
    # Summary
    shuf_baseline = results["conditions"]["baseline_shuffled"]["accuracy"]
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<25} | {'Accuracy':>10} | {'vs Baseline':>12}")
    print("-" * 55)
    
    for config_name, data in results["conditions"].items():
        change = data["accuracy"] - shuf_baseline if config_name != "baseline_unshuffled" else data["accuracy"] - shuf_baseline
        label = "(target)" if config_name == "baseline_unshuffled" else ""
        print(f"{config_name:<25} | {data['accuracy']:>9.1f}% | {change:>+11.1f}% {label}")
    
    # Save
    with open("experiment6_correct_boost_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Check improvement
    best = max([(k, v["accuracy"]) for k, v in results["conditions"].items() 
                if k not in ["baseline_shuffled", "baseline_unshuffled"]], 
               key=lambda x: x[1], default=(None, 0))
    
    print()
    if best[1] > shuf_baseline + 3:
        print(f"✓ IMPROVEMENT! {best[0]}: {best[1]:.1f}% (+{best[1]-shuf_baseline:.1f}%)")
    elif best[1] > shuf_baseline:
        print(f"~ Small improvement: {best[0]} (+{best[1]-shuf_baseline:.1f}%)")
    else:
        print("✗ No improvement")
    
    print("\nSaved: experiment6_correct_boost_results.json")


if __name__ == "__main__":
    main()


# -*- coding: utf-8 -*-
"""
Needle in Haystack Context Length Sweep - FULL SECTION 1 VERSION

This script tests at what context length the model fails to retrieve
a simple fact (state of incorporation) when the answer is buried in
a large chunk of text (entire Section 1) surrounded by irrelevant
"haystack" content from Alice in Wonderland.

REPLICATING ANANYA'S APPROACH:
- Needle = ENTIRE Section 1 (hundreds of tokens with answer buried inside)
- NOT just one sentence - makes it harder!
- The state is mentioned somewhere within Section 1
- Model must process all of Section 1 to find the answer

Goal: Find the context length threshold where retrieval breaks down.

Methodology:
1. Use Alice in Wonderland as irrelevant padding (haystack)
2. Embed entire Section 1 (needle) at position 0.5 (middle of haystack)
3. Sweep context lengths: 200, 500, 1K, 2K, 4K, 8K, 10K, 12K, 15K tokens
4. Measure accuracy at each length to find failure point

Expected Results (based on Ananya's findings):
- Short contexts (< 4K): ~80%+ accuracy
- Around 10K tokens: Dramatic degradation begins
- 15K tokens: Near-zero accuracy
"""

import os
import json
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODELS = {
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",  # Llama 3 (same as Ananya)
    "llama31": "meta-llama/Llama-3.1-8B-Instruct",   # Llama 3.1 (better long-context)
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

# Context lengths to sweep (extended to 50K for Llama 3.1 testing)
CONTEXT_LENGTHS = [200, 500, 1000, 2000, 4000, 8000, 10000, 15000, 20000, 30000, 50000]

# Needle positions to test (where in the haystack is the fact placed)
# 0.0 = beginning, 0.5 = middle, 1.0 = end
NEEDLE_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]

QUESTION_TEMPLATE = """Based on the following text, answer the question.

Context:
{context}

Question: In which US state was this company incorporated?
Answer with just the state name:"""

NUM_SAMPLES = 30  # Per context length

# =============================================================================
# PROJECT GUTENBERG HAYSTACK
# =============================================================================

ALICE_IN_WONDERLAND = """
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out again.

The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled "ORANGE MARMALADE", but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody underneath, so managed to put it into one of the cupboards as she fell past it.

"Well!" thought Alice to herself, "after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!" (Which was very likely true.)

Down, down, down. Would the fall never come to an end? "I wonder how many miles I've fallen by this time?" she said aloud. "I must be getting somewhere near the centre of the earth. Let me see: that would be four thousand miles down, I think—" (for, you see, Alice had learnt several things of this sort in her lessons in the schoolroom, and though this was not a very good opportunity for showing off her knowledge, as there was no one to listen to her, still it was good practice to say it over) "—yes, that's about the right distance—but then I wonder what Latitude or Longitude I've got to?" (Alice had no idea what Latitude was, or Longitude either, but thought they were nice grand words to say.)

Presently she began again. "I wonder if I shall fall right through the earth! How funny it'll seem to come out among the people that walk with their heads downward! The Antipathies, I think—" (she was rather glad there was no one listening, this time, as it didn't sound at all the right word) "—but I shall have to ask them what the name of the country is, you know. Please, Ma'am, is this New Zealand or Australia?" (and she tried to curtsey as she spoke—fancy curtseying as you're falling through the air! Do you think you could manage it?) "And what an ignorant little girl she'll think me for asking! No, it'll never do to ask: perhaps I shall see it written up somewhere."

Down, down, down. There was nothing else to do, so Alice soon began talking again. "Dinah'll miss me very much to-night, I should think!" (Dinah was the cat.) "I hope they'll remember her saucer of milk at tea-time. Dinah my dear! I wish you were down here with me! There are no mice in the air, I'm afraid, but you might catch a bat, and that's very like a mouse, you know. But do cats eat bats, I wonder?" And here Alice began to get rather sleepy, and went on saying to herself, in a dreamy sort of way, "Do cats eat bats? Do cats eat bats?" and sometimes, "Do bats eat cats?" for, you see, as she couldn't answer either question, it didn't much matter which way she put it. She felt that she was dozing off, and had just begun to dream that she was walking hand in hand with Dinah, and saying to her very earnestly, "Now, Dinah, tell me the truth: did you ever eat a bat?" when suddenly, thump! thump! down she came upon a heap of sticks and dry leaves, and the fall was over.

Alice was not a bit hurt, and she jumped up on to her feet in a moment: she looked up, but it was all dark overhead; before her was another long passage, and the White Rabbit was still in sight, hurrying down it. There was not a moment to be lost: away went Alice like the wind, and was just in time to hear it say, as it turned a corner, "Oh my ears and whiskers, how late it's getting!" She was close behind it when she turned the corner, but the Rabbit was no longer to be seen: she found herself in a long, low hall, which was lit up by a row of lamps hanging from the roof.

There were doors all round the hall, but they were all locked; and when Alice had been all the way down one side and up the other, trying every door, she walked sadly down the middle, wondering how she was ever to get out again.

Suddenly she came upon a little three-legged table, all made of solid glass; there was nothing on it except a tiny golden key, and Alice's first thought was that it might belong to one of the doors of the hall; but, alas! either the locks were too large, or the key was too small, but at any rate it would not open any of them. However, on the second time round, she came upon a low curtain she had not noticed before, and behind it was a little door about fifteen inches high: she tried the little golden key in the lock, and to her great delight it fitted!

Alice opened the door and found that it led into a small passage, not much larger than a rat-hole: she knelt down and looked along the passage into the loveliest garden you ever saw. How she longed to get out of that dark hall, and wander about among those beds of bright flowers and those cool fountains, but she could not even get her head through the doorway; "and even if my head would go through," thought poor Alice, "it would be of very little use without my shoulders. Oh, how I wish I could shut up like a telescope! I think I could, if I only knew how to begin." For, you see, so many out-of-the-way things had happened lately, that Alice had begun to think that very few things indeed were really impossible.

There seemed to be no use in waiting by the little door, so she went back to the table, half hoping she might find another key on it, or at any rate a book of rules for shutting people up like telescopes: this time she found a little bottle on it, ("which certainly was not here before," said Alice,) and round the neck of the bottle was a paper label, with the words "DRINK ME" beautifully printed on it in large letters.
"""

def get_haystack_text(min_length=50000):
    """
    Get haystack text from Project Gutenberg or use built-in Alice text.
    Returns a long string of irrelevant content.
    """
    # Use built-in Alice text, repeated to get desired length
    haystack = ALICE_IN_WONDERLAND
    while len(haystack) < min_length:
        haystack = haystack + "\n\n" + ALICE_IN_WONDERLAND
    return haystack

# =============================================================================
# GROUND TRUTH
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

def load_edgar_samples(num_samples, ground_truth):
    """Load EDGAR samples with ground truth."""
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
            'ground_truth_state': ground_truth[filename],
        })
    
    print(f"Loaded {len(samples)} samples")
    return samples

# =============================================================================
# NEEDLE IN HAYSTACK CONSTRUCTION
# =============================================================================

def extract_needle_full_section(section_1: str, state: str, shuffle: bool = False) -> str:
    """
    Return the ENTIRE Section 1 as the needle (like Ananya's approach).
    This makes the task harder - the answer is buried somewhere in Section 1,
    not in an obvious single sentence.
    
    The model must process all of Section 1 to find where the state is mentioned.
    
    If shuffle=True, shuffle the words in the section (destroys word order but keeps all words).
    """
    text = section_1.strip()
    
    if shuffle:
        import random
        words = text.split()
        random.shuffle(words)
        text = ' '.join(words)
    
    return text

def create_needle_in_haystack(
    needle: str,
    tokenizer,
    target_tokens: int,
    needle_position: float = 0.5,
    haystack_text: str = None
) -> str:
    """
    Create a context with the needle embedded in irrelevant haystack.
    
    Args:
        needle: The fact to embed (e.g., "The company was incorporated in Delaware.")
        tokenizer: Tokenizer for token counting
        target_tokens: Total context length in tokens
        needle_position: Where to place needle (0.0=start, 0.5=middle, 1.0=end)
        haystack_text: Irrelevant text to use as padding
    
    Returns:
        Context string with needle embedded in haystack
    """
    if haystack_text is None:
        haystack_text = get_haystack_text()
    
    # Tokenize needle
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
    needle_token_count = len(needle_tokens)
    
    # Calculate haystack tokens needed
    haystack_tokens_needed = target_tokens - needle_token_count
    
    if haystack_tokens_needed <= 0:
        return needle
    
    # Calculate split based on position
    tokens_before = int(haystack_tokens_needed * needle_position)
    tokens_after = haystack_tokens_needed - tokens_before
    
    # Tokenize haystack and split
    haystack_tokens = tokenizer.encode(haystack_text, add_special_tokens=False)
    
    # Get tokens for before and after
    if len(haystack_tokens) < haystack_tokens_needed:
        # Repeat haystack if needed
        while len(haystack_tokens) < haystack_tokens_needed:
            haystack_tokens = haystack_tokens + haystack_tokens
    
    before_tokens = haystack_tokens[:tokens_before]
    after_tokens = haystack_tokens[tokens_before:tokens_before + tokens_after]
    
    # Decode back to text
    before_text = tokenizer.decode(before_tokens, skip_special_tokens=True)
    after_text = tokenizer.decode(after_tokens, skip_special_tokens=True)
    
    # Combine
    context = f"{before_text}\n\n{needle}\n\n{after_text}"
    
    return context

# =============================================================================
# MODEL INFERENCE
# =============================================================================

def load_model(model_name):
    """Load model for inference."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    return model, tokenizer

def run_inference(model, tokenizer, prompt, max_new_tokens=15):
    """Run inference and return answer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def check_answer(answer: str, ground_truth: str) -> bool:
    """Check if answer matches ground truth."""
    answer_lower = answer.lower().strip().split()[0] if answer.strip() else ""
    truth_lower = ground_truth.lower().strip()
    
    # Handle multi-word states
    if truth_lower.startswith("new ") or truth_lower.startswith("north ") or truth_lower.startswith("south "):
        truth_first = truth_lower.split()[0]
        return answer_lower == truth_first or truth_lower in answer.lower()
    
    return answer_lower == truth_lower or truth_lower in answer_lower or answer_lower in truth_lower

# =============================================================================
# SWEEP EXPERIMENT
# =============================================================================

def run_context_length_sweep(
    model_key: str,
    samples: list,
    context_lengths: list = CONTEXT_LENGTHS,
    needle_position: float = 0.5,
    save_dir: str = ".",
    shuffle_needle: bool = False
) -> dict:
    """
    Run the context length sweep experiment.
    
    Tests retrieval accuracy at each context length.
    """
    model_name = MODELS[model_key]
    model, tokenizer = load_model(model_name)
    
    haystack = get_haystack_text()
    
    results_by_length = {}
    
    for ctx_len in context_lengths:
        print(f"\n{'='*60}")
        print(f"Testing context length: {ctx_len} tokens")
        print(f"{'='*60}")
        
        results = []
        correct = 0
        
        for sample in tqdm(samples, desc=f"{ctx_len} tokens"):
            # Extract the needle (ENTIRE Section 1, not just one sentence)
            needle = extract_needle_full_section(
                sample['section_1'], 
                sample['ground_truth_state'],
                shuffle=shuffle_needle
            )
            
            # Create context with needle in haystack
            context = create_needle_in_haystack(
                needle=needle,
                tokenizer=tokenizer,
                target_tokens=ctx_len,
                needle_position=needle_position,
                haystack_text=haystack
            )
            
            # Create prompt
            prompt = QUESTION_TEMPLATE.format(context=context)
            
            # Run inference
            try:
                answer = run_inference(model, tokenizer, prompt)
            except Exception as e:
                print(f"Error: {e}")
                answer = ""
            
            # Check accuracy
            is_correct = check_answer(answer, sample['ground_truth_state'])
            if is_correct:
                correct += 1
            
            results.append({
                'filename': sample['filename'],
                'ground_truth': sample['ground_truth_state'],
                'answer': answer,
                'correct': is_correct,
                'needle': needle,
            })
        
        accuracy = correct / len(results) if results else 0
        results_by_length[ctx_len] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(results),
            'results': results,
        }
        
        print(f"Accuracy at {ctx_len} tokens: {accuracy:.1%} ({correct}/{len(results)})")
    
    # Save results
    shuffle_label = "_shuffled" if shuffle_needle else ""
    output = {
        'model': model_key,
        'model_name': model_name,
        'needle_position': needle_position,
        'shuffle_needle': shuffle_needle,
        'results_by_length': {
            str(k): {key: val for key, val in v.items() if key != 'results'}
            for k, v in results_by_length.items()
        },
        'detailed_results': {
            str(k): v['results'] for k, v in results_by_length.items()
        },
    }
    
    # Find next available filename (incremental)
    i = 0
    while os.path.exists(os.path.join(save_dir, f"{model_key}_context_sweep{shuffle_label}_{i}.json")):
        i += 1
    output_path = os.path.join(save_dir, f"{model_key}_context_sweep{shuffle_label}_{i}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved: {output_path}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return output

def run_position_sweep(
    model_key: str,
    samples: list,
    context_length: int = 2000,
    needle_positions: list = NEEDLE_POSITIONS,
    save_dir: str = "."
) -> dict:
    """
    Test how needle position affects retrieval at a fixed context length.
    """
    model_name = MODELS[model_key]
    model, tokenizer = load_model(model_name)
    
    haystack = get_haystack_text()
    
    results_by_position = {}
    
    for position in needle_positions:
        print(f"\n{'='*60}")
        print(f"Testing needle position: {position:.0%} through context")
        print(f"{'='*60}")
        
        results = []
        correct = 0
        
        for sample in tqdm(samples, desc=f"Position {position:.0%}"):
            needle = extract_needle_full_section(
                sample['section_1'],
                sample['ground_truth_state']
            )
            
            context = create_needle_in_haystack(
                needle=needle,
                tokenizer=tokenizer,
                target_tokens=context_length,
                needle_position=position,
                haystack_text=haystack
            )
            
            prompt = QUESTION_TEMPLATE.format(context=context)
            
            try:
                answer = run_inference(model, tokenizer, prompt)
            except Exception as e:
                answer = ""
            
            is_correct = check_answer(answer, sample['ground_truth_state'])
            if is_correct:
                correct += 1
            
            results.append({
                'filename': sample['filename'],
                'ground_truth': sample['ground_truth_state'],
                'answer': answer,
                'correct': is_correct,
            })
        
        accuracy = correct / len(results) if results else 0
        results_by_position[position] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(results),
        }
        
        print(f"Accuracy at position {position:.0%}: {accuracy:.1%}")
    
    # Save
    output = {
        'model': model_key,
        'context_length': context_length,
        'results_by_position': results_by_position,
    }
    
    output_path = os.path.join(save_dir, f"{model_key}_position_sweep.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    del model
    torch.cuda.empty_cache()
    
    return output

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_context_sweep(save_dir: str):
    """Create visualization of context length sweep results."""
    
    results = {}
    for model_key in MODELS.keys():
        path = os.path.join(save_dir, f"{model_key}_context_sweep.json")
        if os.path.exists(path):
            with open(path) as f:
                results[model_key] = json.load(f)
    
    if not results:
        print("No results to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'llama': '#e74c3c', 'qwen': '#3498db'}
    markers = {'llama': 'o', 'qwen': 's'}
    
    for model_key, data in results.items():
        lengths = []
        accuracies = []
        
        for length_str, result in data['results_by_length'].items():
            lengths.append(int(length_str))
            accuracies.append(result['accuracy'] * 100)
        
        # Sort by length
        sorted_data = sorted(zip(lengths, accuracies))
        lengths, accuracies = zip(*sorted_data)
        
        ax.plot(lengths, accuracies, 
                marker=markers.get(model_key, 'o'),
                color=colors.get(model_key, 'gray'),
                linewidth=2,
                markersize=10,
                label=MODELS[model_key].split('/')[-1])
    
    ax.set_xlabel('Context Length (tokens)', fontsize=12)
    ax.set_ylabel('Retrieval Accuracy (%)', fontsize=12)
    ax.set_title('Needle in Haystack: Accuracy vs Context Length\n(Haystack = Alice in Wonderland)', fontsize=14)
    ax.set_xscale('log')
    ax.set_xticks(CONTEXT_LENGTHS)
    ax.set_xticklabels([str(x) for x in CONTEXT_LENGTHS])
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for failure points
    for model_key, data in results.items():
        for length_str, result in data['results_by_length'].items():
            if result['accuracy'] < 0.5:  # Below 50%
                ax.annotate(f"↓ Failure",
                           (int(length_str), result['accuracy'] * 100),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center',
                           fontsize=9,
                           color='red')
                break
    
    plt.tight_layout()
    # Find next available filename (incremental)
    i = 0
    while os.path.exists(os.path.join(save_dir, f'context_length_sweep_{i}.png')):
        i += 1
    png_path = os.path.join(save_dir, f'context_length_sweep_{i}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {png_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Needle in Haystack Sweep")
    parser.add_argument('--model', type=str, choices=['llama', 'llama31', 'qwen', 'all'], default='all')
    parser.add_argument('--samples', type=int, default=NUM_SAMPLES)
    parser.add_argument('--sweep', type=str, choices=['length', 'position', 'both'], default='length')
    parser.add_argument('--visualize-only', action='store_true')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle words in the needle (Section 1)')
    
    args = parser.parse_args()
    
    save_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.visualize_only:
        plot_context_sweep(save_dir)
        return
    
    # Load data
    ground_truth = load_ground_truth()
    samples = load_edgar_samples(args.samples, ground_truth)
    
    models_to_run = list(MODELS.keys()) if args.model == 'all' else [args.model]
    
    for model_key in models_to_run:
        if args.sweep in ['length', 'both']:
            run_context_length_sweep(model_key, samples, save_dir=save_dir, shuffle_needle=args.shuffle)
        
        if args.sweep in ['position', 'both']:
            run_position_sweep(model_key, samples, save_dir=save_dir)
    
    # Visualize
    plot_context_sweep(save_dir)
    
    print("\n" + "="*60)
    print("NEEDLE IN HAYSTACK SWEEP COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()


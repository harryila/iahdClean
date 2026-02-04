"""
Batch runner for Phase 2, Method 2 (Wu24 Retrieval Head)

Runs all 32 experiments: 2 models × 4 questions × 4 token lengths
"""

import os
import sys
import subprocess
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

MODELS = ["instruct", "base"]
QUESTIONS = ["inc_state", "inc_year", "employee_count", "hq_state"]
TOKEN_LENGTHS = [2048, 4096, 6144, 8192]

def check_exists(model_key, question_key, tokens):
    """Check if result already exists."""
    model_dir = "llama3_instruct" if model_key == "instruct" else "llama3_base"
    output_path = os.path.join(RESULTS_DIR, model_dir, question_key, f"tokens_{tokens}.json")
    return os.path.exists(output_path)

def run_experiment(model_key, question_key, tokens):
    """Run a single experiment."""
    cmd = [
        sys.executable, "run_detection.py",
        "--model", model_key,
        "--question", question_key,
        "--tokens", str(tokens)
    ]
    
    print(f"\n{'='*70}")
    print(f"Running: model={model_key}, question={question_key}, tokens={tokens}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0

def main():
    total = len(MODELS) * len(QUESTIONS) * len(TOKEN_LENGTHS)
    completed = 0
    skipped = 0
    
    print(f"Wu24 Retrieval Head Detection - Batch Runner")
    print(f"Total experiments: {total}")
    print()
    
    for model in MODELS:
        for question in QUESTIONS:
            for tokens in TOKEN_LENGTHS:
                # Check if already done
                if check_exists(model, question, tokens):
                    print(f"SKIP: {model}/{question}/{tokens} (exists)")
                    skipped += 1
                    completed += 1
                    continue
                
                # Clear GPU memory
                torch.cuda.empty_cache()
                
                # Run experiment
                success = run_experiment(model, question, tokens)
                
                if success:
                    completed += 1
                else:
                    print(f"FAILED: {model}/{question}/{tokens}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {completed}/{total} completed, {skipped} skipped")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

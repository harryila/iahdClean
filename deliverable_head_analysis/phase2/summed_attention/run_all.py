#!/usr/bin/env python3
"""Batch runner for all Phase 2 Method 1 experiments."""

import subprocess
import sys
import os
import torch

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODELS = ["instruct", "base"]
QUESTIONS = ["inc_state", "inc_year", "employee_count", "hq_state"]
TOKEN_LENGTHS = [2048, 4096, 6144, 8192]

def check_exists(model, question, tokens):
    """Check if result already exists."""
    model_dir = "llama3_instruct" if model == "instruct" else "llama3_base"
    path = f"results/{model_dir}/{question}/tokens_{tokens}.json"
    return os.path.exists(path)

def run_experiment(model, question, tokens):
    """Run a single experiment."""
    print(f"\n{'='*70}")
    print(f"Running: model={model}, question={question}, tokens={tokens}")
    print(f"{'='*70}")
    
    # Clear GPU before each run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    result = subprocess.run(
        [sys.executable, "run_detection.py", 
         "--model", model,
         "--question", question,
         "--tokens", str(tokens)],
        capture_output=False
    )
    return result.returncode == 0

def main():
    total = len(MODELS) * len(QUESTIONS) * len(TOKEN_LENGTHS)
    completed = 0
    skipped = 0
    failed = 0
    
    for model in MODELS:
        for question in QUESTIONS:
            for tokens in TOKEN_LENGTHS:
                if check_exists(model, question, tokens):
                    print(f"[SKIP] {model}/{question}/{tokens} already exists")
                    skipped += 1
                    continue
                
                success = run_experiment(model, question, tokens)
                if success:
                    completed += 1
                else:
                    failed += 1
                    print(f"[FAIL] {model}/{question}/{tokens}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total experiments: {total}")
    print(f"Skipped (already done): {skipped}")
    print(f"Completed this run: {completed}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()

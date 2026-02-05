#!/usr/bin/env python3
"""
Batch runner for Phase 3 Method 1 (Summed Attention Ablation).
Runs all 32 experiments: 2 models × 4 questions × 4 token lengths.
"""
import subprocess
import sys
from itertools import product

MODELS = ["instruct", "base"]
QUESTIONS = ["inc_state", "inc_year", "employee_count", "hq_state"]
TOKEN_LENGTHS = [2048, 4096, 6144, 8192]

def main():
    total = len(MODELS) * len(QUESTIONS) * len(TOKEN_LENGTHS)
    completed = 0
    
    print(f"Running {total} experiments for Phase 3 Method 1 (Summed Attention Ablation)")
    print("=" * 70)
    
    for model, question, tokens in product(MODELS, QUESTIONS, TOKEN_LENGTHS):
        completed += 1
        print(f"\n[{completed}/{total}] {model} / {question} / {tokens} tokens")
        print("-" * 50)
        
        cmd = [
            sys.executable,
            "run_ablation.py",
            "--model", model,
            "--question", question,
            "--tokens", str(tokens),
        ]
        
        result = subprocess.run(cmd, cwd=__file__.rsplit("/", 1)[0])
        
        if result.returncode != 0:
            print(f"WARNING: Experiment failed with code {result.returncode}")
    
    print("\n" + "=" * 70)
    print(f"Completed {completed}/{total} experiments")


if __name__ == "__main__":
    main()

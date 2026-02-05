#!/usr/bin/env python3
"""
Batch runner for Phase 3 Method 2: Wu24 Retrieval Head ablation experiments.
Runs all 32 configurations: 2 models × 4 questions × 4 token lengths
"""
import subprocess
import sys
import os

MODELS = ["instruct", "base"]
QUESTIONS = ["inc_state", "inc_year", "employee_count", "hq_state"]
TOKEN_LENGTHS = [2048, 4096, 6144, 8192]

def main():
    total = len(MODELS) * len(QUESTIONS) * len(TOKEN_LENGTHS)
    completed = 0
    
    print(f"Starting Phase 3 Method 2 (Wu24) Ablation batch run: {total} experiments")
    print("=" * 70)
    
    for model in MODELS:
        for question in QUESTIONS:
            for tokens in TOKEN_LENGTHS:
                completed += 1
                print(f"\n[{completed}/{total}] Model={model}, Question={question}, Tokens={tokens}")
                print("-" * 50)
                
                cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "run_ablation.py"),
                    "--model", model,
                    "--question", question,
                    "--tokens", str(tokens),
                ]
                
                result = subprocess.run(cmd, capture_output=False)
                
                if result.returncode != 0:
                    print(f"ERROR: Experiment failed!")
                else:
                    print(f"Completed successfully")
    
    print("\n" + "=" * 70)
    print(f"BATCH COMPLETE: {completed}/{total} experiments")

if __name__ == "__main__":
    main()

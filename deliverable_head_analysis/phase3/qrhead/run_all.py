#!/usr/bin/env python3
"""
Batch runner for Phase 3 Method 3: QRHead Ablation Study

Runs all 32 experiments:
- 2 models (instruct, base)
- 4 questions (inc_state, inc_year, employee_count, hq_state)
- 4 token lengths (2048, 4096, 6144, 8192)
"""

import subprocess
import sys
from datetime import datetime

MODELS = ["instruct", "base"]
QUESTIONS = ["inc_state", "inc_year", "employee_count", "hq_state"]
TOKEN_LENGTHS = [2048, 4096, 6144, 8192]


def main():
    total = len(MODELS) * len(QUESTIONS) * len(TOKEN_LENGTHS)
    completed = 0
    failed = []
    
    print("=" * 70)
    print(f"Phase 3 Method 3 (QRHead) Ablation - Batch Run")
    print(f"Total experiments: {total}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    for model in MODELS:
        for question in QUESTIONS:
            for tokens in TOKEN_LENGTHS:
                completed += 1
                print(f"\n[{completed}/{total}] {model} / {question} / {tokens} tokens")
                print("-" * 50)
                
                cmd = [
                    sys.executable,
                    "run_ablation.py",
                    "--model", model,
                    "--question", question,
                    "--tokens", str(tokens)
                ]
                
                try:
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=False
                    )
                    print(f"[OK] Completed: {model}/{question}/{tokens}")
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] Failed: {model}/{question}/{tokens}")
                    failed.append((model, question, tokens))
                except Exception as e:
                    print(f"[ERROR] Exception: {e}")
                    failed.append((model, question, tokens))
    
    print("\n" + "=" * 70)
    print("BATCH RUN COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"Total: {total}")
    print(f"Completed: {completed - len(failed)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed experiments:")
        for model, question, tokens in failed:
            print(f"  - {model} / {question} / {tokens}")


if __name__ == "__main__":
    main()

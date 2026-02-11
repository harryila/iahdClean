#!/usr/bin/env python3
"""
Batch runner for Phase 3 Overlap Ablation.

Runs overlap ablation for all (model, question, token_length) combinations.
Groups by model to avoid reloading the model repeatedly.

Usage:
    python run_all.py                          # Run everything
    python run_all.py --model instruct         # Just instruct model
    python run_all.py --model instruct --tokens 4096  # Single token length
    python run_all.py --skip-random            # Skip random baseline
"""

import argparse
import subprocess
import sys
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABLATION_SCRIPT = os.path.join(SCRIPT_DIR, "run_ablation.py")
OVERLAP_JSON = os.path.join(SCRIPT_DIR, "..", "..", "phase4", "figures", "qr_wu24_overlap_heads.json")

MODELS = ["instruct", "base"]
QUESTIONS = ["inc_state", "inc_year", "employee_count", "hq_state"]
TOKEN_LENGTHS = [2048, 4096, 6144, 8192]


def main():
    parser = argparse.ArgumentParser(description="Batch runner for overlap ablation")
    parser.add_argument("--model", choices=MODELS, help="Run only this model")
    parser.add_argument("--question", choices=QUESTIONS, help="Run only this question")
    parser.add_argument("--tokens", type=int, choices=TOKEN_LENGTHS, help="Run only this token length")
    parser.add_argument("--skip-random", action="store_true", help="Skip random baseline")
    args = parser.parse_args()

    models = [args.model] if args.model else MODELS
    questions = [args.question] if args.question else QUESTIONS
    token_lengths = [args.tokens] if args.tokens else TOKEN_LENGTHS

    # Check which combos have overlap heads
    with open(OVERLAP_JSON, "r") as f:
        overlap_data = json.load(f)

    total_runs = 0
    skipped = 0
    failed = 0

    for model in models:
        for question in questions:
            mkey = model  # JSON uses "instruct"/"base"
            n_overlap = overlap_data.get(mkey, {}).get(question, {}).get("n_overlap", 0)
            if n_overlap == 0:
                print(f"SKIP {model}/{question}: no overlap heads")
                skipped += 1
                continue

            for tokens in token_lengths:
                total_runs += 1
                print(f"\n{'='*70}")
                print(f"RUN {total_runs}: {model} / {question} / {tokens} tokens ({n_overlap} overlap heads)")
                print(f"{'='*70}")

                cmd = [
                    sys.executable, ABLATION_SCRIPT,
                    "--model", model,
                    "--question", question,
                    "--tokens", str(tokens),
                ]
                if args.skip_random:
                    cmd.append("--skip-random")

                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"FAILED: {model}/{question}/{tokens}")
                    failed += 1

    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE: {total_runs} runs, {skipped} skipped, {failed} failed")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

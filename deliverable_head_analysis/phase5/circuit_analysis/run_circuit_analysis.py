#!/usr/bin/env python3
"""
Run the circuit analysis experiments for Phase C.

Primary config: summed_attention / base / inc_year / 2048 (strongest non-monotonic signal).
Optional: additional configs for robustness.
"""
import subprocess
import sys
import os

SCRIPT = os.path.join(os.path.dirname(__file__), "attention_comparison.py")

# Primary experiment (Config 1 from Phase A — the statistically significant one)
CONFIGS = [
    {
        "method": "summed_attention",
        "model": "base",
        "question": "inc_year",
        "tokens": 2048,
        "ablate_top_n": 5,
        "label": "PRIMARY: strongest non-monotonic signal (bootstrap confirmed)",
    },
    # Additional ablation levels for Config 1 to trace the full curve
    {
        "method": "summed_attention",
        "model": "base",
        "question": "inc_year",
        "tokens": 2048,
        "ablate_top_n": 10,
        "label": "Config 1 at ablate=10 (just before recovery)",
    },
    {
        "method": "summed_attention",
        "model": "base",
        "question": "inc_year",
        "tokens": 2048,
        "ablate_top_n": 20,
        "label": "Config 1 at ablate=20 (the recovery point — backup heads now ablated too)",
    },
]


def main():
    print("=" * 70)
    print("Phase 5 — Circuit Analysis Runner")
    print(f"Running {len(CONFIGS)} experiments")
    print("=" * 70)

    for i, config in enumerate(CONFIGS):
        print(f"\n{'=' * 70}")
        print(f"[{i+1}/{len(CONFIGS)}] {config['label']}")
        print(f"  {config['method']} / {config['model']} / {config['question']} / {config['tokens']} / ablate={config['ablate_top_n']}")
        print("=" * 70)

        cmd = [
            sys.executable, SCRIPT,
            "--method", config["method"],
            "--model", config["model"],
            "--question", config["question"],
            "--tokens", str(config["tokens"]),
            "--ablate-top-n", str(config["ablate_top_n"]),
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING: experiment {i+1} failed with return code {result.returncode}")
        else:
            print(f"Experiment {i+1} completed successfully.")

    print(f"\n{'=' * 70}")
    print("All experiments complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run only the 3 critical configs that show non-monotonic ablation curves.
These are the configs we need per-sample data for to run bootstrap validation.

Configs:
  1. summed_attention / base / inc_year / 2048
  2. summed_attention / instruct / inc_year / 2048
  3. qrhead / instruct / inc_year / 2048
"""
import subprocess
import sys

CONFIGS = [
    {"method": "summed_attention", "model": "base",     "question": "inc_year", "tokens": 2048},
    {"method": "summed_attention", "model": "instruct",  "question": "inc_year", "tokens": 2048},
    {"method": "qrhead",           "model": "instruct",  "question": "inc_year", "tokens": 2048},
]

def main():
    script = "run_ablation_with_samples.py"
    script_path = __file__.replace("run_critical_configs.py", script)

    for i, cfg in enumerate(CONFIGS, 1):
        print(f"\n{'='*70}")
        print(f"Config {i}/{len(CONFIGS)}: {cfg['method']} / {cfg['model']} / {cfg['question']} / {cfg['tokens']}")
        print(f"{'='*70}\n")

        cmd = [
            sys.executable, script_path,
            "--method", cfg["method"],
            "--model", cfg["model"],
            "--question", cfg["question"],
            "--tokens", str(cfg["tokens"]),
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"WARNING: Config {i} failed with return code {result.returncode}")
        else:
            print(f"Config {i} completed successfully.")

    print(f"\n{'='*70}")
    print("All critical configs complete. Run bootstrap_validation.py next.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

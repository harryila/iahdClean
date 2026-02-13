#!/usr/bin/env python3
"""
Bootstrap Validation of Non-Monotonic Ablation Curves

Reads per-sample results from Phase 5 ablation runs and performs bootstrap
resampling to compute 95% confidence intervals on accuracy at each ablation
level. Tests whether the "recovery" points in the non-monotonic curves are
statistically significant or within sampling noise.

No GPU required — this is pure resampling of existing per-sample data.

Usage:
    python bootstrap_validation.py                    # Run all available configs
    python bootstrap_validation.py --n-bootstrap 5000 # Custom bootstrap count
"""
import argparse
import json
import os
import glob
import numpy as np
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# The 3 critical configs we expect
CRITICAL_CONFIGS = [
    {"method": "summed_attention", "model": "base",     "question": "inc_year", "tokens": 2048},
    {"method": "summed_attention", "model": "instruct",  "question": "inc_year", "tokens": 2048},
    {"method": "qrhead",           "model": "instruct",  "question": "inc_year", "tokens": 2048},
]


def load_result(method, model_key, question_key, total_tokens):
    """Load a Phase 5 result file with per-sample data."""
    model_dir = "llama3_instruct" if model_key == "instruct" else "llama3_base"
    path = os.path.join(RESULTS_DIR, model_dir, question_key, f"{method}_tokens_{total_tokens}.json")

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def extract_correctness_vectors(result):
    """
    Extract per-sample correctness vectors for baseline and each ablation level.

    Returns:
        baseline_vec: list of bools (one per sample)
        top_heads: dict of {num_heads: list of bools}
        random_heads: dict of {num_heads: list of bools}
    """
    baseline_vec = [s["correct"] for s in result["baseline"]["sample_results"]]

    top_heads = {}
    for abl in result["top_heads_ablations"]:
        n = abl["num_heads"]
        top_heads[n] = [s["correct"] for s in abl["sample_results"]]

    random_heads = {}
    for abl in result.get("random_heads_ablations", []):
        n = abl["num_heads"]
        random_heads[n] = [s["correct"] for s in abl["sample_results"]]

    return baseline_vec, top_heads, random_heads


def bootstrap_accuracy(correctness_vec, n_bootstrap=10000, seed=42):
    """
    Bootstrap resample a correctness vector and return:
      - mean accuracy
      - 95% CI (lower, upper)
      - standard error
    """
    rng = np.random.RandomState(seed)
    vec = np.array(correctness_vec, dtype=float)
    n = len(vec)

    boot_accs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(vec, size=n, replace=True)
        boot_accs[i] = sample.mean()

    mean_acc = vec.mean()
    ci_lower = np.percentile(boot_accs, 2.5)
    ci_upper = np.percentile(boot_accs, 97.5)
    se = boot_accs.std()

    return {
        "mean": float(mean_acc),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "se": float(se),
        "n_samples": n,
        "n_bootstrap": n_bootstrap,
    }


def test_nonmonotonicity(top_heads_bootstrap, ablation_levels):
    """
    Test whether any "recovery" point is statistically significant.

    A recovery is when accuracy at level k is higher than at both level k-1 and k+1.
    We check if the CIs are non-overlapping.
    """
    findings = []

    for i in range(1, len(ablation_levels) - 1):
        level_prev = ablation_levels[i - 1]
        level_curr = ablation_levels[i]
        level_next = ablation_levels[i + 1]

        prev = top_heads_bootstrap[level_prev]
        curr = top_heads_bootstrap[level_curr]
        nxt = top_heads_bootstrap[level_next]

        # Is current level higher than both neighbors?
        if curr["mean"] > prev["mean"] and curr["mean"] > nxt["mean"]:
            # Check CI overlap with previous
            gap_vs_prev = curr["ci_lower"] - prev["ci_upper"]
            gap_vs_next = curr["ci_lower"] - nxt["ci_upper"]

            findings.append({
                "recovery_at": level_curr,
                "prev_level": level_prev,
                "next_level": level_next,
                "recovery_mean": curr["mean"],
                "prev_mean": prev["mean"],
                "next_mean": nxt["mean"],
                "recovery_ci": (curr["ci_lower"], curr["ci_upper"]),
                "prev_ci": (prev["ci_lower"], prev["ci_upper"]),
                "next_ci": (nxt["ci_lower"], nxt["ci_upper"]),
                "ci_gap_vs_prev": gap_vs_prev,
                "ci_gap_vs_next": gap_vs_next,
                "significant_vs_prev": gap_vs_prev > 0,
                "significant_vs_next": gap_vs_next > 0,
                "significant_both": gap_vs_prev > 0 and gap_vs_next > 0,
            })

    return findings


def analyze_config(result, n_bootstrap=10000):
    """Run full bootstrap analysis on a single config."""
    baseline_vec, top_heads, random_heads = extract_correctness_vectors(result)

    config_label = f"{result['method']} / {result['model_key']} / {result['question']} / {result['total_tokens']}"
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_label}")
    print(f"{'='*70}")
    print(f"Test samples: {len(baseline_vec)}")

    # Bootstrap baseline
    bl = bootstrap_accuracy(baseline_vec, n_bootstrap)
    print(f"\nBaseline: {bl['mean']:.1%}  95% CI: [{bl['ci_lower']:.1%}, {bl['ci_upper']:.1%}]")

    # Bootstrap each ablation level
    print(f"\n{'Heads':>6}  {'Accuracy':>10}  {'95% CI':>22}  {'Drop':>8}")
    print("-" * 55)

    ablation_levels = sorted(top_heads.keys())
    top_bootstrap = {}

    for n in ablation_levels:
        b = bootstrap_accuracy(top_heads[n], n_bootstrap)
        top_bootstrap[n] = b
        drop = bl["mean"] - b["mean"]
        print(f"{n:>6}  {b['mean']:>10.1%}  [{b['ci_lower']:.1%}, {b['ci_upper']:.1%}]  {drop:>8.1%}")

    # Random baseline
    if random_heads:
        print(f"\nRandom baseline:")
        print(f"{'Heads':>6}  {'Accuracy':>10}  {'95% CI':>22}  {'Drop':>8}")
        print("-" * 55)
        for n in sorted(random_heads.keys()):
            b = bootstrap_accuracy(random_heads[n], n_bootstrap)
            drop = bl["mean"] - b["mean"]
            print(f"{n:>6}  {b['mean']:>10.1%}  [{b['ci_lower']:.1%}, {b['ci_upper']:.1%}]  {drop:>8.1%}")

    # Test non-monotonicity
    findings = test_nonmonotonicity(top_bootstrap, ablation_levels)

    print(f"\n--- Non-Monotonicity Test ---")
    if not findings:
        print("No recovery points detected (curve is monotonic).")
    else:
        for f in findings:
            sig = "SIGNIFICANT" if f["significant_both"] else "NOT significant"
            print(f"\nRecovery at {f['recovery_at']} heads:")
            print(f"  {f['prev_level']} heads: {f['prev_mean']:.1%}  CI [{f['prev_ci'][0]:.1%}, {f['prev_ci'][1]:.1%}]")
            print(f"  {f['recovery_at']} heads: {f['recovery_mean']:.1%}  CI [{f['recovery_ci'][0]:.1%}, {f['recovery_ci'][1]:.1%}]")
            print(f"  {f['next_level']} heads: {f['next_mean']:.1%}  CI [{f['next_ci'][0]:.1%}, {f['next_ci'][1]:.1%}]")
            print(f"  CI gap vs prev: {f['ci_gap_vs_prev']:.1%}  {'✓' if f['significant_vs_prev'] else '✗'}")
            print(f"  CI gap vs next: {f['ci_gap_vs_next']:.1%}  {'✓' if f['significant_vs_next'] else '✗'}")
            print(f"  → {sig}")

    return {
        "config": config_label,
        "baseline": bl,
        "top_heads_bootstrap": {str(k): v for k, v in top_bootstrap.items()},
        "nonmonotonicity_findings": findings,
    }


def main():
    parser = argparse.ArgumentParser(description="Bootstrap validation of non-monotonic ablation curves")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Number of bootstrap samples")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 5: Bootstrap Validation of Non-Monotonic Ablation Curves")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print("=" * 70)

    all_results = []

    for cfg in CRITICAL_CONFIGS:
        result = load_result(cfg["method"], cfg["model"], cfg["question"], cfg["tokens"])
        if result is None:
            print(f"\nSkipping {cfg['method']}/{cfg['model']}/{cfg['question']}/{cfg['tokens']} — not found.")
            print(f"  Run `python run_critical_configs.py` first.")
            continue

        # Verify per-sample data exists
        if "sample_results" not in result.get("baseline", {}):
            print(f"\nSkipping {cfg['method']}/{cfg['model']}/{cfg['question']}/{cfg['tokens']} — no per-sample data.")
            print(f"  This file was generated by Phase 3, not Phase 5. Re-run with run_critical_configs.py.")
            continue

        analysis = analyze_config(result, n_bootstrap=args.n_bootstrap)
        all_results.append(analysis)

    # Save combined results
    if all_results:
        output_path = os.path.join(os.path.dirname(__file__), "bootstrap_results.json")
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{'='*70}")
        print(f"Saved bootstrap results: {output_path}")

        # Final verdict
        print(f"\n{'='*70}")
        print("VERDICT SUMMARY")
        print(f"{'='*70}")
        any_significant = False
        for r in all_results:
            print(f"\n{r['config']}:")
            if not r["nonmonotonicity_findings"]:
                print("  No recovery points (monotonic curve)")
            else:
                for f in r["nonmonotonicity_findings"]:
                    if f["significant_both"]:
                        print(f"  ✓ SIGNIFICANT recovery at {f['recovery_at']} heads")
                        any_significant = True
                    else:
                        print(f"  ✗ Recovery at {f['recovery_at']} heads NOT significant (CIs overlap)")

        print(f"\n{'='*70}")
        if any_significant:
            print("→ NON-MONOTONICITY IS REAL in at least one config.")
            print("  Proceed with Phase C (circuit analysis) and backup circuit paper framing.")
        else:
            print("→ NON-MONOTONICITY IS NOT STATISTICALLY SIGNIFICANT.")
            print("  Pivot to context-length circuit replacement paper framing.")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()

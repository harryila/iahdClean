#!/usr/bin/env python3
"""
Compute overlap between QRHead and Wu24 Retrieval Head methods,
grouped by question (aggregated across all context lengths).

For each (question, model):
  - Load head rankings from both methods across all 4 context lengths
  - Aggregate by average rank across lengths (heads not in top-50 get rank 100)
  - Take top-N from each method's aggregated ranking
  - Compute intersection (the "consensus retrieval heads")

Outputs:
  - Console summary
  - overlap figure: qr_wu24_overlap.png
  - overlap head lists as JSON: qr_wu24_overlap_heads.json
"""

import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_DIR = os.path.join(SCRIPT_DIR, "..", "..", "phase2")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
METHODS = ["retrieval_head_wu24", "qrhead"]
METHOD_LABELS = {"retrieval_head_wu24": "Wu24", "qrhead": "QRHead"}
MODELS = ["llama3_instruct", "llama3_base"]
MODEL_LABELS = {"llama3_instruct": "Instruct", "llama3_base": "Base"}
QUESTIONS = ["inc_state", "inc_year", "employee_count", "hq_state"]
QUESTION_LABELS = {
    "inc_state": "Inc. State",
    "inc_year": "Inc. Year",
    "employee_count": "Employee Count",
    "hq_state": "HQ State",
}
TOKEN_LENGTHS = [2048, 4096, 6144, 8192]
TOP_N = 50          # top-N heads per method per (question, length)
AGG_TOP_N = 50      # top-N from the aggregated ranking
MISSING_RANK = 100  # rank assigned to heads outside top-N


def load_head_rankings(method, model, question, tokens, top_n=TOP_N):
    """Load top-N head rankings from a phase2 result file.
    Returns list of (head_str, rank) tuples, e.g. [("L20H14", 1), ...]
    """
    path = os.path.join(
        PHASE2_DIR, method, "results", model, question, f"tokens_{tokens}.json"
    )
    if not os.path.exists(path):
        print(f"  WARNING: missing {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    rankings = data.get("head_rankings", [])[:top_n]
    return [(item["head"], item["rank"]) for item in rankings]


def aggregate_across_lengths(method, model, question):
    """For a given (method, model, question), aggregate head rankings
    across all context lengths by average rank.
    Returns sorted list of (head_str, avg_rank).
    """
    # Collect ranks per head across all lengths
    head_ranks = defaultdict(list)

    for tokens in TOKEN_LENGTHS:
        rankings = load_head_rankings(method, model, question, tokens)
        seen = set()
        for head_str, rank in rankings:
            head_ranks[head_str].append(rank)
            seen.add(head_str)
        # Heads NOT in this length's top-N get MISSING_RANK
        for head_str in head_ranks:
            if head_str not in seen:
                head_ranks[head_str].append(MISSING_RANK)

    # Average rank across lengths
    avg_ranks = []
    for head_str, ranks in head_ranks.items():
        # Pad missing lengths (head only appeared in some)
        while len(ranks) < len(TOKEN_LENGTHS):
            ranks.append(MISSING_RANK)
        avg_ranks.append((head_str, np.mean(ranks)))

    avg_ranks.sort(key=lambda x: x[1])
    return avg_ranks


def compute_all_overlaps():
    """Compute QRHead ∩ Wu24 for every (question, model)."""
    results = {}

    for model in MODELS:
        for question in QUESTIONS:
            key = (model, question)

            # Aggregate rankings across context lengths
            wu24_agg = aggregate_across_lengths("retrieval_head_wu24", model, question)
            qr_agg = aggregate_across_lengths("qrhead", model, question)

            # Take top AGG_TOP_N from each
            wu24_top = set(h for h, _ in wu24_agg[:AGG_TOP_N])
            qr_top = set(h for h, _ in qr_agg[:AGG_TOP_N])

            overlap = wu24_top & qr_top
            wu24_only = wu24_top - qr_top
            qr_only = qr_top - wu24_top

            jaccard = len(overlap) / len(wu24_top | qr_top) if (wu24_top | qr_top) else 0

            # Build ranked overlap list (sort by average of both methods' avg ranks)
            wu24_rank_map = {h: r for h, r in wu24_agg}
            qr_rank_map = {h: r for h, r in qr_agg}
            overlap_ranked = sorted(
                overlap,
                key=lambda h: (wu24_rank_map.get(h, 999) + qr_rank_map.get(h, 999)) / 2,
            )

            results[key] = {
                "wu24_top": wu24_top,
                "qr_top": qr_top,
                "overlap": overlap,
                "overlap_ranked": overlap_ranked,
                "wu24_only": wu24_only,
                "qr_only": qr_only,
                "jaccard": jaccard,
                "n_overlap": len(overlap),
                "wu24_rank_map": wu24_rank_map,
                "qr_rank_map": qr_rank_map,
            }

    return results


def print_summary(results):
    """Print a nice summary table."""
    print("\n" + "=" * 70)
    print("QRHead ∩ Wu24 Overlap (aggregated across context lengths)")
    print("=" * 70)

    for model in MODELS:
        print(f"\n── {MODEL_LABELS[model]} ──")
        print(f"{'Question':<18} {'Overlap':>8} {'Jaccard':>9}   Top overlap heads")
        print("-" * 70)
        for question in QUESTIONS:
            r = results[(model, question)]
            top_heads = ", ".join(r["overlap_ranked"][:8])
            print(
                f"{QUESTION_LABELS[question]:<18} "
                f"{r['n_overlap']:>5}/{AGG_TOP_N}  "
                f"{r['jaccard']:>8.1%}   "
                f"{top_heads}"
            )


def save_overlap_json(results):
    """Save overlap heads to JSON for downstream ablation."""
    output = {}
    for (model, question), r in results.items():
        mkey = MODEL_LABELS[model].lower()
        if mkey not in output:
            output[mkey] = {}
        output[mkey][question] = {
            "overlap_heads": r["overlap_ranked"],
            "n_overlap": r["n_overlap"],
            "jaccard": round(r["jaccard"], 4),
            "wu24_only_heads": sorted(r["wu24_only"]),
            "qr_only_heads": sorted(r["qr_only"]),
            "head_details": [
                {
                    "head": h,
                    "wu24_avg_rank": round(r["wu24_rank_map"].get(h, -1), 1),
                    "qr_avg_rank": round(r["qr_rank_map"].get(h, -1), 1),
                }
                for h in r["overlap_ranked"]
            ],
        }

    out_path = os.path.join(FIGURES_DIR, "qr_wu24_overlap_heads.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved overlap data: {out_path}")


def generate_figure(results):
    """Simple bar chart: overlap size per question, grouped by model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    bar_width = 0.55
    colors = {"overlap": "#2ecc71", "wu24_only": "#3498db", "qr_only": "#e74c3c"}

    for ax, model in zip(axes, MODELS):
        questions = QUESTIONS
        n_overlap = [results[(model, q)]["n_overlap"] for q in questions]
        n_wu24_only = [len(results[(model, q)]["wu24_only"]) for q in questions]
        n_qr_only = [len(results[(model, q)]["qr_only"]) for q in questions]

        x = np.arange(len(questions))

        # Stacked bar: overlap at bottom, then wu24-only, then qr-only
        bars1 = ax.bar(x, n_overlap, bar_width, label="Overlap (QR ∩ Wu24)",
                       color=colors["overlap"], edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x, n_wu24_only, bar_width, bottom=n_overlap,
                       label="Wu24 only", color=colors["wu24_only"],
                       edgecolor="white", linewidth=0.5, alpha=0.7)
        bars3 = ax.bar(x, n_qr_only, bar_width,
                       bottom=[a + b for a, b in zip(n_overlap, n_wu24_only)],
                       label="QRHead only", color=colors["qr_only"],
                       edgecolor="white", linewidth=0.5, alpha=0.7)

        # Annotate overlap count on bars
        for i, (bar, n) in enumerate(zip(bars1, n_overlap)):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                        str(n), ha="center", va="center", fontsize=12,
                        fontweight="bold", color="white")

        ax.set_xticks(x)
        ax.set_xticklabels([QUESTION_LABELS[q] for q in questions], fontsize=10)
        ax.set_title(f"{MODEL_LABELS[model]}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Number of heads" if model == MODELS[0] else "")
        ax.set_ylim(0, AGG_TOP_N * 2 + 5)

        # Add Jaccard annotations above bars
        for i, q in enumerate(questions):
            total = n_overlap[i] + n_wu24_only[i] + n_qr_only[i]
            j = results[(model, q)]["jaccard"]
            ax.text(i, total + 1.5, f"J={j:.0%}", ha="center", va="bottom",
                    fontsize=9, color="#555")

    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("QRHead ∩ Wu24 Overlap by Question\n(top-50 heads, aggregated across context lengths)",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "qr_wu24_overlap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def main():
    print("Computing QRHead ∩ Wu24 overlap across questions...")
    results = compute_all_overlaps()
    print_summary(results)
    save_overlap_json(results)
    generate_figure(results)
    print("\nDone.")


if __name__ == "__main__":
    main()

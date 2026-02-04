"""
Phase 1: Data Preparation

This script:
1. Loads the ground truth CSV (edgar_gt_verified_slim.csv)
2. Reports valid sample counts per GT category
3. Creates 80/20 train/test split
4. Saves splits as JSON files

Usage:
    python prepare_data.py
"""

import pandas as pd
import json
import os
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

GT_PATH = "../../edgar_gt_verified_slim.csv"  # Relative to phase1/
OUTPUT_DIR = "."  # Save in phase1/
RANDOM_STATE = 42
TEST_SIZE = 0.2

# GT column mapping: deliverable question -> CSV column name
GT_COLUMNS = {
    "inc_state": "original_Inc_state_truth",
    "inc_year": "original_Inc_year_truth",
    "employee_count": "employee_count_truth",
    "hq_state": "headquarters_state_truth",
}

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Phase 1: Data Preparation")
    print("=" * 60)
    
    # Step 1: Load ground truth
    print(f"\n[Step 1] Loading ground truth from: {GT_PATH}")
    
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"Ground truth file not found: {GT_PATH}")
    
    gt_df = pd.read_csv(GT_PATH)
    print(f"  Loaded {len(gt_df)} total samples")
    print(f"  Columns: {list(gt_df.columns)}")
    
    # Step 2: Report valid samples per GT category
    print(f"\n[Step 2] Valid samples per GT category:")
    print("-" * 50)
    
    valid_counts = {}
    for question, col in GT_COLUMNS.items():
        # Count non-null and non-"NULL" string values
        valid_mask = gt_df[col].notna() & (gt_df[col].astype(str) != "NULL")
        valid_count = valid_mask.sum()
        valid_counts[question] = valid_count
        pct = 100 * valid_count / len(gt_df)
        print(f"  {question:20} {valid_count:>4} / {len(gt_df)} ({pct:.0f}%)")
    
    # Step 3: Create train/test split
    print(f"\n[Step 3] Creating train/test split")
    print(f"  Test size: {TEST_SIZE} ({int(TEST_SIZE*100)}%)")
    print(f"  Random state: {RANDOM_STATE}")
    
    # Manual train/test split (avoiding sklearn dependency issues)
    random.seed(RANDOM_STATE)
    all_indices = list(range(len(gt_df)))
    random.shuffle(all_indices)
    
    test_count = int(len(gt_df) * TEST_SIZE)
    test_indices = all_indices[:test_count]
    train_indices = all_indices[test_count:]
    
    train_df = gt_df.iloc[train_indices].copy()
    test_df = gt_df.iloc[test_indices].copy()
    
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Step 4: Verify split validity per category
    print(f"\n[Step 4] Verifying split validity per category:")
    print("-" * 50)
    
    for question, col in GT_COLUMNS.items():
        train_valid = train_df[col].notna() & (train_df[col].astype(str) != "NULL")
        test_valid = test_df[col].notna() & (test_df[col].astype(str) != "NULL")
        train_count = train_valid.sum()
        test_count = test_valid.sum()
        print(f"  {question:20} Train: {train_count:>4}  Test: {test_count:>3}  Total: {train_count + test_count}")
    
    # Step 5: Save splits as JSON
    print(f"\n[Step 5] Saving splits to JSON files")
    
    train_samples = train_df["filename"].tolist()
    test_samples = test_df["filename"].tolist()
    
    train_path = os.path.join(OUTPUT_DIR, "train_samples.json")
    test_path = os.path.join(OUTPUT_DIR, "test_samples.json")
    
    with open(train_path, "w") as f:
        json.dump(train_samples, f, indent=2)
    print(f"  Saved: {train_path} ({len(train_samples)} samples)")
    
    with open(test_path, "w") as f:
        json.dump(test_samples, f, indent=2)
    print(f"  Saved: {test_path} ({len(test_samples)} samples)")
    
    # Step 6: Also save full GT info for reference
    print(f"\n[Step 6] Saving GT summary for reference")
    
    # Convert numpy int64 to Python int for JSON serialization
    valid_counts_json = {k: int(v) for k, v in valid_counts.items()}
    
    summary = {
        "gt_source": GT_PATH,
        "total_samples": int(len(gt_df)),
        "train_count": int(len(train_df)),
        "test_count": int(len(test_df)),
        "random_state": RANDOM_STATE,
        "valid_counts_per_category": valid_counts_json,
        "gt_columns": GT_COLUMNS,
    }
    
    summary_path = os.path.join(OUTPUT_DIR, "data_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")
    
    print("\n" + "=" * 60)
    print("Phase 1 Complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - train_samples.json ({len(train_samples)} samples)")
    print(f"  - test_samples.json ({len(test_samples)} samples)")
    print(f"  - data_summary.json (metadata)")


if __name__ == "__main__":
    main()

# Phase 1: Data Preparation

## Purpose

Create reproducible train/test splits from the ground truth data for the head analysis experiments.

- **Training set (80%):** Used in Phase 2 to identify important attention heads
- **Test set (20%):** Used in Phase 3 to evaluate ablation effects on unseen data

This separation prevents overfitting—heads are identified on one set of samples and evaluated on a completely different set.

## Data Source

**File:** `../../edgar_gt_verified_slim.csv`

This CSV contains 250 SEC 10-K filings with verified ground truth for 4 question categories:

| Question | CSV Column | Type | Valid Samples |
|----------|------------|------|---------------|
| Incorporation State | `original_Inc_state_truth` | Categorical | 164 / 250 (66%) |
| Incorporation Year | `original_Inc_year_truth` | Numerical | 161 / 250 (64%) |
| Employee Count | `employee_count_truth` | Numerical | 188 / 250 (75%) |
| HQ State | `headquarters_state_truth` | Categorical | 133 / 250 (53%) |

Note: Not all samples have valid ground truth for every category (some contain NULL values).

## What the Script Does

`prepare_data.py` performs the following steps:

1. **Load GT CSV** - Reads `edgar_gt_verified_slim.csv`
2. **Report valid counts** - Counts non-NULL samples per GT category
3. **Create 80/20 split** - Shuffles indices with random seed 42, then splits
4. **Verify split validity** - Confirms each category has sufficient train/test samples
5. **Save outputs** - Writes JSON files for downstream phases

### Implementation Details

```python
# From prepare_data.py
import random
random.seed(42)  # Ensures reproducibility

# Manual shuffle-based split
indices = list(range(len(df)))
random.shuffle(indices)
train_indices = indices[:int(0.8 * len(indices))]
test_indices = indices[int(0.8 * len(indices)):]
```

**Why manual split?** Originally planned to use `sklearn.model_selection.train_test_split()`, but encountered a numpy version conflict (sklearn compiled against numpy 1.x, environment has numpy 2.x). The manual implementation produces equivalent results.

## Usage

```bash
cd phase1/
python prepare_data.py
```

## Output Files

| File | Description |
|------|-------------|
| `train_samples.json` | List of 200 filenames for head identification (Phase 2) |
| `test_samples.json` | List of 50 filenames for ablation evaluation (Phase 3) |
| `data_summary.json` | Metadata: split sizes, valid counts, column mappings |

## Results

### Split Summary

| Set | Samples | Percentage |
|-----|---------|------------|
| Train | 200 | 80% |
| Test | 50 | 20% |

### Valid Samples per Category After Split

| Question | Train | Test | Total |
|----------|-------|------|-------|
| inc_state | 131 | 33 | 164 |
| inc_year | 128 | 33 | 161 |
| employee_count | 152 | 36 | 188 |
| hq_state | 106 | 27 | 133 |

### Validation Checks

- ✅ **No overlap:** 0 files appear in both train and test
- ✅ **Complete coverage:** All 250 original files accounted for
- ✅ **Sufficient samples:** Each question type has enough samples in both splits

## How Downstream Phases Use This Data

### Phase 2 (Head Identification)
Loads `train_samples.json` to identify which attention heads are important for each question type. Uses the `data_summary.json` to map question names to GT columns.

```python
# Example from phase2/*/run_detection.py
with open("../phase1/train_samples.json") as f:
    train_files = json.load(f)
```

### Phase 3 (Ablation Study)  
Loads `test_samples.json` to evaluate the causal importance of identified heads.

```python
# Example from phase3/*/run_ablation.py
with open("../phase1/test_samples.json") as f:
    test_files = json.load(f)
```

## Reproducibility

Running `prepare_data.py` multiple times will always produce identical outputs because:
1. Random seed is fixed at 42
2. Shuffle algorithm is deterministic given the seed
3. No external randomness is introduced

## Challenges Encountered

### sklearn Import Error

**Problem:** Initial implementation used `sklearn.model_selection.train_test_split()`, but this failed with:
```
ImportError: numpy.core.multiarray failed to import
```

**Cause:** sklearn was compiled against numpy 1.x, but environment has numpy 2.x.

**Solution:** Implemented equivalent manual shuffle-based split using Python's built-in `random` module, which has no numpy dependency.

```python
# Instead of:
# from sklearn.model_selection import train_test_split
# train, test = train_test_split(files, test_size=0.2, random_state=42)

# We use:
import random
random.seed(42)
indices = list(range(len(files)))
random.shuffle(indices)
split_point = int(0.8 * len(indices))
train_indices = indices[:split_point]
test_indices = indices[split_point:]
```

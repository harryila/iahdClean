"""
Cache Section 1 data from EDGAR corpus for all samples.

This script downloads Section 1 content for all 250 GT samples
and saves it locally, so we don't have to stream through the 
entire corpus for each experiment.

Run this ONCE before running Phase 2 experiments.

Usage:
    python cache_section1_data.py
"""

import os
import json
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GT_PATH = os.path.join(SCRIPT_DIR, "..", "..", "edgar_gt_verified_slim.csv")
CACHE_PATH = os.path.join(SCRIPT_DIR, "section1_cache.json")

def main():
    print("=" * 60)
    print("Caching Section 1 Data from EDGAR Corpus")
    print("=" * 60)
    
    # Load all filenames from GT
    gt_df = pd.read_csv(GT_PATH)
    all_filenames = set(gt_df["filename"].tolist())
    print(f"\nTotal GT samples: {len(all_filenames)}")
    
    # Check if cache exists
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            existing_cache = json.load(f)
        cached_files = set(existing_cache.keys())
        missing = all_filenames - cached_files
        print(f"Existing cache: {len(cached_files)} files")
        print(f"Missing: {len(missing)} files")
        
        if len(missing) == 0:
            print("Cache is complete! No need to download.")
            return
        
        target_files = missing
        section1_cache = existing_cache
    else:
        print("No existing cache, downloading all files...")
        target_files = all_filenames
        section1_cache = {}
    
    print(f"\nStreaming EDGAR corpus to find {len(target_files)} files...")
    print("(This may take 10-20 minutes the first time)")
    
    # Stream through EDGAR
    dataset = load_dataset(
        "c3po-ai/edgar-corpus",
        "full",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    found_count = 0
    for item in tqdm(dataset, desc="Streaming EDGAR"):
        filename = item.get("filename", "")
        
        if filename in target_files:
            section_1 = item.get("section_1", "")
            if section_1 and len(section_1) > 100:
                section1_cache[filename] = section_1
                found_count += 1
                
                # Save periodically
                if found_count % 10 == 0:
                    with open(CACHE_PATH, "w") as f:
                        json.dump(section1_cache, f)
                    print(f"\n  Saved checkpoint: {len(section1_cache)} files cached")
        
        # Early exit if we have all
        if len(section1_cache) >= len(all_filenames):
            break
    
    # Final save
    with open(CACHE_PATH, "w") as f:
        json.dump(section1_cache, f)
    
    print(f"\n" + "=" * 60)
    print(f"Caching Complete!")
    print(f"=" * 60)
    print(f"Total cached: {len(section1_cache)} files")
    print(f"Saved to: {CACHE_PATH}")
    
    # Report any missing
    still_missing = all_filenames - set(section1_cache.keys())
    if still_missing:
        print(f"\nWarning: {len(still_missing)} files not found in EDGAR:")
        for f in list(still_missing)[:5]:
            print(f"  - {f}")

if __name__ == "__main__":
    main()

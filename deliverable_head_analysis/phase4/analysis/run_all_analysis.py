#!/usr/bin/env python3
"""
Phase 4: Run All Analysis

Master script that runs all Phase 4 analysis and generates all outputs.
"""

import subprocess
import sys
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(__file__)

def run_script(script_name):
    """Run a Python script and report status."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        print(f"[OK] {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"[ERROR] {script_name} failed: {e}")
        return False


def main():
    print("="*70)
    print("PHASE 4: COMPLETE ANALYSIS PIPELINE")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*70)
    
    scripts = [
        "generate_figure8.py",
        "research_questions.py",
        "generate_findings.py",
    ]
    
    results = {}
    for script in scripts:
        results[script] = run_script(script)
    
    print("\n" + "="*70)
    print("PHASE 4 ANALYSIS COMPLETE")
    print("="*70)
    print(f"Finished: {datetime.now().isoformat()}")
    print("\nResults:")
    for script, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {script}: {status}")
    
    # Summary of outputs
    figures_dir = os.path.join(SCRIPT_DIR, "..", "figures")
    findings_path = os.path.join(SCRIPT_DIR, "..", "FINDINGS.md")
    
    print("\nOutputs generated:")
    if os.path.exists(figures_dir):
        figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        print(f"  Figures: {len(figures)} PNG files in figures/")
        for f in sorted(figures):
            print(f"    - {f}")
    
    if os.path.exists(findings_path):
        print(f"  Report: FINDINGS.md")


if __name__ == "__main__":
    main()

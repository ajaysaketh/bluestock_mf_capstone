"""
run_pipeline.py  —  Bluestock MF Capstone
Master execution script — runs the full pipeline end to end.

Usage:
    python run_pipeline.py

Steps:
    1. Data Ingestion  (scripts/data_ingestion.py)
    2. ETL Pipeline    (scripts/etl_pipeline.py)
    3. EDA Analysis    (scripts/eda_analysis.py)
    4. Performance     (scripts/performance_analytics.py)
    5. Advanced        (scripts/advanced_analytics.py)
"""

import os
import sys
import time
import subprocess

ROOT   = os.getcwd()
PYTHON = sys.executable
SEP    = "=" * 60

STEPS = [
    ("Data Ingestion",       "scripts/data_ingestion.py"),
    ("ETL Pipeline",         "scripts/etl_pipeline.py"),
    ("EDA Analysis",         "scripts/eda_analysis.py"),
    ("Performance Analytics","scripts/performance_analytics.py"),
    ("Advanced Analytics",   "scripts/advanced_analytics.py"),
]

def run_step(name, script_path):
    full_path = os.path.join(ROOT, script_path)
    if not os.path.exists(full_path):
        print(f"  SKIP  {name} — {script_path} not found")
        return True
    print(f"\n  Running: {name}")
    print(f"  Script : {script_path}")
    t0     = time.time()
    result = subprocess.run([PYTHON, full_path], capture_output=False)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  Done in {elapsed:.1f}s")
        return True
    else:
        print(f"  FAILED (exit code {result.returncode})")
        return False

if __name__ == "__main__":
    print(SEP)
    print("  BLUESTOCK MF CAPSTONE — Master Pipeline")
    print(SEP)

    passed = 0
    for name, script in STEPS:
        ok = run_step(name, script)
        if ok:
            passed += 1

    print(f"\n{SEP}")
    print(f"  Pipeline complete: {passed}/{len(STEPS)} steps passed")
    print(SEP)
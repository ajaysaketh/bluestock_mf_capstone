"""
data_ingestion.py  —  Bluestock MF Capstone | Day 1
Loads all 10 raw CSVs, prints diagnostics, detects anomalies,
validates AMFI codes, and writes reports/data_quality_report.txt
"""

import os
import pandas as pd

# ── Paths (works on Windows + Mac + Linux) ─────────────────────────────────
ROOT     = os.getcwd()                          # run from project root
RAW_DIR  = os.path.join(ROOT, "data", "raw")
REP_DIR  = os.path.join(ROOT, "reports")
os.makedirs(REP_DIR, exist_ok=True)

# ── Dataset registry ────────────────────────────────────────────────────────
DATASETS = [
    ("01_fund_master",           "01_fund_master.csv"),
    ("02_nav_history",           "02_nav_history.csv"),
    ("03_aum_by_fund_house",     "03_aum_by_fund_house.csv"),
    ("04_monthly_sip_inflows",   "04_monthly_sip_inflows.csv"),
    ("05_category_inflows",      "05_category_inflows.csv"),
    ("06_industry_folio_count",  "06_industry_folio_count.csv"),
    ("07_scheme_performance",    "07_scheme_performance.csv"),
    ("08_investor_transactions", "08_investor_transactions.csv"),
    ("09_portfolio_holdings",    "09_portfolio_holdings.csv"),
    ("10_benchmark_indices",     "10_benchmark_indices.csv"),
]

SEP = "=" * 65
report = ["BLUESTOCK MF CAPSTONE — DATA QUALITY REPORT", "Day 1: Data Ingestion", SEP]

print(SEP)
print("  BLUESTOCK MF CAPSTONE | Day 1 — Data Ingestion")
print(SEP)

loaded = {}

for name, filename in DATASETS:
    path = os.path.join(RAW_DIR, filename)
    print(f"\n>>> {filename}")

    if not os.path.exists(path):
        msg = f"  FILE NOT FOUND: {path}"
        print(msg)
        report.append(f"\n[MISSING] {filename}")
        continue

    df = pd.read_csv(path, low_memory=False)
    loaded[name] = df

    print(f"  Shape   : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Dtypes  :\n{df.dtypes.to_string()}")
    print(f"  Head(3) :\n{df.head(3).to_string()}")

    # Null check
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        print(f"  Nulls   : {nulls.to_dict()}")
    else:
        print("  Nulls   : None")

    # Duplicate check
    dups = df.duplicated().sum()
    print(f"  Duplicates: {dups}")

    report.append(f"\n{filename} | {df.shape[0]:,} rows x {df.shape[1]} cols")
    report.append(f"  Nulls: {nulls.to_dict() if not nulls.empty else 'None'}")
    report.append(f"  Duplicates: {dups}")

# ── Fund Master exploration ─────────────────────────────────────────────────
if "01_fund_master" in loaded:
    df = loaded["01_fund_master"]
    print(f"\n{SEP}")
    print("  FUND MASTER — Exploration (Task 6)")
    print(SEP)
    for col in ["fund_house", "category", "sub_category", "risk_category"]:
        if col in df.columns:
            print(f"\n  {col.upper()}:\n{df[col].value_counts().to_string()}")

# ── AMFI Code Validation ────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  AMFI CODE VALIDATION (Task 7)")
print(SEP)

if "01_fund_master" in loaded and "02_nav_history" in loaded:
    master_codes = set(loaded["01_fund_master"]["amfi_code"].astype(str).unique())
    nav_codes    = set(loaded["02_nav_history"]["amfi_code"].astype(str).unique())
    missing = master_codes - nav_codes
    extra   = nav_codes - master_codes
    if missing:
        print(f"  In fund_master but NOT in nav_history : {sorted(missing)}")
    if extra:
        print(f"  In nav_history but NOT in fund_master : {sorted(extra)}")
    if not missing and not extra:
        print("  All AMFI codes match perfectly")
    report.append(f"\nAMFI Validation: missing={sorted(missing)}, extra={sorted(extra)}")
else:
    print("  Cannot validate — fund_master or nav_history missing")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  SUMMARY: {len(loaded)}/{len(DATASETS)} datasets loaded")
total = sum(len(d) for d in loaded.values())
print(f"  Total rows: {total:,}")
print(SEP)

# ── Write report ─────────────────────────────────────────────────────────────
rep_path = os.path.join(REP_DIR, "data_quality_report.txt")
with open(rep_path, "w") as f:
    f.write("\n".join(report))
print(f"\n  Report saved: {rep_path}")
print("\n  Day 1 Task 3, 6, 7 — COMPLETE\n")

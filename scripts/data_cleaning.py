"""
data_cleaning.py  —  Day 2 Tasks 1, 2, 3
Cleans nav_history, investor_transactions, scheme_performance
Run: python scripts/data_cleaning.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW  = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# ── Task 1: Clean NAV History ─────────────────────────────────────────────────
print("=" * 55)
print("  Task 1: Cleaning NAV History")
print("=" * 55)

nav = pd.read_csv(RAW / "02_nav_history.csv")
print(f"  Raw shape     : {nav.shape}")
print(f"  Columns       : {nav.columns.tolist()}")

# Parse date
nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
nav["amfi_code"] = nav["amfi_code"].astype(str)

# Sort
nav = nav.sort_values(["amfi_code", "date"]).reset_index(drop=True)

# Remove duplicates
before = len(nav)
nav = nav.drop_duplicates(subset=["amfi_code", "date"])
print(f"  Duplicates removed : {before - len(nav)}")

# Remove invalid NAV
nav = nav[nav["nav"] > 0]

# Forward-fill missing dates (weekends/holidays)
frames = []
full_dates = pd.bdate_range(nav["date"].min(), nav["date"].max())
for code, grp in nav.groupby("amfi_code"):
    grp = grp.set_index("date").reindex(full_dates)
    grp["nav"] = grp["nav"].ffill()
    grp["amfi_code"] = code
    frames.append(grp.reset_index().rename(columns={"index": "date"}))

clean_nav = pd.concat(frames, ignore_index=True)
clean_nav["date"] = clean_nav["date"].dt.strftime("%Y-%m-%d")
clean_nav.to_csv(PROC / "clean_nav.csv", index=False)
print(f"  ✅ clean_nav.csv saved : {clean_nav.shape}")

# ── Task 2: Clean Investor Transactions ───────────────────────────────────────
print("\n" + "=" * 55)
print("  Task 2: Cleaning Investor Transactions")
print("=" * 55)

tx = pd.read_csv(RAW / "08_investor_transactions.csv")
print(f"  Raw shape : {tx.shape}")

# Fix date
tx["transaction_date"] = pd.to_datetime(tx["transaction_date"], errors="coerce")
tx["transaction_date"] = tx["transaction_date"].dt.strftime("%Y-%m-%d")

# Standardise transaction type
tx["transaction_type"] = tx["transaction_type"].str.strip().str.title()
valid_types = ["Sip", "Lumpsum", "Redemption"]
before = len(tx)
tx = tx[tx["transaction_type"].isin(valid_types)]
print(f"  Invalid types removed : {before - len(tx)}")

# Validate amount > 0
tx = tx[tx["amount_inr"] > 0]

# Check KYC status
tx = tx[tx["kyc_status"].isin(["Verified", "Pending"])]

# Remove duplicates
tx = tx.drop_duplicates()

tx.to_csv(PROC / "clean_transactions.csv", index=False)
print(f"  ✅ clean_transactions.csv saved : {tx.shape}")

# ── Task 3: Clean Scheme Performance ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  Task 3: Cleaning Scheme Performance")
print("=" * 55)

perf = pd.read_csv(RAW / "07_scheme_performance.csv")
print(f"  Raw shape : {perf.shape}")

# Validate numeric columns
num_cols = ["return_1yr_pct", "return_3yr_pct", "return_5yr_pct",
            "sharpe_ratio", "sortino_ratio", "beta", "alpha",
            "std_dev_ann_pct", "max_drawdown_pct"]
for col in num_cols:
    if col in perf.columns:
        perf[col] = pd.to_numeric(perf[col], errors="coerce")

# Flag negative Sharpe ratios
if "sharpe_ratio" in perf.columns:
    neg_sharpe = (perf["sharpe_ratio"] < 0).sum()
    perf["negative_sharpe_flag"] = perf["sharpe_ratio"] < 0
    print(f"  Negative Sharpe flags : {neg_sharpe}")

# Remove duplicates
perf = perf.drop_duplicates()

perf.to_csv(PROC / "clean_performance.csv", index=False)
print(f"  ✅ clean_performance.csv saved : {perf.shape}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  DAY 2 CLEANING COMPLETE")
print("=" * 55)
print(f"  clean_nav.csv          : {clean_nav.shape[0]:,} rows")
print(f"  clean_transactions.csv : {tx.shape[0]:,} rows")
print(f"  clean_performance.csv  : {perf.shape[0]:,} rows")
print("\n  Files saved in data/processed/")

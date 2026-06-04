"""
etl_pipeline.py  —  Bluestock MF Capstone | Day 2
Master ETL: Cleans all 10 datasets and loads them into SQLite DB.

Tasks covered:
  Task 1 — Clean nav_history (tdate, tnav columns)
  Task 2 — Clean investor_transactions
  Task 3 — Clean scheme_performance
  Task 5 — Load all cleaned data into bluestock_mf.db

Usage:
    python scripts/etl_pipeline.py
"""

import os
import re
import sqlite3
import pandas as pd
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT      = os.getcwd()
RAW_DIR   = os.path.join(ROOT, "data", "raw")
PROC_DIR  = os.path.join(ROOT, "data", "processed")
DB_DIR    = os.path.join(ROOT, "data", "db")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(DB_DIR,   exist_ok=True)

DB_PATH   = os.path.join(DB_DIR, "bluestock_mf.db")
SEP       = "=" * 65


def load_raw(filename):
    path = os.path.join(RAW_DIR, filename)
    df   = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {filename}: {df.shape}")
    return df


def save_clean(df, filename):
    path = os.path.join(PROC_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved  {filename}: {df.shape}")


# ════════════════════════════════════════════════════════════════
# TASK 1 — Clean nav_history
# ════════════════════════════════════════════════════════════════
def clean_nav_history():
    print(f"\n{SEP}\n  TASK 1 — Cleaning nav_history\n{SEP}")
    df = load_raw("02_nav_history.csv")

    # Rename columns to standard names
    df.rename(columns={"tdate": "date", "tnav": "nav"}, inplace=True)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    # Sort
    df.sort_values(["amfi_code", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Forward-fill missing NAV per fund (holidays/weekends)
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="B")
    chunks = []
    for code, grp in df.groupby("amfi_code"):
        grp = grp.set_index("date").reindex(full_dates)
        grp["amfi_code"] = code
        grp["nav"] = pd.to_numeric(grp["nav"], errors="coerce")
        grp["nav"] = grp["nav"].ffill()
        grp.index.name = "date"
        grp = grp.reset_index()
        chunks.append(grp)
    df = pd.concat(chunks, ignore_index=True)

    # Remove duplicates
    df.drop_duplicates(subset=["amfi_code", "date"], inplace=True)

    # Validate NAV > 0
    before = len(df)
    df = df[df["nav"] > 0]
    print(f"  Removed {before - len(df)} rows with NAV <= 0")

    # Compute daily return
    df.sort_values(["amfi_code", "date"], inplace=True)
    df["daily_return_pct"] = df.groupby("amfi_code")["nav"].pct_change() * 100
    df["daily_return_pct"] = df["daily_return_pct"].round(6)

    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Final shape: {df.shape}")
    print(f"  Null check:\n{df.isnull().sum()}")

    save_clean(df, "clean_nav.csv")
    return df


# ════════════════════════════════════════════════════════════════
# TASK 2 — Clean investor_transactions
# ════════════════════════════════════════════════════════════════
def clean_transactions():
    print(f"\n{SEP}\n  TASK 2 — Cleaning investor_transactions\n{SEP}")
    df = load_raw("08_investor_transactions.csv")

    # Standardise transaction_type
    df["transaction_type"] = df["transaction_type"].str.strip().str.title()
    valid_types = ["Sip", "Lumpsum", "Redemption"]
    before = len(df)
    df = df[df["transaction_type"].isin(valid_types)]
    print(f"  Removed {before - len(df)} rows with invalid transaction_type")

    # Fix date format
    date_col = "transaction_date" if "transaction_date" in df.columns else df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df.dropna(subset=[date_col], inplace=True)

    # Validate amount > 0
    amt_col = "amount_inr" if "amount_inr" in df.columns else "amount"
    df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce")
    before = len(df)
    df = df[df[amt_col] > 0]
    print(f"  Removed {before - len(df)} rows with amount <= 0")

    # KYC status check
    if "kyc_status" in df.columns:
        print(f"  KYC status counts:\n{df['kyc_status'].value_counts().to_string()}")

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    print(f"  Final shape: {df.shape}")
    print(f"  Null check:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    save_clean(df, "clean_transactions.csv")
    return df


# ════════════════════════════════════════════════════════════════
# TASK 3 — Clean scheme_performance
# ════════════════════════════════════════════════════════════════
def clean_performance():
    print(f"\n{SEP}\n  TASK 3 — Cleaning scheme_performance\n{SEP}")
    df = load_raw("07_scheme_performance.csv")

    # Validate numeric return columns
    return_cols = [c for c in df.columns if "return" in c.lower() or "pct" in c.lower()]
    for col in return_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flag negative Sharpe ratios (don't drop — just flag)
    if "sharpe_ratio" in df.columns:
        df["sharpe_ratio"] = pd.to_numeric(df["sharpe_ratio"], errors="coerce")
        neg_sharpe = (df["sharpe_ratio"] < 0).sum()
        print(f"  Negative Sharpe ratios: {neg_sharpe} funds (flagged)")
        df["sharpe_flag"] = df["sharpe_ratio"] < 0

    # Validate expense_ratio range 0.1% – 2.5%
    if "expense_ratio_pct" in df.columns:
        df["expense_ratio_pct"] = pd.to_numeric(df["expense_ratio_pct"], errors="coerce")
        out = ((df["expense_ratio_pct"] < 0.05) | (df["expense_ratio_pct"] > 3.0)).sum()
        print(f"  Expense ratio out of range [0.05–3.0]: {out} funds")

    df.drop_duplicates(inplace=True)
    print(f"  Final shape: {df.shape}")
    save_clean(df, "clean_performance.csv")
    return df


# ════════════════════════════════════════════════════════════════
# Clean remaining 7 datasets (straightforward)
# ════════════════════════════════════════════════════════════════
def clean_others():
    print(f"\n{SEP}\n  Cleaning remaining datasets\n{SEP}")
    others = [
        ("01_fund_master.csv",           "clean_fund_master.csv"),
        ("03_aum_by_fund_house.csv",     "clean_aum_by_fund_house.csv"),
        ("04_monthly_sip_inflows.csv",   "clean_monthly_sip_inflows.csv"),
        ("05_category_inflows.csv",      "clean_category_inflows.csv"),
        ("06_industry_folio_count.csv",  "clean_industry_folio_count.csv"),
        ("09_portfolio_holdings.csv",    "clean_portfolio_holdings.csv"),
        ("10_benchmark_indices.csv",     "clean_benchmark_indices.csv"),
    ]
    dfs = {}
    for raw_file, clean_file in others:
        df = load_raw(raw_file)
        df.drop_duplicates(inplace=True)
        df.dropna(how="all", inplace=True)
        save_clean(df, clean_file)
        dfs[clean_file] = df
    return dfs


# ════════════════════════════════════════════════════════════════
# TASK 5 — Load into SQLite
# ════════════════════════════════════════════════════════════════
def load_to_sqlite(nav_df, tx_df, perf_df):
    print(f"\n{SEP}\n  TASK 5 — Loading into SQLite: {DB_PATH}\n{SEP}")
    conn = sqlite3.connect(DB_PATH)

    # Helper to load processed CSV or raw if processed missing
    def get_df(clean_file, raw_file, **kwargs):
        p = os.path.join(PROC_DIR, clean_file)
        if os.path.exists(p):
            return pd.read_csv(p, low_memory=False, **kwargs)
        return pd.read_csv(os.path.join(RAW_DIR, raw_file), low_memory=False, **kwargs)

    tables = [
        (nav_df,  "fact_nav"),
        (tx_df,   "fact_transactions"),
        (perf_df, "fact_performance"),
        (get_df("clean_fund_master.csv",           "01_fund_master.csv"),           "dim_fund"),
        (get_df("clean_aum_by_fund_house.csv",     "03_aum_by_fund_house.csv"),     "fact_aum"),
        (get_df("clean_monthly_sip_inflows.csv",   "04_monthly_sip_inflows.csv"),   "fact_sip_industry"),
        (get_df("clean_category_inflows.csv",      "05_category_inflows.csv"),      "fact_category_inflows"),
        (get_df("clean_industry_folio_count.csv",  "06_industry_folio_count.csv"),  "fact_folio_count"),
        (get_df("clean_portfolio_holdings.csv",    "09_portfolio_holdings.csv"),    "fact_portfolio"),
        (get_df("clean_benchmark_indices.csv",     "10_benchmark_indices.csv"),     "fact_benchmark"),
    ]

    for df, table in tables:
        df.to_sql(table, conn, if_exists="replace", index=False)
        count = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", conn).iloc[0, 0]
        print(f"  ✓  {table:<30} {count:>8,} rows")

    conn.close()
    print(f"\n  Database saved: {DB_PATH}")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(SEP)
    print("  BLUESTOCK MF CAPSTONE | Day 2 — ETL Pipeline")
    print(SEP)

    nav_df  = clean_nav_history()
    tx_df   = clean_transactions()
    perf_df = clean_performance()
    clean_others()
    load_to_sqlite(nav_df, tx_df, perf_df)

    print(f"\n{SEP}")
    print("  ETL COMPLETE — All cleaned CSVs + bluestock_mf.db ready")
    print(f"{SEP}\n")

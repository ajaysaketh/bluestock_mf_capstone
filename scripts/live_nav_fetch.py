"""
live_nav_fetch.py
-----------------
Fetches live NAV data from mfapi.in for 5 key large-cap schemes
and saves each as a raw CSV in data/raw/.

Usage:
    python scripts/live_nav_fetch.py

Output files (in data/raw/):
    nav_hdfc_top100.csv
    nav_sbi_bluechip.csv
    nav_icici_bluechip.csv
    nav_nippon_largecap.csv
    nav_axis_bluechip.csv
    nav_kotak_bluechip.csv
"""

import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Project paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Scheme registry ────────────────────────────────────────────────────────────
SCHEMES = {
    "hdfc_top100":      {"code": 125497, "name": "HDFC Top 100 Direct Growth"},
    "sbi_bluechip":     {"code": 119551, "name": "SBI Bluechip Direct Growth"},
    "icici_bluechip":   {"code": 120503, "name": "ICICI Pru Bluechip Direct Growth"},
    "nippon_largecap":  {"code": 118632, "name": "Nippon India Large Cap Direct Growth"},
    "axis_bluechip":    {"code": 119092, "name": "Axis Bluechip Direct Growth"},
    "kotak_bluechip":   {"code": 120841, "name": "Kotak Bluechip Direct Growth"},
}

BASE_URL = "https://api.mfapi.in/mf/{code}"

# ── Fetch & save helper ────────────────────────────────────────────────────────
def fetch_scheme_nav(scheme_key: str, scheme_info: dict) -> pd.DataFrame | None:
    """
    Hits the mfapi endpoint, parses JSON, returns a tidy DataFrame.
    Columns: date, nav, scheme_code, scheme_name
    """
    code = scheme_info["code"]
    url  = BASE_URL.format(code=code)
    print(f"\n[FETCH] {scheme_info['name']}  (code={code})")
    print(f"        URL → {url}")

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Network error: {e}")
        return None

    payload = resp.json()

    # ── Meta info ──────────────────────────────────────────────────────────────
    meta = payload.get("meta", {})
    print(f"  Fund  : {meta.get('fund_house', 'N/A')}")
    print(f"  Scheme: {meta.get('scheme_name', 'N/A')}")
    print(f"  Type  : {meta.get('scheme_type', 'N/A')}  |  "
          f"Category: {meta.get('scheme_category', 'N/A')}")

    # ── NAV history ────────────────────────────────────────────────────────────
    nav_records = payload.get("data", [])
    if not nav_records:
        print("  ✗ No NAV data returned.")
        return None

    df = pd.DataFrame(nav_records)                     # cols: date, nav
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
    df["scheme_code"] = code
    df["scheme_name"] = meta.get("scheme_name", scheme_info["name"])
    df["fund_house"]  = meta.get("fund_house", "")
    df["scheme_type"] = meta.get("scheme_type", "")
    df["category"]    = meta.get("scheme_category", "")

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Quick diagnostics ──────────────────────────────────────────────────────
    latest = df.iloc[-1]
    print(f"  ✓ Records: {len(df):,}  |  "
          f"Range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Latest NAV: ₹{latest['nav']:.4f}  on {latest['date'].date()}")
    null_count = df["nav"].isna().sum()
    if null_count:
        print(f"  ⚠ Null NAVs: {null_count}")

    return df


def save_csv(df: pd.DataFrame, scheme_key: str) -> Path:
    """Saves the DataFrame as CSV; returns the output path."""
    out_path = RAW_DIR / f"nav_{scheme_key}.csv"
    df.to_csv(out_path, index=False)
    print(f"  💾 Saved → {out_path.relative_to(BASE_DIR)}  ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  BLUESTOCK MF CAPSTONE — Live NAV Fetch")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    summary = []
    for scheme_key, scheme_info in SCHEMES.items():
        df = fetch_scheme_nav(scheme_key, scheme_info)
        if df is not None:
            out_path = save_csv(df, scheme_key)
            summary.append({
                "scheme_key":  scheme_key,
                "scheme_code": scheme_info["code"],
                "scheme_name": scheme_info["name"],
                "records":     len(df),
                "latest_date": df["date"].max().date(),
                "latest_nav":  df["nav"].iloc[-1],
                "file":        str(out_path.relative_to(BASE_DIR)),
                "status":      "SUCCESS",
            })
        else:
            summary.append({
                "scheme_key":  scheme_key,
                "scheme_code": scheme_info["code"],
                "scheme_name": scheme_info["name"],
                "status":      "FAILED",
            })

    # ── Fetch summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FETCH SUMMARY")
    print("=" * 65)
    summary_df = pd.DataFrame(summary)
    print(summary_df[["scheme_key", "scheme_code", "records",
                       "latest_nav", "status"]].to_string(index=False))

    # Save fetch log
    log_path = RAW_DIR / "fetch_log.csv"
    summary_df.to_csv(log_path, index=False)
    print(f"\n  📋 Fetch log saved → {log_path.relative_to(BASE_DIR)}")
    print("\n✅  Live NAV fetch complete.\n")


if __name__ == "__main__":
    main()

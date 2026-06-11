"""
performance_analytics.py  —  Bluestock MF Capstone | Day 4
Computes all key performance and risk metrics from NAV history.

Tasks:
  1. Daily returns for all 40 funds
  2. CAGR for 1yr / 3yr / 5yr
  3. Sharpe Ratio  (Rf = 6.5%)
  4. Sortino Ratio
  5. Alpha & Beta  (OLS vs Nifty 100)
  6. Maximum Drawdown
  7. Fund Scorecard (composite 0-100)
  8. Benchmark comparison chart

Usage:
    python scripts/performance_analytics.py
"""

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT     = os.getcwd()
RAW_DIR  = os.path.join(ROOT, "data", "raw")
PROC_DIR = os.path.join(ROOT, "data", "processed")
REP_DIR  = os.path.join(ROOT, "reports")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(REP_DIR,  exist_ok=True)

RF            = 0.065          # RBI repo rate proxy
TRADING_DAYS  = 252
SEP           = "=" * 60

print(SEP)
print("  BLUESTOCK MF CAPSTONE | Day 4 — Performance Analytics")
print(SEP)

# ── Load data ─────────────────────────────────────────────────
nav = pd.read_csv(os.path.join(RAW_DIR, "02_nav_history.csv"), low_memory=False)
if "tdate" in nav.columns:
    nav.rename(columns={"tdate": "date", "tnav": "nav"}, inplace=True)
nav["date"]      = pd.to_datetime(nav["date"], errors="coerce")
nav["amfi_code"] = nav["amfi_code"].astype(str)
nav["nav"]       = pd.to_numeric(nav["nav"], errors="coerce")
nav.sort_values(["amfi_code", "date"], inplace=True)
nav.reset_index(drop=True, inplace=True)

fm = pd.read_csv(os.path.join(RAW_DIR, "01_fund_master.csv"))
fm["amfi_code"] = fm["amfi_code"].astype(str)

bench = pd.read_csv(os.path.join(RAW_DIR, "10_benchmark_indices.csv"), low_memory=False)
bench["date"] = pd.to_datetime(bench["date"] if "date" in bench.columns
                               else bench.columns[0], errors="coerce")
# Find nifty100 column
nifty_col = next((c for c in bench.columns if "100" in c.lower() or "nifty" in c.lower()), bench.columns[1])
bench = bench[["date", nifty_col]].rename(columns={nifty_col: "nifty100"})
bench["nifty100"] = pd.to_numeric(bench["nifty100"], errors="coerce")
bench.dropna(inplace=True)
bench.sort_values("date", inplace=True)
bench["bench_return"] = bench["nifty100"].pct_change()

print(f"  NAV rows   : {len(nav):,}")
print(f"  Funds      : {nav['amfi_code'].nunique()}")
print(f"  Bench rows : {len(bench):,}")

# ── Task 1: Daily returns ──────────────────────────────────────
print(f"\n--- Task 1: Daily Returns ---")
nav["daily_return"] = nav.groupby("amfi_code")["nav"].pct_change()
nav.to_csv(os.path.join(PROC_DIR, "returns_computed.csv"), index=False)
print(f"  returns_computed.csv saved  {len(nav):,} rows")

# ── Helper: CAGR ──────────────────────────────────────────────
def compute_cagr(nav_series, years):
    nav_series = nav_series.dropna()
    if len(nav_series) < 2:
        return np.nan
    n  = min(int(years * TRADING_DAYS), len(nav_series) - 1)
    v0 = nav_series.iloc[-n-1]
    v1 = nav_series.iloc[-1]
    if v0 <= 0:
        return np.nan
    return round(((v1 / v0) ** (1 / years) - 1) * 100, 4)

# ── Task 2: CAGR ──────────────────────────────────────────────
print(f"\n--- Task 2: CAGR ---")
cagr_rows = []
for code, grp in nav.groupby("amfi_code"):
    cagr_rows.append({
        "amfi_code":    code,
        "cagr_1yr_pct": compute_cagr(grp["nav"], 1),
        "cagr_3yr_pct": compute_cagr(grp["nav"], 3),
        "cagr_5yr_pct": compute_cagr(grp["nav"], 5),
    })
cagr_df = pd.DataFrame(cagr_rows)
cagr_df.to_csv(os.path.join(PROC_DIR, "cagr_report.csv"), index=False)
print(f"  cagr_report.csv saved")
print(cagr_df.dropna().head(5).to_string(index=False))

# ── Task 3: Sharpe Ratio ──────────────────────────────────────
print(f"\n--- Task 3: Sharpe Ratio ---")
daily_rf = RF / TRADING_DAYS
sharpe_rows = []
for code, grp in nav.groupby("amfi_code"):
    r = grp["daily_return"].dropna()
    if len(r) < 30:
        continue
    excess = r.mean() - daily_rf
    std    = r.std()
    sharpe = round((excess / std * np.sqrt(TRADING_DAYS)), 4) if std > 0 else 0
    vol    = round(r.std() * np.sqrt(TRADING_DAYS) * 100, 4)
    name   = fm[fm["amfi_code"] == code]["scheme_name"].values
    sharpe_rows.append({
        "amfi_code":      code,
        "scheme_name":    name[0] if len(name) else code,
        "sharpe_ratio":   sharpe,
        "volatility_pct": vol,
    })
sharpe_df = pd.DataFrame(sharpe_rows)
sharpe_df.to_csv(os.path.join(PROC_DIR, "sharpe_values.csv"), index=False)
print(f"  sharpe_values.csv saved  {len(sharpe_df)} funds")
print(f"  Top 5 by Sharpe:")
print(sharpe_df.nlargest(5, "sharpe_ratio")[["scheme_name","sharpe_ratio","volatility_pct"]].to_string(index=False))

# ── Task 4: Sortino Ratio ─────────────────────────────────────
print(f"\n--- Task 4: Sortino Ratio ---")
sortino_rows = []
for code, grp in nav.groupby("amfi_code"):
    r = grp["daily_return"].dropna()
    if len(r) < 30:
        continue
    excess       = r.mean() - daily_rf
    downside_std = r[r < 0].std()
    sortino      = round((excess / downside_std * np.sqrt(TRADING_DAYS)), 4) if downside_std > 0 else 0
    sortino_rows.append({"amfi_code": code, "sortino_ratio": sortino})
sortino_df = pd.DataFrame(sortino_rows)
sortino_df.to_csv(os.path.join(PROC_DIR, "sortino_values.csv"), index=False)
print(f"  sortino_values.csv saved")

# ── Task 5: Alpha & Beta ──────────────────────────────────────
print(f"\n--- Task 5: Alpha & Beta ---")
ab_rows = []
for code, grp in nav.groupby("amfi_code"):
    merged = grp[["date","daily_return"]].merge(bench[["date","bench_return"]], on="date", how="inner")
    merged.dropna(inplace=True)
    if len(merged) < 60:
        continue
    slope, intercept, r_val, p_val, _ = stats.linregress(
        merged["bench_return"], merged["daily_return"])
    alpha = round(intercept * TRADING_DAYS * 100, 4)
    beta  = round(slope, 4)
    ab_rows.append({"amfi_code": code, "alpha": alpha, "beta": beta, "r_squared": round(r_val**2, 4)})
ab_df = pd.DataFrame(ab_rows)
ab_df.to_csv(os.path.join(PROC_DIR, "alpha_beta.csv"), index=False)
print(f"  alpha_beta.csv saved  {len(ab_df)} funds")
print(f"  Top Alpha funds:")
print(ab_df.nlargest(5, "alpha")[["amfi_code","alpha","beta","r_squared"]].to_string(index=False))

# ── Task 6: Max Drawdown ──────────────────────────────────────
print(f"\n--- Task 6: Max Drawdown ---")
dd_rows = []
for code, grp in nav.groupby("amfi_code"):
    nav_s   = grp["nav"].dropna()
    if len(nav_s) < 10:
        continue
    rolling_max = nav_s.cummax()
    drawdown    = (nav_s / rolling_max - 1)
    max_dd      = round(drawdown.min() * 100, 4)
    dd_rows.append({"amfi_code": code, "max_drawdown_pct": max_dd})
dd_df = pd.DataFrame(dd_rows)
dd_df.to_csv(os.path.join(PROC_DIR, "max_drawdown.csv"), index=False)
print(f"  max_drawdown.csv saved")

# ── Task 7: Fund Scorecard ────────────────────────────────────
print(f"\n--- Task 7: Fund Scorecard ---")
score_df = (
    cagr_df[["amfi_code","cagr_3yr_pct"]]
    .merge(sharpe_df[["amfi_code","sharpe_ratio","volatility_pct"]], on="amfi_code", how="outer")
    .merge(ab_df[["amfi_code","alpha","beta"]], on="amfi_code", how="outer")
    .merge(dd_df[["amfi_code","max_drawdown_pct"]], on="amfi_code", how="outer")
    .merge(fm[["amfi_code","scheme_name","fund_house","category","sub_category","expense_ratio_pct"]], on="amfi_code", how="left")
)
score_df.dropna(subset=["cagr_3yr_pct","sharpe_ratio"], inplace=True)

def rank_score(series, ascending=False):
    return series.rank(ascending=ascending, pct=True) * 100

score_df["score_3yr"]    = rank_score(score_df["cagr_3yr_pct"])
score_df["score_sharpe"] = rank_score(score_df["sharpe_ratio"])
score_df["score_alpha"]  = rank_score(score_df["alpha"].fillna(0))
score_df["score_er"]     = rank_score(score_df["expense_ratio_pct"].fillna(1.5), ascending=True)
score_df["score_dd"]     = rank_score(score_df["max_drawdown_pct"].fillna(-20), ascending=True)

score_df["composite_score"] = (
    score_df["score_3yr"]    * 0.30 +
    score_df["score_sharpe"] * 0.25 +
    score_df["score_alpha"]  * 0.20 +
    score_df["score_er"]     * 0.15 +
    score_df["score_dd"]     * 0.10
).round(1)

score_df.sort_values("composite_score", ascending=False, inplace=True)
score_df["rank"] = range(1, len(score_df) + 1)
score_df.to_csv(os.path.join(PROC_DIR, "fund_scorecard.csv"), index=False)
print(f"  fund_scorecard.csv saved  {len(score_df)} funds")
print(f"\n  TOP 10 FUND SCORECARD:")
cols = ["rank","scheme_name","cagr_3yr_pct","sharpe_ratio","alpha","composite_score"]
print(score_df.head(10)[cols].to_string(index=False))

# ── Task 8: Benchmark Chart ───────────────────────────────────
print(f"\n--- Task 8: Benchmark Comparison Chart ---")
top5 = score_df.head(5)["amfi_code"].tolist()
fig, ax = plt.subplots(figsize=(14, 6))
for code in top5:
    grp  = nav[nav["amfi_code"] == code].sort_values("date")
    name = fm[fm["amfi_code"] == code]["scheme_name"].values
    label = str(name[0]).split(" Fund")[0][:22] if len(name) else code
    base  = grp["nav"].iloc[0]
    ax.plot(grp["date"], grp["nav"] / base * 100,
            linewidth=1.8, label=label, alpha=0.9)

bench_filt = bench[(bench["date"] >= nav["date"].min()) & (bench["date"] <= nav["date"].max())]
base_b = bench_filt["nifty100"].iloc[0]
ax.plot(bench_filt["date"], bench_filt["nifty100"] / base_b * 100,
        color="gray", linewidth=2, linestyle="--", label="Nifty 100 (Benchmark)", alpha=0.8)

ax.set_title("Top 5 Funds vs Nifty 100 — Indexed to 100", fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Indexed Return (Base=100)")
ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1)); ax.grid(True, alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(REP_DIR, "benchmark_chart.png"), dpi=120, bbox_inches="tight")
plt.close()
print(f"  benchmark_chart.png saved")

print(f"\n{SEP}")
print(f"  DAY 4 COMPLETE — All performance metrics computed")
print(f"  Files in data/processed/:")
for f in ["returns_computed.csv","cagr_report.csv","sharpe_values.csv",
          "sortino_values.csv","alpha_beta.csv","max_drawdown.csv","fund_scorecard.csv"]:
    p = os.path.join(PROC_DIR, f)
    if os.path.exists(p):
        rows = len(open(p).readlines()) - 1
        print(f"    {f:<30} {rows} rows")
print(SEP)

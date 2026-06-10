"""
performance_analytics.py  —  Day 4: Fund Performance Analytics
Run: python scripts/performance_analytics.py
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

RAW  = Path(r'C:\Users\ajays\bluestock_mf_capstone\data\raw')
PROC = Path(r'C:\Users\ajays\bluestock_mf_capstone\data\processed')
FIGS = Path(r'C:\Users\ajays\bluestock_mf_capstone\reports')
PROC.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

RF = 0.065
TRADING_DAYS = 252

print("="*60)
print("  DAY 4 — Fund Performance Analytics")
print("="*60)

# ── Load & fix column names automatically ─────────────────────────────────────
print("\nLoading data...")
nav = pd.read_csv(RAW / '02_nav_history.csv')
print(f"  NAV columns found: {nav.columns.tolist()}")

# Auto-detect and rename columns
col_map = {}
for col in nav.columns:
    cl = col.lower().strip()
    if cl in ('date','nav_date','navdate','trade_date'): col_map[col] = 'date'
    elif cl in ('nav','net_asset_value','nav_value'):    col_map[col] = 'nav'
    elif cl in ('amfi_code','scheme_code','code','amficode'): col_map[col] = 'amfi_code'
nav.rename(columns=col_map, inplace=True)
print(f"  Renamed columns: {nav.columns.tolist()}")

nav['date'] = pd.to_datetime(nav['date'], errors='coerce')
nav['nav']  = pd.to_numeric(nav['nav'],  errors='coerce')
nav['amfi_code'] = nav['amfi_code'].astype(str)
nav = nav.dropna(subset=['date','nav'])
nav = nav.sort_values(['amfi_code','date']).reset_index(drop=True)

fm    = pd.read_csv(RAW / '01_fund_master.csv')
fm['amfi_code'] = fm['amfi_code'].astype(str)
bench = pd.read_csv(RAW / '10_benchmark_indices.csv')
bench_col_map = {}
for col in bench.columns:
    if col.lower().strip() in ('date','trade_date','nav_date'):
        bench_col_map[col] = 'date'
bench.rename(columns=bench_col_map, inplace=True)
bench['date'] = pd.to_datetime(bench['date'], errors='coerce')

print(f"  NAV records : {len(nav):,}")
print(f"  Funds       : {len(fm)}")
print(f"  Bench cols  : {bench.columns.tolist()}")

# ── Task 1: Daily Returns ─────────────────────────────────────────────────────
print("\n--- Task 1: Daily Returns ---")
nav['daily_return'] = nav.groupby('amfi_code')['nav'].pct_change()
nav[['amfi_code','date','nav','daily_return']].to_csv(PROC / 'returns_computed.csv', index=False)
print(f"  ✅ returns_computed.csv  {len(nav):,} rows")

# ── Task 2: CAGR ──────────────────────────────────────────────────────────────
print("\n--- Task 2: CAGR ---")
def cagr(grp, years):
    grp = grp.sort_values('date')
    end = grp['date'].max()
    start = end - pd.DateOffset(years=years)
    s = grp[grp['date'] >= start]
    if len(s) < 2: return np.nan
    n = (s['date'].iloc[-1] - s['date'].iloc[0]).days / 365.25
    if n <= 0 or s['nav'].iloc[0] <= 0: return np.nan
    return round(((s['nav'].iloc[-1]/s['nav'].iloc[0])**(1/n)-1)*100, 2)

rows = []
for code, g in nav.groupby('amfi_code'):
    rows.append({'amfi_code':code,'cagr_1yr':cagr(g,1),'cagr_3yr':cagr(g,3),'cagr_5yr':cagr(g,5)})
cagr_df = pd.DataFrame(rows).merge(fm[['amfi_code','scheme_name','sub_category','fund_house']], on='amfi_code', how='left')
cagr_df.to_csv(PROC / 'cagr_report.csv', index=False)
print(f"  ✅ cagr_report.csv  {len(cagr_df)} rows")
print(cagr_df.nlargest(5,'cagr_3yr')[['scheme_name','cagr_1yr','cagr_3yr','cagr_5yr']].to_string(index=False))

# ── Task 3: Sharpe ────────────────────────────────────────────────────────────
print("\n--- Task 3: Sharpe Ratio ---")
daily_rf = RF / TRADING_DAYS
rows = []
for code, g in nav.groupby('amfi_code'):
    r = g['daily_return'].dropna()
    if len(r) < 30: continue
    sharpe = (r - daily_rf).mean() / r.std() * np.sqrt(TRADING_DAYS)
    rows.append({'amfi_code':code,'sharpe_ratio':round(sharpe,4),'volatility_pct':round(r.std()*np.sqrt(TRADING_DAYS)*100,2)})
sharpe_df = pd.DataFrame(rows).merge(fm[['amfi_code','scheme_name','sub_category']], on='amfi_code', how='left')
sharpe_df.to_csv(PROC / 'sharpe_values.csv', index=False)
print(f"  ✅ sharpe_values.csv  {len(sharpe_df)} rows")
print(sharpe_df.nlargest(5,'sharpe_ratio')[['scheme_name','sharpe_ratio','volatility_pct']].to_string(index=False))

# ── Task 4: Sortino ───────────────────────────────────────────────────────────
print("\n--- Task 4: Sortino Ratio ---")
rows = []
for code, g in nav.groupby('amfi_code'):
    r = g['daily_return'].dropna()
    if len(r) < 30: continue
    down_std = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    sortino  = ((r - daily_rf).mean() * TRADING_DAYS / down_std) if down_std > 0 else np.nan
    rows.append({'amfi_code':code,'sortino_ratio':round(sortino,4)})
sortino_df = pd.DataFrame(rows).merge(fm[['amfi_code','scheme_name','sub_category']], on='amfi_code', how='left')
sortino_df.to_csv(PROC / 'sortino_values.csv', index=False)
print(f"  ✅ sortino_values.csv  {len(sortino_df)} rows")

# ── Task 5: Alpha & Beta ──────────────────────────────────────────────────────
print("\n--- Task 5: Alpha & Beta ---")
bench_ret = bench[['date','Nifty100']].copy()
bench_ret['bench_return'] = bench_ret['Nifty100'].pct_change()
bench_ret = bench_ret.dropna()
rows = []
for code, g in nav.groupby('amfi_code'):
    g2 = g[['date','daily_return']].dropna()
    m  = g2.merge(bench_ret[['date','bench_return']], on='date', how='inner')
    if len(m) < 60: continue
    slope, intercept, r_val, _, _ = stats.linregress(m['bench_return'], m['daily_return'])
    rows.append({'amfi_code':code,'alpha':round(intercept*TRADING_DAYS*100,4),'beta':round(slope,4),'r_squared':round(r_val**2,4)})
ab_df = pd.DataFrame(rows).merge(fm[['amfi_code','scheme_name','sub_category']], on='amfi_code', how='left')
ab_df.to_csv(PROC / 'alpha_beta.csv', index=False)
print(f"  ✅ alpha_beta.csv  {len(ab_df)} rows")
print(ab_df.nlargest(5,'alpha')[['scheme_name','alpha','beta']].to_string(index=False))

# ── Task 6: Max Drawdown ──────────────────────────────────────────────────────
print("\n--- Task 6: Maximum Drawdown ---")
rows = []
for code, g in nav.groupby('amfi_code'):
    g = g.sort_values('date')
    dd = (g['nav'] - g['nav'].cummax()) / g['nav'].cummax() * 100
    rows.append({'amfi_code':code,'max_drawdown_pct':round(dd.min(),2)})
dd_df = pd.DataFrame(rows).merge(fm[['amfi_code','scheme_name','sub_category']], on='amfi_code', how='left')
dd_df.to_csv(PROC / 'max_drawdown.csv', index=False)
print(f"  ✅ max_drawdown.csv  {len(dd_df)} rows")
print(dd_df.nsmallest(5,'max_drawdown_pct')[['scheme_name','max_drawdown_pct']].to_string(index=False))

# ── Task 7: Scorecard ─────────────────────────────────────────────────────────
print("\n--- Task 7: Fund Scorecard ---")
score_df = cagr_df[['amfi_code','scheme_name','sub_category','fund_house','cagr_3yr']].copy()
score_df = score_df.merge(sharpe_df[['amfi_code','sharpe_ratio']], on='amfi_code', how='left')
score_df = score_df.merge(ab_df[['amfi_code','alpha']], on='amfi_code', how='left')
score_df = score_df.merge(dd_df[['amfi_code','max_drawdown_pct']], on='amfi_code', how='left')
score_df = score_df.merge(fm[['amfi_code','expense_ratio_pct']], on='amfi_code', how='left')
score_df['composite_score'] = (
    score_df['cagr_3yr'].rank(pct=True)          * 0.30 +
    score_df['sharpe_ratio'].rank(pct=True)       * 0.25 +
    score_df['alpha'].rank(pct=True)              * 0.20 +
    (1-score_df['expense_ratio_pct'].rank(pct=True)) * 0.15 +
    (1-score_df['max_drawdown_pct'].rank(pct=True))  * 0.10
) * 100
score_df = score_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
score_df['rank'] = range(1, len(score_df)+1)
score_df['composite_score'] = score_df['composite_score'].round(2)
score_df.to_csv(PROC / 'fund_scorecard.csv', index=False)
print(f"  ✅ fund_scorecard.csv  {len(score_df)} rows")
print(score_df.head(10)[['rank','scheme_name','composite_score','cagr_3yr','sharpe_ratio']].to_string(index=False))

# ── Task 8: Chart ─────────────────────────────────────────────────────────────
print("\n--- Task 8: Benchmark Chart ---")
sns.set_theme(style='darkgrid')
top5 = score_df.head(5)
cutoff = nav['date'].max() - pd.DateOffset(years=3)
fig, axes = plt.subplots(2,1,figsize=(14,12))

ax = axes[0]
for _, row in top5.iterrows():
    g = nav[(nav['amfi_code']==row['amfi_code']) & (nav['date']>=cutoff)].sort_values('date')
    if len(g) > 0:
        ax.plot(g['date'], g['nav']/g['nav'].iloc[0]*100, label=row['scheme_name'].split(' Fund')[0][:22], linewidth=2)
bench3 = bench[bench['date']>=cutoff]
ax.plot(bench3['date'], bench3['Nifty50']/bench3['Nifty50'].iloc[0]*100, 'k--', label='Nifty50', linewidth=2)
ax.plot(bench3['date'], bench3['Nifty100']/bench3['Nifty100'].iloc[0]*100, 'gray', linestyle=':', label='Nifty100', linewidth=2)
ax.set_title('Top 5 Funds vs Benchmark — 3 Year Performance', fontsize=13, fontweight='bold')
ax.set_ylabel('Indexed Return (Base=100)')
ax.legend(fontsize=8)

ax2 = axes[1]
colors = plt.cm.RdYlGn(np.linspace(0.3,0.9,10))
top10 = score_df.head(10)
bars = ax2.barh([n.split(' Fund')[0][:28] for n in top10['scheme_name']], top10['composite_score'], color=colors)
for bar, val in zip(bars, top10['composite_score']):
    ax2.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, f'{val:.1f}', va='center', fontsize=9, fontweight='bold')
ax2.set_title('Fund Scorecard — Top 10', fontsize=13, fontweight='bold')
ax2.set_xlabel('Composite Score (0-100)')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(str(FIGS/'benchmark_chart.png'), dpi=120, bbox_inches='tight')
plt.close()
print(f"  ✅ benchmark_chart.png saved")

print("\n" + "="*60)
print("  ✅ DAY 4 COMPLETE!")
print("="*60)
print("  Next: git add -A && git commit -m 'Day 4: Performance analytics complete' && git push origin main")

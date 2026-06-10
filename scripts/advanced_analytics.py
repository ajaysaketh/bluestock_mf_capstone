"""
advanced_analytics.py  —  Day 6: Advanced Analytics + Risk Metrics
Run: python scripts/advanced_analytics.py
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RAW  = Path(r'C:\Users\ajays\bluestock_mf_capstone\data\raw')
PROC = Path(r'C:\Users\ajays\bluestock_mf_capstone\data\processed')
FIGS = Path(r'C:\Users\ajays\bluestock_mf_capstone\reports')
PROC.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

RF = 0.065
TRADING_DAYS = 252

print("="*60)
print("  DAY 6 — Advanced Analytics + Risk Metrics")
print("="*60)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading data...")
nav = pd.read_csv(RAW / '02_nav_history.csv')
nav['date'] = pd.to_datetime(nav['date'], errors='coerce')
nav['amfi_code'] = nav['amfi_code'].astype(str)
nav = nav.sort_values(['amfi_code','date']).reset_index(drop=True)
nav['daily_return'] = nav.groupby('amfi_code')['nav'].pct_change()

tx  = pd.read_csv(RAW / '08_investor_transactions.csv')
tx['transaction_date'] = pd.to_datetime(tx['transaction_date'], errors='coerce')

fm  = pd.read_csv(RAW / '01_fund_master.csv')
fm['amfi_code'] = fm['amfi_code'].astype(str)

port = pd.read_csv(RAW / '09_portfolio_holdings.csv')
port['amfi_code'] = port['amfi_code'].astype(str)

print(f"  NAV: {len(nav):,} rows | TX: {len(tx):,} rows")

# ── Task 1: VaR & CVaR ───────────────────────────────────────────────────────
print("\n--- Task 1: VaR & CVaR (95%) ---")
var_rows = []
for code, grp in nav.groupby('amfi_code'):
    r = grp['daily_return'].dropna()
    if len(r) < 30: continue
    var_95  = np.percentile(r, 5)
    cvar_95 = r[r <= var_95].mean()
    var_rows.append({
        'amfi_code':       code,
        'var_95_daily':    round(var_95 * 100, 4),
        'cvar_95_daily':   round(cvar_95 * 100, 4),
        'var_95_annual':   round(var_95 * np.sqrt(TRADING_DAYS) * 100, 4),
    })
var_df = pd.DataFrame(var_rows)
var_df = var_df.merge(fm[['amfi_code','scheme_name','sub_category']], on='amfi_code', how='left')
var_df.to_csv(PROC / 'var_cvar_report.csv', index=False)
print(f"  ✅ var_cvar_report.csv  {len(var_df)} rows")
print(f"\n  Highest VaR (Riskiest Funds):")
print(var_df.nsmallest(5,'var_95_daily')[['scheme_name','var_95_daily','cvar_95_daily']].to_string(index=False))

# ── Task 2: Rolling 90-day Sharpe ────────────────────────────────────────────
print("\n--- Task 2: Rolling 90-day Sharpe Ratio ---")
top5_codes = fm[fm['sub_category']=='Large Cap']['amfi_code'].head(5).tolist()
daily_rf   = RF / TRADING_DAYS

fig, ax = plt.subplots(figsize=(14,6))
sns.set_theme(style='darkgrid')

for code in top5_codes:
    grp = nav[nav['amfi_code']==code].sort_values('date')
    r   = grp.set_index('date')['daily_return']
    rolling_sharpe = (
        r.rolling(90).mean() - daily_rf
    ) / r.rolling(90).std() * np.sqrt(TRADING_DAYS)
    name = fm[fm['amfi_code']==code]['scheme_name'].values[0]
    ax.plot(rolling_sharpe.index, rolling_sharpe.values,
            label=name.split(' Fund')[0][:20], linewidth=1.8)

ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Sharpe=1 (Good)')
ax.axhline(y=0.0, color='red',   linestyle='--', alpha=0.7, label='Sharpe=0')
ax.set_title('Rolling 90-Day Sharpe Ratio — Top 5 Large Cap Funds', fontsize=13, fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Rolling Sharpe Ratio')
ax.legend(fontsize=8, bbox_to_anchor=(1.01,1))
plt.tight_layout()
plt.savefig(str(FIGS/'rolling_sharpe_chart.png'), dpi=120, bbox_inches='tight')
plt.close()
print(f"  ✅ rolling_sharpe_chart.png saved")

# ── Task 3: Investor Cohort Analysis ─────────────────────────────────────────
print("\n--- Task 3: Investor Cohort Analysis ---")
tx['year'] = tx['transaction_date'].dt.year
first_tx   = tx.groupby('investor_id')['transaction_date'].min().reset_index()
first_tx['cohort_year'] = first_tx['transaction_date'].dt.year

tx_cohort = tx.merge(first_tx[['investor_id','cohort_year']], on='investor_id', how='left')

cohort_df = tx_cohort.groupby('cohort_year').agg(
    num_investors    = ('investor_id', 'nunique'),
    total_invested   = ('amount_inr',  'sum'),
    avg_sip_amount   = ('amount_inr',  'mean'),
    num_transactions = ('investor_id', 'count'),
).reset_index()
cohort_df['avg_investment_per_investor'] = (cohort_df['total_invested'] / cohort_df['num_investors']).round(0)
cohort_df['total_invested_crore'] = (cohort_df['total_invested'] / 1e7).round(2)
cohort_df.to_csv(PROC / 'cohort_analysis.csv', index=False)
print(f"  ✅ cohort_analysis.csv saved")
print(cohort_df[['cohort_year','num_investors','avg_sip_amount','total_invested_crore']].to_string(index=False))

# ── Task 4: SIP Continuation Analysis ────────────────────────────────────────
print("\n--- Task 4: SIP Continuation Analysis ---")
sip_tx = tx[tx['transaction_type'].str.lower().isin(['sip'])].copy()
sip_tx = sip_tx.sort_values(['investor_id','transaction_date'])

sip_gaps = sip_tx.groupby('investor_id').apply(
    lambda x: x['transaction_date'].diff().dt.days.mean()
).reset_index()
sip_gaps.columns = ['investor_id','avg_gap_days']

sip_count = sip_tx.groupby('investor_id').size().reset_index(name='sip_count')
sip_cont  = sip_gaps.merge(sip_count, on='investor_id', how='left')
sip_cont['at_risk'] = sip_cont['avg_gap_days'] > 35
sip_cont.to_csv(PROC / 'sip_continuity.csv', index=False)

at_risk_count = sip_cont['at_risk'].sum()
print(f"  ✅ sip_continuity.csv saved  {len(sip_cont)} investors")
print(f"  At-risk investors (gap>35 days): {at_risk_count} ({at_risk_count/len(sip_cont)*100:.1f}%)")

# ── Task 5: Fund Recommender ──────────────────────────────────────────────────
print("\n--- Task 5: Fund Recommender ---")
perf = pd.read_csv(PROC / 'sharpe_values.csv')
perf = perf.merge(fm[['amfi_code','risk_category','expense_ratio_pct']], on='amfi_code', how='left')

def recommend_funds(risk_appetite: str, top_n: int = 3):
    risk_map = {
        'Low':      ['Low','Moderate'],
        'Moderate': ['Moderate','High'],
        'High':     ['High','Very High'],
    }
    allowed = risk_map.get(risk_appetite, ['High','Very High'])
    filtered = perf[perf['risk_category'].isin(allowed)]
    top = filtered.nlargest(top_n, 'sharpe_ratio')[
        ['scheme_name','sub_category','sharpe_ratio','volatility_pct','risk_category']
    ]
    return top

print(f"\n  Recommendations for LOW risk investor:")
print(recommend_funds('Low').to_string(index=False))
print(f"\n  Recommendations for MODERATE risk investor:")
print(recommend_funds('Moderate').to_string(index=False))
print(f"\n  Recommendations for HIGH risk investor:")
print(recommend_funds('High').to_string(index=False))

# ── Task 6: Sector HHI ────────────────────────────────────────────────────────
print("\n--- Task 6: Sector Concentration (HHI) ---")
def compute_hhi(weights):
    w = np.array(weights) / 100
    return round((w**2).sum(), 4)

hhi_rows = []
for code, grp in port.groupby('amfi_code'):
    sector_weights = grp.groupby('sector')['weight_pct'].sum()
    hhi = compute_hhi(sector_weights.values)
    hhi_rows.append({
        'amfi_code':      code,
        'hhi':            hhi,
        'top_sector':     sector_weights.idxmax(),
        'top_sector_wt':  round(sector_weights.max(), 2),
        'num_sectors':    len(sector_weights),
        'concentration':  'High' if hhi > 0.25 else 'Moderate' if hhi > 0.15 else 'Low',
    })

hhi_df = pd.DataFrame(hhi_rows)
hhi_df = hhi_df.merge(fm[['amfi_code','scheme_name','sub_category']], on='amfi_code', how='left')
hhi_df.to_csv(PROC / 'sector_hhi.csv', index=False)
print(f"  ✅ sector_hhi.csv saved  {len(hhi_df)} funds")
print(f"\n  Most concentrated funds (High HHI):")
print(hhi_df.nlargest(5,'hhi')[['scheme_name','hhi','top_sector','concentration']].to_string(index=False))

# ── Summary Chart ─────────────────────────────────────────────────────────────
print("\n--- Generating Summary Chart ---")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# VaR distribution
axes[0,0].hist(var_df['var_95_daily'], bins=15, color='#EF5350', edgecolor='white', linewidth=1.2)
axes[0,0].set_title('Distribution of Daily VaR (95%) Across Funds', fontweight='bold')
axes[0,0].set_xlabel('Daily VaR (%)'); axes[0,0].set_ylabel('Number of Funds')
axes[0,0].axvline(var_df['var_95_daily'].mean(), color='black', linestyle='--', label=f"Mean={var_df['var_95_daily'].mean():.2f}%")
axes[0,0].legend()

# Cohort analysis
axes[0,1].bar(cohort_df['cohort_year'].astype(str), cohort_df['num_investors'],
              color='#42A5F5', edgecolor='white', linewidth=1.5)
axes[0,1].set_title('Investor Count by Cohort Year', fontweight='bold')
axes[0,1].set_xlabel('First Investment Year'); axes[0,1].set_ylabel('Number of Investors')

# SIP gap distribution
axes[1,0].hist(sip_cont['avg_gap_days'].dropna(), bins=20, color='#66BB6A', edgecolor='white', linewidth=1.2)
axes[1,0].axvline(35, color='red', linestyle='--', label='At-risk threshold (35 days)')
axes[1,0].set_title('SIP Transaction Gap Distribution', fontweight='bold')
axes[1,0].set_xlabel('Avg Days Between SIP'); axes[1,0].set_ylabel('Number of Investors')
axes[1,0].legend()

# HHI by fund
top_hhi = hhi_df.nlargest(10,'hhi')
axes[1,1].barh([n.split(' Fund')[0][:25] for n in top_hhi['scheme_name']],
               top_hhi['hhi'], color='#FFA726', edgecolor='white')
axes[1,1].set_title('Top 10 Funds by Sector Concentration (HHI)', fontweight='bold')
axes[1,1].set_xlabel('HHI Score (higher = more concentrated)')
axes[1,1].invert_yaxis()

plt.suptitle('Day 6 — Advanced Analytics Summary', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(str(FIGS/'advanced_analytics_summary.png'), dpi=120, bbox_inches='tight')
plt.close()
print(f"  ✅ advanced_analytics_summary.png saved")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ✅ DAY 6 COMPLETE!")
print("="*60)
outputs = ['var_cvar_report.csv','cohort_analysis.csv','sip_continuity.csv','sector_hhi.csv']
for f in outputs:
    rows = pd.read_csv(PROC/f).shape[0]
    print(f"  ✅ {f:<30} {rows} rows")
print(f"  ✅ rolling_sharpe_chart.png")
print(f"  ✅ advanced_analytics_summary.png")
print(f"\n  Run:")
print(f"  git add -A")
print(f"  git commit -m 'Day 6: Advanced analytics complete'")
print(f"  git push origin main")

"""
eda_analysis.py  —  Day 3: Complete EDA with 15 charts
Run: python scripts/eda_analysis.py
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # No display needed — saves directly to files
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

RAW  = Path(r'C:\Users\ajays\bluestock_mf_capstone\data\raw')
FIGS = Path(r'C:\Users\ajays\bluestock_mf_capstone\reports')
FIGS.mkdir(exist_ok=True)

sns.set_theme(style='darkgrid')
plt.rcParams.update({'figure.dpi':120, 'figure.figsize':(12,5)})

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
nav   = pd.read_csv(RAW / '02_nav_history.csv')
nav['date'] = pd.to_datetime(nav['date'])
aum   = pd.read_csv(RAW / '03_aum_by_fund_house.csv')
aum['quarter'] = pd.to_datetime(aum['quarter'])
aum['year'] = aum['quarter'].dt.year
sip   = pd.read_csv(RAW / '04_monthly_sip_inflows.csv')
sip['month_dt'] = pd.to_datetime(sip['month'] + '-01')
cat   = pd.read_csv(RAW / '05_category_inflows.csv')
fol   = pd.read_csv(RAW / '06_industry_folio_count.csv')
fol['month_dt'] = pd.to_datetime(fol['month'] + '-01')
tx    = pd.read_csv(RAW / '08_investor_transactions.csv')
tx['transaction_date'] = pd.to_datetime(tx['transaction_date'])
port  = pd.read_csv(RAW / '09_portfolio_holdings.csv')
fm    = pd.read_csv(RAW / '01_fund_master.csv')
bench = pd.read_csv(RAW / '10_benchmark_indices.csv')
bench['date'] = pd.to_datetime(bench['date'])
nav   = nav.merge(fm[['amfi_code','scheme_name','sub_category','fund_house']], on='amfi_code', how='left')
print("✅ All data loaded!")

# ── Chart 1: AUM Growth ───────────────────────────────────────────────────────
print("Chart 1: AUM Growth...")
aum_yr = aum.groupby(['fund_house','year'])['aum_crore'].max().reset_index()
aum_yr['aum_lakh_cr'] = aum_yr['aum_crore'] / 100000
fig, ax = plt.subplots(figsize=(14,6))
fh_order = aum_yr.groupby('fund_house')['aum_lakh_cr'].max().sort_values(ascending=False).index
sns.barplot(data=aum_yr, x='fund_house', y='aum_lakh_cr', hue='year', order=fh_order, palette='Blues', ax=ax)
ax.set_xticklabels([x.replace(' Mutual Fund','').replace(' Prudential','') for x in fh_order], rotation=30, ha='right')
ax.set_title('AUM Growth by Fund House 2022–2025 (₹ Lakh Crore)', fontsize=14, fontweight='bold')
ax.set_xlabel('Fund House'); ax.set_ylabel('AUM (₹ Lakh Crore)')
ax.axhline(y=12.5, color='red', linestyle='--', alpha=0.6, label='SBI ₹12.5L Cr')
ax.legend(title='Year', bbox_to_anchor=(1.01,1))
plt.tight_layout()
plt.savefig(str(FIGS/'chart1_aum_growth.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart1_aum_growth.png saved")

# ── Chart 2: SIP Inflow Trend ─────────────────────────────────────────────────
print("Chart 2: SIP Trend...")
fig, ax = plt.subplots(figsize=(14,5))
ax.plot(sip['month_dt'], sip['sip_inflow_crore'], color='#2196F3', linewidth=2, marker='o', markersize=3)
ax.plot(sip['month_dt'], sip['sip_inflow_crore'].rolling(3).mean(), color='orange', linewidth=2, linestyle='--', label='3M Avg')
peak = sip.loc[sip['sip_inflow_crore'].idxmax()]
ax.annotate(f"All-Time High\n₹{peak['sip_inflow_crore']:,.0f} Cr",
            xy=(peak['month_dt'], peak['sip_inflow_crore']),
            xytext=(peak['month_dt'], peak['sip_inflow_crore']-3000),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red', fontweight='bold')
ax.set_title('Monthly SIP Inflows Jan 2022 – Dec 2025 (₹ Crore)', fontsize=14, fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('SIP Inflow (₹ Crore)')
ax.legend(); plt.tight_layout()
plt.savefig(str(FIGS/'chart2_sip_trend.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart2_sip_trend.png saved")

# ── Chart 3: Category Heatmap ─────────────────────────────────────────────────
print("Chart 3: Category Heatmap...")
pivot = cat.pivot_table(index='category', columns='month', values='net_inflow_crore', aggfunc='sum')
fig, ax = plt.subplots(figsize=(16,7))
sns.heatmap(pivot, cmap='RdYlGn', center=0, linewidths=0.3, cbar_kws={'label':'Net Inflow (₹ Cr)'}, ax=ax)
ax.set_title('Category-wise Net Inflows Heatmap FY2024-25', fontsize=13, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(str(FIGS/'chart3_category_heatmap.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart3_category_heatmap.png saved")

# ── Chart 4: Investor Demographics ───────────────────────────────────────────
print("Chart 4: Demographics...")
fig, axes = plt.subplots(1, 3, figsize=(16,5))
age_cnt = tx['age_group'].value_counts()
axes[0].pie(age_cnt.values, labels=age_cnt.index, autopct='%1.1f%%',
            colors=sns.color_palette('Set2', len(age_cnt)), startangle=90)
axes[0].set_title('Age Group Distribution', fontweight='bold')
sip_tx = tx[tx['transaction_type']=='SIP']
age_order = ['18-25','26-35','36-45','46-55','56+']
sns.boxplot(data=sip_tx, x='age_group', y='amount_inr', order=age_order, palette='Set2', ax=axes[1])
axes[1].set_title('SIP Amount by Age Group', fontweight='bold')
axes[1].set_xlabel('Age Group'); axes[1].set_ylabel('Amount (₹)')
gen_cnt = tx['gender'].value_counts()
axes[2].pie(gen_cnt.values, labels=gen_cnt.index, autopct='%1.1f%%',
            colors=['#64B5F6','#F48FB1'], startangle=90)
axes[2].set_title('Gender Split', fontweight='bold')
plt.suptitle('Investor Demographics Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(FIGS/'chart4_demographics.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart4_demographics.png saved")

# ── Chart 5: Geographic Distribution ─────────────────────────────────────────
print("Chart 5: Geographic...")
fig, axes = plt.subplots(1, 2, figsize=(16,6))
state_amt = tx.groupby('state')['amount_inr'].sum().sort_values() / 1e7
state_amt.plot(kind='barh', ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('Transaction Amount by State (₹ Crore)', fontweight='bold')
axes[0].set_xlabel('Amount (₹ Crore)')
tier_cnt = tx['city_tier'].value_counts()
axes[1].pie(tier_cnt.values, labels=tier_cnt.index, autopct='%1.1f%%',
            colors=['#42A5F5','#FFA726'], startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[1].set_title('T30 vs B30 City Tier Split', fontweight='bold')
plt.tight_layout()
plt.savefig(str(FIGS/'chart5_geographic.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart5_geographic.png saved")

# ── Chart 6: Folio Count Growth ───────────────────────────────────────────────
print("Chart 6: Folio Growth...")
fol_total = fol.groupby('month_dt')['folio_count_crore'].sum().reset_index()
fig, ax = plt.subplots(figsize=(14,5))
ax.fill_between(fol_total['month_dt'], fol_total['folio_count_crore'], alpha=0.3, color='purple')
ax.plot(fol_total['month_dt'], fol_total['folio_count_crore'], color='purple', linewidth=2.5)
ax.set_title('Industry Folio Count Growth 2022–2025 (Crore)', fontsize=13, fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('Total Folios (Crore)')
plt.tight_layout()
plt.savefig(str(FIGS/'chart6_folio_growth.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart6_folio_growth.png saved")

# ── Chart 7: Correlation Matrix ───────────────────────────────────────────────
print("Chart 7: Correlation Matrix...")
top10_codes = fm[fm['sub_category'].isin(['Large Cap','Mid Cap'])]['amfi_code'].head(10).tolist()
nav10 = nav[nav['amfi_code'].isin(top10_codes)].pivot_table(index='date', columns='scheme_name', values='nav')
returns10 = nav10.pct_change().dropna()
corr = returns10.corr()
corr.columns = [c.split(' Fund')[0][:18] for c in corr.columns]
corr.index   = [c.split(' Fund')[0][:18] for c in corr.index]
fig, ax = plt.subplots(figsize=(12,9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            mask=mask, square=True, linewidths=0.5, ax=ax)
ax.set_title('NAV Return Correlation — 10 Selected Funds', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(str(FIGS/'chart7_correlation.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart7_correlation.png saved")

# ── Chart 8: Sector Donut ─────────────────────────────────────────────────────
print("Chart 8: Sector Donut...")
sector_wt = port.groupby('sector')['weight_pct'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10,8))
ax.pie(sector_wt.values, labels=sector_wt.index, autopct='%1.1f%%',
       pctdistance=0.82, colors=sns.color_palette('tab10', len(sector_wt)),
       wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2), startangle=90)
ax.set_title('Sector Allocation — Equity Fund Portfolios (Avg Weight %)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(str(FIGS/'chart8_sector_donut.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart8_sector_donut.png saved")

# ── Chart 9: NAV Trend Lines ──────────────────────────────────────────────────
print("Chart 9: NAV Trends...")
large_caps = fm[fm['sub_category']=='Large Cap']['amfi_code'].head(6).tolist()
nav_lc = nav[nav['amfi_code'].isin(large_caps)]
nav_pivot = nav_lc.pivot_table(index='date', columns='scheme_name', values='nav')
nav_norm = nav_pivot.div(nav_pivot.iloc[0]) * 100
fig, ax = plt.subplots(figsize=(14,6))
for col in nav_norm.columns:
    ax.plot(nav_norm.index, nav_norm[col], label=col.split(' Fund')[0][:20], linewidth=1.5)
ax.axvspan(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-12-31'), alpha=0.1, color='green', label='2023 Bull Run')
ax.axvspan(pd.Timestamp('2024-06-01'), pd.Timestamp('2024-10-31'), alpha=0.1, color='red', label='2024 Correction')
ax.set_title('NAV Growth Indexed to 100 — Large Cap Funds 2022–2026', fontsize=13, fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('NAV Index (Base=100)')
ax.legend(fontsize=8, bbox_to_anchor=(1.01,1)); plt.tight_layout()
plt.savefig(str(FIGS/'chart9_nav_trends.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart9_nav_trends.png saved")

# ── Chart 10: Payment Mode ────────────────────────────────────────────────────
print("Chart 10: Payment Mode...")
pay_cnt = tx['payment_mode'].value_counts()
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(pay_cnt.index, pay_cnt.values, color=sns.color_palette('Set2', len(pay_cnt)), edgecolor='white')
for bar, val in zip(bars, pay_cnt.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100, f'{val:,}', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Transaction Count by Payment Mode', fontsize=13, fontweight='bold')
ax.set_xlabel('Payment Mode'); ax.set_ylabel('Number of Transactions')
plt.tight_layout()
plt.savefig(str(FIGS/'chart10_payment_mode.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart10_payment_mode.png saved")

# ── Chart 11: Top Stocks ──────────────────────────────────────────────────────
print("Chart 11: Top Stocks...")
top_stocks = port.groupby(['stock_name','sector'])['weight_pct'].mean().sort_values(ascending=False).head(10).reset_index()
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(data=top_stocks, y='stock_name', x='weight_pct', hue='sector', dodge=False, palette='tab10', ax=ax)
ax.set_title('Top 10 Stocks by Average Portfolio Weight', fontsize=13, fontweight='bold')
ax.set_xlabel('Average Weight (%)'); ax.set_ylabel('Stock')
ax.legend(title='Sector', bbox_to_anchor=(1.01,1)); plt.tight_layout()
plt.savefig(str(FIGS/'chart11_top_stocks.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart11_top_stocks.png saved")

# ── Chart 12: Monthly TX Volume ───────────────────────────────────────────────
print("Chart 12: Monthly TX...")
tx['month'] = tx['transaction_date'].dt.to_period('M').astype(str)
monthly = tx.groupby(['month','transaction_type']).size().reset_index(name='count')
pivot_tx = monthly.pivot(index='month', columns='transaction_type', values='count').fillna(0)
fig, ax = plt.subplots(figsize=(14,5))
pivot_tx.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='white')
ax.set_title('Monthly Transaction Volume by Type', fontsize=13, fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('Transactions')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(str(FIGS/'chart12_monthly_tx.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart12_monthly_tx.png saved")

# ── Chart 13: Benchmark Comparison ───────────────────────────────────────────
print("Chart 13: Benchmark...")
fig, ax = plt.subplots(figsize=(14,5))
for idx, color in [('Nifty50','#1565C0'),('Nifty100','#0288D1'),('NiftyMidcap150','#00897B'),('BSESmallCap','#F4511E')]:
    norm = bench[idx] / bench[idx].iloc[0] * 100
    ax.plot(bench['date'], norm, label=idx, color=color, linewidth=1.8)
ax.set_title('Benchmark Index Performance (Indexed to 100) 2022–2026', fontsize=13, fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Index (Base=100)')
ax.legend(); plt.tight_layout()
plt.savefig(str(FIGS/'chart13_benchmark.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart13_benchmark.png saved")

# ── Chart 14: SIP Accounts vs Inflow ─────────────────────────────────────────
print("Chart 14: SIP Dual Axis...")
fig, ax1 = plt.subplots(figsize=(14,5))
ax2 = ax1.twinx()
ax1.bar(sip['month_dt'], sip['sip_inflow_crore'], color='#42A5F5', alpha=0.7, label='SIP Inflow')
ax2.plot(sip['month_dt'], sip['active_sip_accounts_crore'], color='#FF7043', linewidth=2.5, marker='o', markersize=3, label='Active Accounts')
ax1.set_title('SIP Inflow vs Active SIP Accounts Growth', fontsize=13, fontweight='bold')
ax1.set_xlabel('Month'); ax1.set_ylabel('SIP Inflow (₹ Crore)', color='#42A5F5')
ax2.set_ylabel('Active Accounts (Crore)', color='#FF7043')
plt.tight_layout()
plt.savefig(str(FIGS/'chart14_sip_dual.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart14_sip_dual.png saved")

# ── Chart 15: Risk Category Distribution ─────────────────────────────────────
print("Chart 15: Risk Distribution...")
risk_cnt = fm['risk_category'].value_counts()
sub_cnt  = fm['sub_category'].value_counts().head(8)
fig, axes = plt.subplots(1, 2, figsize=(14,6))
axes[0].pie(risk_cnt.values, labels=risk_cnt.index, autopct='%1.1f%%',
            colors=sns.color_palette('RdYlGn_r', len(risk_cnt)), startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[0].set_title('Fund Risk Category Distribution', fontweight='bold')
sns.barplot(x=sub_cnt.values, y=sub_cnt.index, palette='viridis', ax=axes[1])
axes[1].set_title('Number of Funds by Sub-Category', fontweight='bold')
axes[1].set_xlabel('Number of Funds')
plt.suptitle('Fund Universe Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(FIGS/'chart15_risk_distribution.png'), dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ chart15_risk_distribution.png saved")

# ── Summary ───────────────────────────────────────────────────────────────────
charts = list(FIGS.glob('chart*.png'))
print(f"\n{'='*50}")
print(f"  ✅ DAY 3 COMPLETE — {len(charts)} charts saved!")
print(f"  Location: {FIGS}")
print(f"{'='*50}")
for c in sorted(charts):
    print(f"  {c.name}")

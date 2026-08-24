"""
mf_app.py — Bluestock MF Capstone Streamlit Dashboard
Run: streamlit run scripts/mf_app.py
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path

RAW  = Path(r'C:\Users\ajays\bluestock_mf_capstone\data\raw')
PROC = Path(r'C:\Users\ajays\bluestock_mf_capstone\data\processed')

st.set_page_config(
    page_title="Bluestock MF Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stMetric {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border-radius: 12px;
        padding: 15px;
        border-left: 4px solid #00d4aa;
        box-shadow: 0 4px 15px rgba(0,212,170,0.2);
    }
    .stMetric label { color: #00d4aa !important; font-weight: bold; }
    .stMetric [data-testid="stMetricValue"] { color: white !important; font-size: 28px !important; }
    .big-title {
        font-size: 42px;
        font-weight: 900;
        background: linear-gradient(90deg, #00d4aa, #0088ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
    }
    .subtitle {
        text-align: center;
        color: #8892b0;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .insight-box {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #0088ff;
        margin: 10px 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def get_col(df, keywords):
    for kw in keywords:
        for col in df.columns:
            if kw.lower() in col.lower():
                return col
    return df.columns[0]

st.sidebar.markdown("""
<div style='text-align:center; padding:10px;'>
    <h2 style='color:#00d4aa;'>📈 Bluestock MF</h2>
    <p style='color:#8892b0;'>Mutual Fund Analytics Platform</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio("🗂️ Navigate", [
    "🏠 Home",
    "📊 Industry Overview",
    "📈 NAV Trends",
    "🏆 Fund Performance",
    "👥 Investor Analytics",
    "📉 Risk Metrics",
    "🤖 Fund Recommender",
    "💰 SIP Calculator",
    "📋 Data Explorer",
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background:#1e3a5f;border-radius:8px;padding:10px;'>
<p style='color:#00d4aa;font-weight:bold;'>🔗 Project Links</p>
<a href='https://github.com/ajaysaketh/bluestock_mf_capstone' style='color:white;'>📁 GitHub Repo</a>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_all():
    fm = pd.read_csv(RAW / '01_fund_master.csv')
    fm.columns = [c.strip() for c in fm.columns]
    fm['amfi_code'] = fm[get_col(fm, ['amfi_code','code'])].astype(str)

    nav = pd.read_csv(RAW / '02_nav_history.csv')
    nav.columns = [c.strip() for c in nav.columns]
    nav['amfi_code'] = nav[get_col(nav, ['amfi_code','code'])].astype(str)
    nav['date'] = pd.to_datetime(nav[get_col(nav, ['date'])], errors='coerce', dayfirst=True)

    aum = pd.read_csv(RAW / '03_aum_by_fund_house.csv')
    aum.columns = [c.strip() for c in aum.columns]
    date_col = get_col(aum, ['quarter','date','month'])
    aum['quarter'] = pd.to_datetime(aum[date_col], errors='coerce', dayfirst=True)
    aum['year'] = aum['quarter'].dt.year
    aum['fund_house'] = aum[get_col(aum, ['fund_house','amc','fund'])]
    aum['aum_crore'] = pd.to_numeric(aum[get_col(aum, ['aum','crore'])], errors='coerce')

    sip = pd.read_csv(RAW / '04_monthly_sip_inflows.csv')
    sip.columns = [c.strip() for c in sip.columns]
    month_col = get_col(sip, ['month','date'])
    try:
        sip['month_dt'] = pd.to_datetime(sip[month_col] + '-01', errors='coerce')
    except:
        sip['month_dt'] = pd.to_datetime(sip[month_col], errors='coerce')
    sip['sip_inflow_crore'] = pd.to_numeric(
        sip[get_col(sip, ['inflow','sip_inflow','crore'])], errors='coerce')

    tx = pd.read_csv(RAW / '08_investor_transactions.csv')
    tx.columns = [c.strip() for c in tx.columns]

    port = pd.read_csv(RAW / '09_portfolio_holdings.csv')
    port.columns = [c.strip() for c in port.columns]

    score = pd.read_csv(PROC / 'fund_scorecard.csv')
    score.columns = [c.strip() for c in score.columns]
    score['amfi_code'] = score[get_col(score, ['amfi_code','code'])].astype(str)

    sharpe = pd.read_csv(PROC / 'sharpe_values.csv')
    sharpe.columns = [c.strip() for c in sharpe.columns]
    sharpe['amfi_code'] = sharpe[get_col(sharpe, ['amfi_code','code'])].astype(str)

    cagr = pd.read_csv(PROC / 'cagr_report.csv')
    cagr.columns = [c.strip() for c in cagr.columns]
    cagr['amfi_code'] = cagr[get_col(cagr, ['amfi_code','code'])].astype(str)

    var_df = pd.read_csv(PROC / 'var_cvar_report.csv')
    var_df.columns = [c.strip() for c in var_df.columns]
    var_df['amfi_code'] = var_df[get_col(var_df, ['amfi_code','code'])].astype(str)

    merge_cols = ['amfi_code'] + [c for c in ['scheme_name','sub_category','fund_house'] if c in fm.columns]
    nav = nav.merge(fm[merge_cols], on='amfi_code', how='left')

    return fm, nav, aum, sip, tx, port, score, sharpe, cagr, var_df

with st.spinner("🚀 Loading Bluestock MF Analytics Platform..."):
    fm, nav, aum, sip, tx, port, score, sharpe, cagr, var_df = load_all()

# ── HOME ──────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<div class="big-title">📈 Bluestock Fintech</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Mutual Fund Analytics Platform — End-to-End Data Engineering & Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🏦 Total AUM",    "₹81 Lakh Cr",  "+18% YoY")
    col2.metric("💰 SIP Inflow",   "₹31,002 Cr",   "All-Time High")
    col3.metric("👥 Total Folios", "26.12 Cr",      "+15% YoY")
    col4.metric("📊 Schemes",      f"{len(fm)}",    "Tracked")
    col5.metric("📅 NAV History",  "4.5 Years",     "2022-2026")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🏆 Top Performing Fund</h4>
        <h2 style='color:#00d4aa;'>Axis Midcap</h2>
        <p>Composite Score: 88.5/100</p>
        <p>3yr CAGR: 23.5% | Sharpe: 1.27</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>📈 SIP Growth Story</h4>
        <h2 style='color:#0088ff;'>3× Growth</h2>
        <p>₹11,000 Cr (Jan 2022) → ₹31,002 Cr (Dec 2025)</p>
        <p>CAGR: ~30% in 4 years</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="insight-box">
        <h4>🏦 Largest AMC</h4>
        <h2 style='color:#ff6b6b;'>SBI MF</h2>
        <p>AUM: ₹12.5 Lakh Crore</p>
        <p>20% ahead of ICICI at ₹10.7L Cr</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    score_col = get_col(score, ['composite_score','score'])
    name_col  = get_col(score, ['scheme_name','name'])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 SIP Inflow Growth")
        fig, ax = plt.subplots(figsize=(7,3))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#1e2130')
        ax.fill_between(sip['month_dt'], sip['sip_inflow_crore'], alpha=0.4, color='#00d4aa')
        ax.plot(sip['month_dt'], sip['sip_inflow_crore'], color='#00d4aa', linewidth=2.5)
        ax.set_title('Monthly SIP Inflows (₹ Crore)', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("🏆 Top 5 Funds by Score")
        top5 = score.nlargest(5, score_col)
        fig, ax = plt.subplots(figsize=(7,3))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#1e2130')
        bar_colors = ['#00d4aa','#0088ff','#ff6b6b','#ffa726','#ab47bc']
        bars = ax.barh(
            [str(n).split(' Fund')[0][:20] for n in top5[name_col]],
            top5[score_col], color=bar_colors, edgecolor='#0E1117')
        for bar, val in zip(bars, top5[score_col]):
            ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}', va='center', color='white', fontsize=9, fontweight='bold')
        ax.set_title('Fund Scorecard — Top 5', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("🏆 Top 10 Funds — Live Scorecard")
    show_cols = [c for c in score.columns if c in
        ['rank','scheme_name','sub_category','composite_score',
         'cagr_3yr','sharpe_ratio','max_drawdown_pct']]
    st.dataframe(score.head(10)[show_cols].style.background_gradient(
        subset=[score_col] if score_col in show_cols else None,
        cmap='RdYlGn'), use_container_width=True)

# ── INDUSTRY OVERVIEW ─────────────────────────────────────────
elif page == "📊 Industry Overview":
    st.markdown('<h1 style="color:#00d4aa;">📊 Industry Overview</h1>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Industry AUM",  "₹81 Lakh Cr", "Dec 2025")
    col2.metric("Active SIPs",   "9.35 Cr",     "Accounts")
    col3.metric("SIP ATH",       "₹31,002 Cr",  "Dec 2025")

    st.subheader("📊 AUM by Fund House")
    years = sorted(aum['year'].dropna().unique().astype(int).tolist(), reverse=True)
    selected_year = st.selectbox("Select Year", years)
    aum_yr = aum.groupby(['fund_house','year'])['aum_crore'].max().reset_index()
    aum_yr['aum_lakh_cr'] = aum_yr['aum_crore'] / 100000
    aum_filtered = aum_yr[aum_yr['year']==selected_year].sort_values('aum_lakh_cr', ascending=False)

    fig, ax = plt.subplots(figsize=(12,5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#1e2130')
    labels = [str(x).replace(' Mutual Fund','').replace(' Prudential','')[:15]
              for x in aum_filtered['fund_house']]
    bars = ax.bar(labels, aum_filtered['aum_lakh_cr'],
                  color=sns.color_palette('cool', len(aum_filtered)), edgecolor='#0E1117')
    for bar, val in zip(bars, aum_filtered['aum_lakh_cr']):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'₹{val:.1f}L', ha='center', fontsize=8, fontweight='bold', color='white')
    ax.set_title(f'AUM by Fund House — {selected_year}', fontweight='bold', color='white')
    ax.set_ylabel('AUM (₹ Lakh Crore)', color='white')
    ax.tick_params(axis='x', rotation=30, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig); plt.close()

    st.subheader("📈 Monthly SIP Inflow Trend")
    fig, ax = plt.subplots(figsize=(12,4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#1e2130')
    ax.fill_between(sip['month_dt'], sip['sip_inflow_crore'], alpha=0.3, color='#00d4aa')
    ax.plot(sip['month_dt'], sip['sip_inflow_crore'],
            color='#00d4aa', linewidth=2.5, marker='o', markersize=3)
    ax.plot(sip['month_dt'], sip['sip_inflow_crore'].rolling(3).mean(),
            color='#ff6b6b', linewidth=2, linestyle='--', label='3M Rolling Avg')
    peak_idx = sip['sip_inflow_crore'].idxmax()
    peak = sip.loc[peak_idx]
    ax.annotate(f"🏆 ATH ₹{peak['sip_inflow_crore']:,.0f} Cr",
                xy=(peak['month_dt'], peak['sip_inflow_crore']),
                xytext=(peak['month_dt'], peak['sip_inflow_crore']-5000),
                arrowprops=dict(arrowstyle='->', color='#ff6b6b'),
                fontsize=10, color='#ff6b6b', fontweight='bold')
    ax.set_title('Monthly SIP Inflows (₹ Crore)', fontweight='bold', color='white')
    ax.set_ylabel('SIP Inflow (₹ Crore)', color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444'); ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#1e2130', labelcolor='white')
    st.pyplot(fig); plt.close()

# ── NAV TRENDS ────────────────────────────────────────────────
elif page == "📈 NAV Trends":
    st.markdown('<h1 style="color:#00d4aa;">📈 NAV Trend Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")

    name_col = get_col(nav, ['scheme_name','name'])
    sub_col  = get_col(fm,  ['sub_category','category'])

    categories = fm[sub_col].dropna().unique().tolist() if sub_col in fm.columns else []
    selected_cat = st.selectbox("Select Category", categories)
    fm_name = get_col(fm, ['scheme_name','name'])
    cat_funds = fm[fm[sub_col]==selected_cat][fm_name].tolist() if sub_col in fm.columns else []
    selected_funds = st.multiselect("Select Funds (max 5)", cat_funds, default=cat_funds[:3])

    if selected_funds:
        nav_filtered = nav[nav[name_col].isin(selected_funds)]
        fig, ax = plt.subplots(figsize=(12,5))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#1e2130')
        nav_col = get_col(nav, ['nav'])
        colors_list = ['#00d4aa','#0088ff','#ff6b6b','#ffa726','#ab47bc']
        for i, fund in enumerate(selected_funds):
            grp = nav_filtered[nav_filtered[name_col]==fund].sort_values('date')
            if len(grp) > 0:
                norm = grp[nav_col] / grp[nav_col].iloc[0] * 100
                ax.plot(grp['date'], norm, label=str(fund)[:25],
                        linewidth=2.5, color=colors_list[i % len(colors_list)])
        ax.axvspan(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-12-31'),
                   alpha=0.15, color='green', label='2023 Bull Run')
        ax.axvspan(pd.Timestamp('2024-06-01'), pd.Timestamp('2024-10-31'),
                   alpha=0.15, color='red', label='2024 Correction')
        ax.set_title('NAV Growth Indexed to 100', fontweight='bold', color='white')
        ax.set_ylabel('NAV Index (Base=100)', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444'); ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8, facecolor='#1e2130', labelcolor='white')
        st.pyplot(fig); plt.close()

# ── FUND PERFORMANCE ──────────────────────────────────────────
elif page == "🏆 Fund Performance":
    st.markdown('<h1 style="color:#00d4aa;">🏆 Fund Performance & Scorecard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    score_col = get_col(score, ['composite_score','score'])
    name_col  = get_col(score, ['scheme_name','name'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🥇 Top Fund",    str(score.iloc[0][name_col]).split(' Fund')[0][:20])
    col2.metric("⭐ Best Score",  f"{score.iloc[0][score_col]:.1f}/100")
    col3.metric("📊 Total Funds", f"{len(score)}")
    col4.metric("📈 Categories",  f"{score['sub_category'].nunique() if 'sub_category' in score.columns else 'N/A'}")

    st.subheader("📋 Fund Scorecard")
    sub_cats = ['All'] + (score['sub_category'].dropna().unique().tolist()
                          if 'sub_category' in score.columns else [])
    selected_sub = st.selectbox("Filter by Category", sub_cats)
    display_score = score if selected_sub == 'All' else score[score['sub_category']==selected_sub]
    show_cols = [c for c in score.columns if c in
        ['rank','scheme_name','sub_category','composite_score',
         'cagr_3yr','sharpe_ratio','max_drawdown_pct','alpha','beta']]
    st.dataframe(
        display_score.head(20)[show_cols].style.background_gradient(
            subset=[score_col] if score_col in show_cols else None, cmap='RdYlGn'),
        use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Top 10 by Score")
        top10 = score.nlargest(10, score_col)
        fig, ax = plt.subplots(figsize=(8,5))
        fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#1e2130')
        colors = plt.cm.RdYlGn(np.linspace(0.3,0.9,10))
        bars = ax.barh([str(n).split(' Fund')[0][:22] for n in top10[name_col]],
                       top10[score_col], color=colors)
        for bar, val in zip(bars, top10[score_col]):
            ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}', va='center', fontsize=9, color='white', fontweight='bold')
        ax.set_title('Top 10 Funds', fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444'); ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("📊 CAGR Comparison")
        cagr_cols = [c for c in cagr.columns if 'cagr' in c.lower() or 'return' in c.lower()]
        cagr_name = get_col(cagr, ['scheme_name','name'])
        if cagr_cols:
            top10_cagr = cagr.nlargest(8, cagr_cols[0])
            fig, ax = plt.subplots(figsize=(8,5))
            fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#1e2130')
            x = np.arange(len(top10_cagr)); w = 0.25
            bar_colors = ['#00d4aa','#0088ff','#ff6b6b']
            for i, (col, color) in enumerate(zip(cagr_cols[:3], bar_colors)):
                ax.bar(x+i*w, pd.to_numeric(top10_cagr[col], errors='coerce'),
                       w, label=col, color=color, edgecolor='#0E1117')
            ax.set_xticks(x+w)
            ax.set_xticklabels([str(n).split(' Fund')[0][:12]
                                for n in top10_cagr[cagr_name]], rotation=30, color='white')
            ax.set_ylabel('CAGR (%)', color='white')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('#444'); ax.spines['left'].set_color('#444')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.legend(facecolor='#1e2130', labelcolor='white')
            ax.set_title('CAGR Comparison', fontweight='bold', color='white')
            st.pyplot(fig); plt.close()

# ── INVESTOR ANALYTICS ────────────────────────────────────────
elif page == "👥 Investor Analytics":
    st.markdown('<h1 style="color:#00d4aa;">👥 Investor Analytics</h1>', unsafe_allow_html=True)
    st.markdown("---")

    tx_type_col   = get_col(tx, ['transaction_type','type'])
    amount_col    = get_col(tx, ['amount','amount_inr'])
    age_col       = get_col(tx, ['age_group','age'])
    state_col     = get_col(tx, ['state'])
    city_tier_col = get_col(tx, ['city_tier','tier'])
    gender_col    = get_col(tx, ['gender'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Investors",    "5,000")
    col2.metric("💳 Transactions", f"{len(tx):,}")
    col3.metric("💰 Avg Amount",   f"₹{pd.to_numeric(tx[amount_col],errors='coerce').mean():,.0f}")
    col4.metric("📍 States",       f"{tx[state_col].nunique()}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💳 Transaction Type Split")
        tx_cnt = tx[tx_type_col].value_counts()
        fig, ax = plt.subplots(figsize=(6,5))
        fig.patch.set_facecolor('#0E1117')
        ax.pie(tx_cnt.values, labels=tx_cnt.index, autopct='%1.1f%%',
               colors=['#00d4aa','#0088ff','#ff6b6b'],
               wedgeprops={'edgecolor':'#0E1117','linewidth':3},
               textprops={'color':'white'})
        ax.set_title('Transaction Distribution', fontweight='bold', color='white')
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("👶 Age Group Distribution")
        age_cnt = tx[age_col].value_counts()
        fig, ax = plt.subplots(figsize=(6,5))
        fig.patch.set_facecolor('#0E1117')
        ax.pie(age_cnt.values, labels=age_cnt.index, autopct='%1.1f%%',
               colors=sns.color_palette('cool', len(age_cnt)),
               wedgeprops={'edgecolor':'#0E1117','linewidth':3},
               textprops={'color':'white'})
        ax.set_title('Age Distribution', fontweight='bold', color='white')
        st.pyplot(fig); plt.close()

    st.subheader("🗺️ Investment by State")
    state_amt = tx.groupby(state_col)[amount_col].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12,4))
    fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#1e2130')
    ax.bar(state_amt.index, state_amt.values/1e7,
           color=sns.color_palette('cool', len(state_amt)), edgecolor='#0E1117')
    ax.set_title('Investment Amount by State (₹ Crore)', fontweight='bold', color='white')
    ax.set_ylabel('Amount (₹ Crore)', color='white')
    ax.tick_params(axis='x', rotation=30, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('#444'); ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏙️ T30 vs B30")
        tier_cnt = tx[city_tier_col].value_counts()
        fig, ax = plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor('#0E1117')
        ax.pie(tier_cnt.values, labels=tier_cnt.index, autopct='%1.1f%%',
               colors=['#00d4aa','#0088ff'],
               wedgeprops={'edgecolor':'#0E1117','linewidth':3},
               textprops={'color':'white'})
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("👫 Gender Split")
        gen_cnt = tx[gender_col].value_counts()
        fig, ax = plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor('#0E1117')
        ax.pie(gen_cnt.values, labels=gen_cnt.index, autopct='%1.1f%%',
               colors=['#0088ff','#ff6b6b'],
               wedgeprops={'edgecolor':'#0E1117','linewidth':3},
               textprops={'color':'white'})
        st.pyplot(fig); plt.close()

# ── RISK METRICS ──────────────────────────────────────────────
elif page == "📉 Risk Metrics":
    st.markdown('<h1 style="color:#00d4aa;">📉 Risk Metrics & Advanced Analytics</h1>', unsafe_allow_html=True)
    st.markdown("---")

    sharpe_col = get_col(sharpe, ['sharpe_ratio','sharpe'])
    name_col   = get_col(sharpe, ['scheme_name','name'])
    var_col    = get_col(var_df, ['var_95_daily','var','cvar'])
    score_col  = get_col(score, ['composite_score','score'])
    vol_col    = get_col(sharpe, ['volatility','vol','std'])

    sharpe_vals = pd.to_numeric(sharpe[sharpe_col], errors='coerce')
    var_vals    = pd.to_numeric(var_df[var_col], errors='coerce')

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Best Sharpe",   f"{sharpe_vals.max():.2f}")
    col2.metric("📉 Avg Sharpe",    f"{sharpe_vals.mean():.2f}")
    col3.metric("⚠️ Avg VaR (95%)", f"{var_vals.mean():.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Top 10 by Sharpe Ratio")
        top_sharpe = sharpe.nlargest(10, sharpe_col)
        fig, ax = plt.subplots(figsize=(8,5))
        fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#1e2130')
        ax.barh([str(n).split(' Fund')[0][:22] for n in top_sharpe[name_col]],
                pd.to_numeric(top_sharpe[sharpe_col], errors='coerce'),
                color=sns.color_palette('cool', 10), edgecolor='#0E1117')
        ax.set_title('Sharpe Ratio — Top 10', fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444'); ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("⚠️ VaR Distribution")
        fig, ax = plt.subplots(figsize=(8,5))
        fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#1e2130')
        ax.hist(var_vals.dropna(), bins=15, color='#ff6b6b', edgecolor='#0E1117', linewidth=1.2)
        ax.axvline(var_vals.mean(), color='#00d4aa', linestyle='--', linewidth=2,
                   label=f"Mean={var_vals.mean():.2f}%")
        ax.set_title('Daily VaR (95%) Distribution', fontweight='bold', color='white')
        ax.set_xlabel('Daily VaR (%)', color='white')
        ax.set_ylabel('Number of Funds', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444'); ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(facecolor='#1e2130', labelcolor='white')
        st.pyplot(fig); plt.close()

    st.subheader("🎯 Risk vs Return Scatter")
    ret_col = get_col(score, ['cagr_3yr','return','cagr'])
    merged  = score.merge(sharpe[['amfi_code', vol_col]], on='amfi_code', how='left')
    m_vol   = pd.to_numeric(merged[vol_col], errors='coerce') if vol_col in merged.columns else pd.Series([0]*len(merged))
    m_sco   = pd.to_numeric(merged[score_col], errors='coerce')
    m_ret   = pd.to_numeric(merged[ret_col], errors='coerce') if ret_col in merged.columns else m_sco

    fig, ax = plt.subplots(figsize=(12,5))
    fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#1e2130')
    scatter = ax.scatter(m_vol, m_ret, s=m_sco*5, c=m_sco,
                         cmap='RdYlGn', alpha=0.8, edgecolors='white', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Composite Score', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    ax.set_xlabel('Volatility (%)', color='white')
    ax.set_ylabel('Return (%)', color='white')
    ax.set_title('Risk vs Return (bubble size = composite score)', fontweight='bold', color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444'); ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    st.pyplot(fig); plt.close()

    if 'sector' in port.columns and 'weight_pct' in port.columns:
        st.subheader("🥧 Sector Allocation")
        sector_wt = port.groupby('sector')['weight_pct'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_facecolor('#0E1117')
        ax.pie(sector_wt.values, labels=sector_wt.index, autopct='%1.1f%%',
               colors=sns.color_palette('cool', len(sector_wt)),
               wedgeprops=dict(width=0.55, edgecolor='#0E1117', linewidth=3),
               textprops={'color':'white'}, startangle=90)
        ax.set_title('Sector Allocation — Equity Funds', fontweight='bold', color='white')
        st.pyplot(fig); plt.close()

# ── FUND RECOMMENDER ──────────────────────────────────────────
elif page == "🤖 Fund Recommender":
    st.markdown('<h1 style="color:#00d4aa;">🤖 AI Fund Recommender</h1>', unsafe_allow_html=True)
    st.markdown("---")
    st.info("🎯 Get personalised fund recommendations based on your investor profile!")

    col1, col2, col3 = st.columns(3)
    with col1:
        risk = st.selectbox("⚠️ Risk Appetite", ['Low','Moderate','High'])
    with col2:
        category = st.selectbox("📂 Category", ['All','Equity','Debt','Hybrid'])
    with col3:
        top_n = st.slider("🔢 Recommendations", 1, 5, 3)

    if st.button("🔍 Get My Recommendations", type="primary", use_container_width=True):
        risk_map   = {'Low':['Low','Moderate'],'Moderate':['Moderate','High'],'High':['High','Very High']}
        risk_col   = get_col(fm, ['risk_category','risk'])
        cat_col    = get_col(fm, ['category'])
        name_col   = get_col(fm, ['scheme_name','name'])
        score_col  = get_col(score, ['composite_score','score'])
        sharpe_col = get_col(sharpe, ['sharpe_ratio','sharpe'])

        filtered = fm.copy()
        if risk_col in filtered.columns:
            filtered = filtered[filtered[risk_col].isin(risk_map[risk])]
        if category != 'All' and cat_col in filtered.columns:
            filtered = filtered[filtered[cat_col]==category]

        filtered = filtered.merge(score[['amfi_code',score_col]], on='amfi_code', how='left')
        filtered = filtered.dropna(subset=[score_col])
        filtered = filtered.merge(sharpe[['amfi_code',sharpe_col]], on='amfi_code', how='left')
        top = filtered.nlargest(top_n, score_col)

        if len(top) == 0:
            st.warning("⚠️ No funds found. Try different filters.")
        else:
            st.success(f"✅ Top {top_n} Funds for **{risk} Risk** investor:")
            for i, (_, row) in enumerate(top.iterrows()):
                medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][i]
                with st.expander(f"{medal} {row[name_col]}", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Score",   f"{row[score_col]:.1f}/100")
                    c2.metric("Sharpe",  f"{row.get(sharpe_col,'N/A')}")
                    c3.metric("Risk",    f"{row.get(risk_col,'N/A')}")
                    c4.metric("Expense", f"{row.get('expense_ratio_pct','N/A')}%")
                    st.write(f"**🏦 Fund House:** {row.get('fund_house','N/A')}")
                    st.write(f"**📂 Category:** {row.get(cat_col,'N/A')} — {row.get('sub_category','N/A')}")

# ── DATA EXPLORER ─────────────────────────────────────────────

# ── SIP CALCULATOR ────────────────────────────────────────────
elif page == "💰 SIP Calculator":
    st.markdown('<h1 style="color:#00d4aa;">💰 SIP Calculator</h1>', unsafe_allow_html=True)
    st.markdown("---")
    st.info("🎯 Calculate your future wealth with SIP investments!")

    col1, col2, col3 = st.columns(3)
    with col1:
        monthly_sip = st.number_input("Monthly SIP Amount (₹)", min_value=500, max_value=100000, value=5000, step=500)
    with col2:
        inv_years = st.slider("Investment Period (Years)", 1, 30, 10)
    with col3:
        expected_return = st.slider("Expected Annual Return (%)", 5, 30, 12)

    if st.button("📊 Calculate Returns", type="primary", use_container_width=True):
        months = inv_years * 12
        monthly_rate = expected_return / 100 / 12
        future_value = monthly_sip * (((1 + monthly_rate)**months - 1) / monthly_rate) * (1 + monthly_rate)
        total_invested = monthly_sip * months
        total_returns  = future_value - total_invested

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Total Invested", f"₹{total_invested/1e5:.2f} L")
        col2.metric("📈 Future Value",   f"₹{future_value/1e5:.2f} L")
        col3.metric("💵 Total Returns",  f"₹{total_returns/1e5:.2f} L")
        col4.metric("🚀 Wealth Gained",  f"{total_returns/total_invested*100:.1f}%")

        values   = []
        invested = []
        for m in range(1, months+1):
            fv = monthly_sip * (((1+monthly_rate)**m - 1)/monthly_rate) * (1+monthly_rate)
            values.append(fv/1e5)
            invested.append(monthly_sip*m/1e5)

        fig, ax = plt.subplots(figsize=(12,5))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#1e2130')
        ax.fill_between(range(1,months+1), values,   alpha=0.4, color='#00d4aa', label='Future Value')
        ax.fill_between(range(1,months+1), invested, alpha=0.4, color='#0088ff', label='Amount Invested')
        ax.plot(range(1,months+1), values,   color='#00d4aa', linewidth=2.5)
        ax.plot(range(1,months+1), invested, color='#0088ff', linewidth=2.5)
        ax.set_title(f'SIP Growth — ₹{monthly_sip:,}/month for {inv_years} years @ {expected_return}%',
                     fontweight='bold', color='white')
        ax.set_xlabel('Months', color='white')
        ax.set_ylabel('Amount (₹ Lakh)', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444'); ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False);   ax.spines['right'].set_visible(False)
        ax.legend(facecolor='#1e2130', labelcolor='white')
        st.pyplot(fig); plt.close()

        st.success(f"🎉 ₹{monthly_sip:,}/month for {inv_years} years grows to ₹{future_value/1e5:.2f} Lakhs!")

        # Comparison table
        st.subheader("📊 Comparison at Different Returns")
        rows = []
        for r in [8, 10, 12, 15, 18, 20]:
            mr = r/100/12
            fv = monthly_sip * (((1+mr)**months-1)/mr) * (1+mr)
            rows.append({'Return %': f'{r}%', 'Future Value': f'₹{fv/1e5:.2f} L',
                         'Total Returns': f'₹{(fv-total_invested)/1e5:.2f} L',
                         'Wealth Gained': f'{(fv-total_invested)/total_invested*100:.1f}%'})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


elif page == "📋 Data Explorer":
    st.markdown('<h1 style="color:#00d4aa;">📋 Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown("---")

    dataset = st.selectbox("Select Dataset", [
        "Fund Master","NAV History","AUM Data","SIP Inflows",
        "Investor Transactions","Fund Scorecard","Sharpe Values","CAGR Report","VaR Report"
    ])

    data_map = {
        "Fund Master":           fm,
        "NAV History":           nav.head(1000),
        "AUM Data":              aum,
        "SIP Inflows":           sip,
        "Investor Transactions": tx.head(1000),
        "Fund Scorecard":        score,
        "Sharpe Values":         sharpe,
        "CAGR Report":           cagr,
        "VaR Report":            var_df,
    }

    selected_df = data_map[dataset]
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows",    f"{len(selected_df):,}")
    col2.metric("Columns", f"{selected_df.shape[1]}")
    col3.metric("Dataset", dataset)

    st.dataframe(selected_df, use_container_width=True)
    st.subheader("📊 Column Statistics")
    st.dataframe(selected_df.describe(), use_container_width=True)

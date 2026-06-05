Data Dictionary — Bluestock MF Capstone
dim_fund — Fund Master (40 rows)
ColumnTypeDescriptionamfi_codeTEXTAMFI unique scheme code (PK)scheme_nameTEXTFull official AMFI scheme namefund_houseTEXTAMC name (e.g. SBI Mutual Fund)categoryTEXTEquity / Debt / Hybridsub_categoryTEXTLarge Cap / Mid Cap / Small Cap / Liquid etcplanTEXTDirect or Regularlaunch_dateDATEFund launch date (YYYY-MM-DD)benchmarkTEXTOfficial benchmark indexexpense_ratio_pctREALAnnual expense ratio % (e.g. 1.05)exit_load_pctREALExit load % (0 for Liquid/Index funds)fund_managerTEXTName of primary fund managerrisk_categoryTEXTSEBI risk: Low / Moderate / High / Very Highsebi_category_codeTEXTEC01=LargeCap, EC03=SmallCap, DC01=Liquid
fact_nav — Daily NAV History (~46,000 rows)
ColumnTypeDescriptionamfi_codeTEXTForeign key to dim_funddateDATENAV date (business days only, YYYY-MM-DD)navREALNAV in Rs. anchored to real mfapi.in values
fact_aum — AUM by Fund House (160 rows)
ColumnTypeDescriptionfund_houseTEXTAMC namequarterDATEQuarter end dateaum_croreREALAUM in Rs. crorenum_schemesINTNumber of schemes
fact_sip_industry — Monthly SIP Inflows (48 rows)
ColumnTypeDescriptionmonthTEXTYYYY-MM formatsip_inflow_croreREALTotal SIP inflows in Rs. croreactive_sip_accounts_croreREALActive SIP accounts in crorenew_sip_accounts_lakhREALNew SIP registrations in lakhsip_aum_lakh_croreREALSIP AUM in Rs. lakh croreyoy_growth_pctREALYoY growth % in SIP inflows
fact_transactions — Investor Transactions (~42,000 rows)
ColumnTypeDescriptioninvestor_idTEXTUnique investor ID (INV000001–INV005000)transaction_dateDATEDate of transactionamfi_codeTEXTFund code (FK to dim_fund)transaction_typeTEXTSIP / Lumpsum / Redemptionamount_inrINTTransaction amount in Rs.stateTEXTInvestor's statecityTEXTInvestor's citycity_tierTEXTT30 (Top 30 cities) or B30 (Beyond Top 30)age_groupTEXT18-25 / 26-35 / 36-45 / 46-55 / 56+genderTEXTMale / Femaleannual_income_lakhREALAnnual income in Rs. lakhpayment_modeTEXTUPI / Net Banking / Mandate / Chequekyc_statusTEXTVerified (92%) / Pending (8%)
fact_performance — Scheme Performance (40 rows)
ColumnTypeDescriptionamfi_codeTEXTFK to dim_fundreturn_1yr_pctREAL1-year absolute return %return_3yr_pctREAL3-year CAGR %return_5yr_pctREAL5-year CAGR %benchmark_3yr_pctREALBenchmark 3yr CAGR %alphaREALReturn above benchmarkbetaREALMarket sensitivity (1.0 = same as market)sharpe_ratioREALRisk-adjusted return (higher = better)sortino_ratioREALDownside-only Sharpestd_dev_ann_pctREALAnnualised standard deviation %max_drawdown_pctREALWorst peak-to-trough decline (negative)morningstar_ratingINT1–5 star rating
fact_portfolio — Portfolio Holdings (282 rows)
ColumnTypeDescriptionamfi_codeTEXTFK to dim_fundreport_dateDATEHoldings as of datestock_symbolTEXTNSE/BSE ticker symbolstock_nameTEXTCompany namesectorTEXTSector (IT, Banking, FMCG etc)weight_pctREALPortfolio weight %
fact_benchmark — Benchmark Indices (1,150 rows)
ColumnTypeDescriptiondateDATETrading dateNifty50REALNifty 50 closing valueNifty100REALNifty 100 closing valueNiftyMidcap150REALNifty Midcap 150 closing valueBSESmallCapREALBSE SmallCap closing valueCRISILLiquidREALCRISIL Liquid Fund IndexCRISILGiltREALCRISIL Gilt Index
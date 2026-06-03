-- ============================================================
-- schema.sql  —  Bluestock MF Capstone · SQLite Schema
-- ============================================================
-- Usage:
--   sqlite3 data/db/bluestock_mf.db < sql/schema.sql
-- ============================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ──────────────────────────────────────────────────────────
-- 1. fund_master — one row per scheme
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fund_master (
    scheme_code     INTEGER PRIMARY KEY,   -- AMFI scheme code
    scheme_name     TEXT    NOT NULL,
    fund_house      TEXT    NOT NULL,
    category        TEXT,                  -- e.g. Equity
    sub_category    TEXT,                  -- e.g. Large Cap
    scheme_type     TEXT,                  -- Direct / Regular
    risk_grade      TEXT,                  -- Low / Moderate / High / Very High
    launch_date     TEXT,                  -- ISO-8601 date string
    aum_crore       REAL,                  -- Assets Under Management (₹ Cr)
    expense_ratio   REAL,                  -- Annual expense ratio (%)
    benchmark       TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────────────
-- 2. nav_history — daily NAV per scheme
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nav_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_code     INTEGER NOT NULL,
    nav_date        TEXT    NOT NULL,      -- ISO-8601: YYYY-MM-DD
    nav             REAL    NOT NULL,
    UNIQUE (scheme_code, nav_date),
    FOREIGN KEY (scheme_code) REFERENCES fund_master(scheme_code)
);

CREATE INDEX IF NOT EXISTS idx_nav_scheme_date
    ON nav_history (scheme_code, nav_date DESC);

-- ──────────────────────────────────────────────────────────
-- 3. performance_metrics — computed from nav_history
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS performance_metrics (
    scheme_code     INTEGER NOT NULL,
    as_of_date      TEXT    NOT NULL,
    return_1m       REAL,   -- 1-month return (%)
    return_3m       REAL,
    return_6m       REAL,
    return_1y       REAL,
    return_3y       REAL,   -- annualised
    return_5y       REAL,   -- annualised
    cagr_inception  REAL,   -- CAGR since inception
    volatility_1y   REAL,   -- annualised std dev (252 trading days)
    sharpe_1y       REAL,
    beta_1y         REAL,
    max_drawdown_1y REAL,
    PRIMARY KEY (scheme_code, as_of_date),
    FOREIGN KEY (scheme_code) REFERENCES fund_master(scheme_code)
);

-- ──────────────────────────────────────────────────────────
-- 4. aum_data — monthly AUM snapshots
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS aum_data (
    scheme_code     INTEGER NOT NULL,
    report_date     TEXT    NOT NULL,
    aum_crore       REAL,
    PRIMARY KEY (scheme_code, report_date),
    FOREIGN KEY (scheme_code) REFERENCES fund_master(scheme_code)
);

-- ──────────────────────────────────────────────────────────
-- 5. portfolio_data — top holdings per scheme
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS portfolio_data (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_code     INTEGER NOT NULL,
    report_date     TEXT    NOT NULL,
    stock_name      TEXT,
    isin            TEXT,
    sector          TEXT,
    weight_pct      REAL,
    FOREIGN KEY (scheme_code) REFERENCES fund_master(scheme_code)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_scheme_date
    ON portfolio_data (scheme_code, report_date);

-- ──────────────────────────────────────────────────────────
-- ETL audit log
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS etl_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at      TEXT DEFAULT (datetime('now')),
    task        TEXT,
    status      TEXT,   -- SUCCESS / ERROR
    rows_loaded INTEGER,
    notes       TEXT
);

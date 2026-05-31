# ÉTAPE 0 (v2) — Multi-Factor Audit Report
## MASI Hybrid Forecasting System (HMM + CNN-LSTM)
**Generated:** 2026-05-19
**Status:** COMPLETE — populated with actual script results
**Data sources:** `data/raw/master_dataset.csv` + `data/raw/masi_raw.csv` (MERGED)

---

## 1. DATA SOURCES & MERGE

### Provenance

| Source | Origin | Coverage | Rows | Role |
|--------|--------|----------|------|------|
| `master_dataset.csv` | External research repo (`masi-risk-research-notebooks-main`) — provided by user | 2007-01-31 → 2026-03-19 | 4,765 | **BASE** — multi-factor, long history |
| `masi_raw.csv` | Investing.com / casablanca-bourse exports | 2016-04-01 → 2026-04-20 | 2,735 | **OVERLAY** — OHLC + most recent obs |

### Merged Output: `data/interim/masi_merged.csv`

| Item | Value |
|------|-------|
| Total observations | **4,786** |
| Date range | 2007-01-31 → 2026-04-20 |
| Columns retained | 14 |
| Extra obs from masi_raw (post master end-date) | 22 |
| Pre-computed derivative cols dropped | 74 |

### Columns Kept (raw, leakage-free)

```
date, masi_close, atw_close, iam_close, lhm_close, mng_close, brent_close, gold_close, eur_mad, gpr_index, bam_policy_rate, masi_open, masi_high, masi_low
```

### Columns Dropped (will be recomputed strictly in Étape 3)

These pre-computed columns were validated as past-only by `leakage_quickcheck.py` (100% pass)
but are nonetheless dropped to enforce a single source-of-truth recomputation pipeline:

```
masi_log_return, masi_realized_vol_5, masi_rolling_mean_5, masi_realized_vol_20, masi_rolling_mean_20, masi_lag_1, masi_lag_2, masi_lag_5, atw_log_return, atw_realized_vol_5, atw_rolling_mean_5, atw_realized_vol_20, atw_rolling_mean_20, atw_lag_1, atw_lag_2, atw_lag_5, iam_log_return, iam_realized_vol_5, iam_rolling_mean_5, iam_realized_vol_20, iam_rolling_mean_20, iam_lag_1, iam_lag_2, iam_lag_5, lhm_log_return, lhm_realized_vol_5, lhm_rolling_mean_5, lhm_realized_vol_20, lhm_rolling_mean_20, lhm_lag_1, lhm_lag_2, lhm_lag_5, mng_log_return, mng_realized_vol_5, mng_rolling_mean_5, mng_realized_vol_20, mng_rolling_mean_20, mng_lag_1, mng_lag_2, mng_lag_5, brent_log_return, brent_realized_vol_5, brent_rolling_mean_5, brent_realized_vol_20, brent_rolling_mean_20, brent_lag_1, brent_lag_2, brent_lag_5, gold_log_return, gold_realized_vol_5, gold_rolling_mean_5, gold_realized_vol_20, gold_rolling_mean_20, gold_lag_1, gold_lag_2, gold_lag_5, eur_mad_log_return, eur_mad_realized_vol_5, eur_mad_rolling_mean_5, eur_mad_realized_vol_20, eur_mad_rolling_mean_20, eur_mad_lag_1, eur_mad_lag_2, eur_mad_lag_5, eur_mad_return, gpr_delta, gpr_index_log_return, gpr_index_realized_vol_5, gpr_index_realized_vol_20, gpr_index_rolling_mean_5, gpr_index_rolling_mean_20, gpr_index_lag_1, gpr_index_lag_2, gpr_index_lag_5
```

### Why this merge?

Per the literature synthesis (Talhartit 2025, Belcaid & El Ghini 2021, Kharbouch & Ouaskou 2023):
- **Multi-factor MASI modeling** (macro + cross-sectional) outperforms univariate
- **Brent + Gold + EUR/MAD + GPR + BAM policy rate** are validated MASI drivers
- **Individual liquid stocks (ATW, IAM, LHM, MNG)** enable optional CSV (cross-sectional vol) proxy
- **2007-2026** captures **2008 + COVID 2020** — two structurally distinct Bear regimes for HMM

---

## 2. DATA STRUCTURE VALIDATION

| Check | Result |
|-------|--------|
| Total rows | 4,786 |
| Total cols | 14 |
| Duplicate dates | 0 |
| Non-positive MASI prices | 0 |
| Chronological order | ✅ enforced |

### Missing Values (top 10 columns)

| Column | Missing | % |
|--------|---------|---|
| masi_open | 2285 | 47.74% |
| masi_high | 2285 | 47.74% |
| masi_low | 2285 | 47.74% |


**Interpretation:** Missing values concentrate in early 2007-2010 (some macro series start later)
and in the OHLC overlay (masi_raw only covers 2016+). These will be handled in Étape 1.

---

## 3. RETURN DISTRIBUTION

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| N log-returns | 4,785 | ✅ Large sample for DL |
| Mean daily | 0.000125 | ~3.16% annualized |
| Std daily | 0.007791 | ~12.37% annualized vol |
| Skewness | -0.8095 | left-tailed |
| Excess kurtosis | 12.3342 | extreme fat tails |
| Jarque-Bera p | 0.0000 | NOT normal ✅ |

---

## 4. STATIONARITY TESTS

| Series | ADF p | KPSS p | Verdict |
|--------|-------|--------|---------|
| Price levels | 0.9453 (non-stationary) | 0.0100 (non-stationary) | — |
| Log-prices | 0.8586 (non-stationary) | 0.0100 (non-stationary) | — |
| Log-returns | 0.0000 (STATIONARY ✅) | 0.1000 (STATIONARY ✅) | — |
| |Log-returns| | 0.0000 (STATIONARY ✅) | 0.0236 (non-stationary) | — |
| Squared log-returns | 0.0000 (STATIONARY ✅) | 0.1000 (STATIONARY ✅) | — |


**Key:** Log-returns I(0) confirmed by both ADF & KPSS → safe target.
Price levels I(1) → never as raw CNN-LSTM input.

---

## 5. ARCH EFFECTS

| Test | Lag | Stat | p-value | Verdict |
|------|-----|------|---------|---------|
| LB on r² | 5 | 2029.67 | 0.0000 | ARCH ✅ |
| LB on r² | 10 | 2307.46 | 0.0000 | ARCH ✅ |
| LB on r² | 20 | 2448.57 | 0.0000 | ARCH ✅ |
| LB on r  | 10 | 174.50 | 0.0000 | autocorr ✅ |

**ARCH detected:** YES ✅

### GARCH(1,1) Parameters

| Param | Value |
|-------|-------|
| ω (omega) | 0.0622 |
| α (alpha) | 0.2548 |
| β (beta)  | 0.6464 |
| α + β (persistence) | **0.9012** |

**Decision:** Primary volatility proxy = **GARCH(1,1)**. Fallback = rolling 21-day realized vol.
Per Korkpoe 2019 (SSA frontier), consider GJR-GARCH / EGARCH with Student-t in Étape 4.

---

## 6. ANOMALY DETECTION

| Threshold | Count |
|-----------|-------|
| \|r\| > 5% | 6 |
| \|r\| > 10% | 0 |
| r == 0 (price unchanged) | 3 |

### Top 10 extreme return days

| Date | Return |
|------|--------|
| 2020-03-16 | -9.23% |
| 2020-03-12 | -6.93% |
| 2020-03-09 | -5.99% |
| 2025-04-07 | -5.81% |
| 2026-03-03 | -5.79% |
| 2020-03-10 | +5.31% |
| 2023-10-05 | +4.95% |
| 2009-01-05 | -4.67% |
| 2007-05-14 | -4.55% |
| 2008-10-27 | -4.49% |


---

## 7. TEMPORAL SPLIT (70/10/20 + 5d gap, L6)

| Set | Date Range | N |
|-----|------------|---|
| TRAIN | 2007-01-31 → 2020-07-10 | 3,350 |
| VAL   | 2020-07-20 → 2022-06-21 | 478 |
| TEST  | 2022-06-29 → 2026-04-20 | 948 |

Gap train→val: **10 calendar days**
Gap val→test: **8 calendar days**
L6 assertions: ✅ PASSED

**Key advantage of merged dataset:** TRAIN now spans 2007-2022, which includes
**both the 2008 financial crisis AND COVID-19 2020** — two structurally distinct
Bear regimes essential for HMM regime identification.

---

## 8. PLOTS GENERATED

| Plot | File |
|------|------|
| audit_plot_1_overview.png | `reports\figures\etape0\audit_plot_1_overview.png` |
| audit_plot_2_acf.png | `reports\figures\etape0\audit_plot_2_acf.png` |
| audit_plot_3_rolling_stats.png | `reports\figures\etape0\audit_plot_3_rolling_stats.png` |
| audit_plot_4_qqplot.png | `reports\figures\etape0\audit_plot_4_qqplot.png` |


---

## 9. ANTI-LEAKAGE INVENTORY

| Rule | Status |
|------|--------|
| **L1** | StandardScaler: fit ONLY on training data. Never on full dataset. |
| **L2** | HMM: train on train set only. Forward-predict on val/test. |
| **L3** | Rolling features: use shift(1) or closed='left'. No centered windows. |
| **L4** | Target variable: compute AFTER features. Never include future return. |
| **L5** | Signal execution: signal at t -> executed at OPEN of t+1. |
| **L6** | Walk-forward gap: minimum L days gap between train end and val start. |
| **L7** | MASI holidays: remove zero-volume days. Forward-fill max 2 days only. |
| **L8** | GARCH proxy: re-estimate in each walk-forward window separately. |


---

## 10. DECISIONS FOR ÉTAPE 1

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Primary input = `data/interim/masi_merged.csv` | Multi-factor + long history |
| 2 | Target y_t = ln(P_{t+1}/P_t) on `masi_close` | Confirmed stationary (ADF p=0.0000) |
| 3 | Multi-factor feature set (~11 raw series) | Literature-validated MASI drivers |
| 4 | Vol proxy = **GARCH(1,1)** | ARCH detected at all lags |
| 5 | Temporal split 70/10/20 + 5d gap | L6 verified ✅ |
| 6 | Forward-fill ≤ 2 days, segment > 5 days | L7 |
| 7 | Drop pre-computed lag/rolling — recompute strict in Étape 3 | L3 enforcement |
| 8 | Input window L ∈ {10, 15, 20} | C3 constraint |

---

## 11. PRE-CONDITIONS FOR ÉTAPE 1 — CHECKLIST

- [x] `etape0_audit.py` (v2) runs without errors
- [x] Merged dataset created at `data/interim/masi_merged.csv`
- [x] ADF on log-returns: stationary
- [x] ADF on prices: non-stationary
- [x] ARCH detected
- [x] GARCH(1,1) parameters estimated, persistence < 1
- [x] 4 audit plots saved
- [x] Temporal split L6 assertions passed
- [x] 8 leakage rules documented

---

*End of Étape 0 (v2) Multi-Factor Audit Report — generated by `scripts/00_data_audit.py`*

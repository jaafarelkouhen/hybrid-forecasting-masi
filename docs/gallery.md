# Visual gallery — all pipeline figures

Every diagnostic figure produced by the pipeline (44 plots total), grouped by
étape. Source files live under `reports/figures/etape*/` in the repo.

---

## Étape 0 — Data audit (4 figures)

Raw MASI series, log-returns, missingness map, plus stationarity / ARCH /
QQ-plot diagnostics that justify the choice of GARCH(1,1) as the volatility
proxy (no high-frequency data available on MASI).

```{image} ../reports/figures/etape0/audit_plot_1_overview.png
:alt: MASI overview
:align: center
```

```{image} ../reports/figures/etape0/audit_plot_2_acf.png
:alt: Returns ACF
:align: center
```

```{image} ../reports/figures/etape0/audit_plot_3_rolling_stats.png
:alt: Rolling stats
:align: center
```

```{image} ../reports/figures/etape0/audit_plot_4_qqplot.png
:alt: QQ-plot
:align: center
```

---

## Étape 1 — Preprocessing & splits (4 figures)

Strict temporal split **TRAIN 2007–2020 / VAL 2020–2022 / TEST 2022–2026** with
8-day gaps between segments — no shuffling, no leakage across boundaries.

```{image} ../reports/figures/etape1/etape1_split_overview.png
:alt: Split overview
:align: center
```

```{image} ../reports/figures/etape1/etape1_target_distribution.png
:alt: Target distribution
:align: center
```

```{image} ../reports/figures/etape1/etape1_factor_overview.png
:alt: Factor overview
:align: center
```

```{image} ../reports/figures/etape1/etape1_scaler_diagnostic.png
:alt: Scaler diagnostic
:align: center
```

---

## Étape 2 — Baselines (3 figures)

Four mandatory baselines before deep learning : Naive (RW), Historical mean,
ARIMA, Random Forest. ARIMA wins on RMSE, sets the floor any CNN-LSTM has to
beat.

```{image} ../reports/figures/etape2/etape2_metrics_summary.png
:alt: Metrics summary
:align: center
```

```{image} ../reports/figures/etape2/etape2_predictions_val.png
:alt: Predictions VAL
:align: center
```

```{image} ../reports/figures/etape2/etape2_predictions_test.png
:alt: Predictions TEST
:align: center
```

---

## Étape 3 — Feature engineering (4 figures)

24 leakage-free features (momentum, volatility proxies, technical indicators)
— every rolling stat is built with `.shift(1).rolling(...)` so the value at
day `t` only uses information from `≤ t-1`.

```{image} ../reports/figures/etape3/etape3_feature_overview.png
:alt: Feature overview
:align: center
```

```{image} ../reports/figures/etape3/etape3_corr_heatmap.png
:alt: Correlation heatmap
:align: center
```

```{image} ../reports/figures/etape3/etape3_rf_importance.png
:alt: Random Forest importance
:align: center
```

```{image} ../reports/figures/etape3/etape3_volatility_proxies.png
:alt: Volatility proxies
:align: center
```

---

## Étape 4 — HMM regimes (5 figures)

3-state Gaussian HMM (`Bear / Neutral / Bull`) fit on TRAIN only and applied
causally on VAL+TEST via forward filter. Coverage on TEST :
**Bull 409 / Neutral 371 / Bear 168**.

```{image} ../reports/figures/etape4/etape4_regime_timeline.png
:alt: Regime timeline
:align: center
```

```{image} ../reports/figures/etape4/etape4_transition_matrix.png
:alt: Transition matrix
:align: center
```

```{image} ../reports/figures/etape4/etape4_regime_characteristics.png
:alt: Regime characteristics
:align: center
```

```{image} ../reports/figures/etape4/etape4_model_selection.png
:alt: Model selection
:align: center
```

```{image} ../reports/figures/etape4/etape4_spec_comparison.png
:alt: Specification comparison
:align: center
```

---

## Étape 5 — CNN-LSTM `base12` (5 figures)

Compact architecture (~5k params, well under the 10k overfitting threshold for
MASI's small data regime). Directional accuracy **0.556** on TEST, stable
across folds.

```{image} ../reports/figures/etape5/etape5_cumulative.png
:alt: Cumulative TEST P&L
:align: center
```

```{image} ../reports/figures/etape5/etape5_fold_stability.png
:alt: Walk-forward fold stability
:align: center
```

```{image} ../reports/figures/etape5/etape5_baseline_comparison.png
:alt: Baseline comparison
:align: center
```

```{image} ../reports/figures/etape5/etape5_threshold_rule.png
:alt: Threshold rule
:align: center
```

```{image} ../reports/figures/etape5/etape5_lscan.png
:alt: L scan (input window)
:align: center
```

---

## Étape 6 — Backtest & DSR (5 figures)

Cost-aware backtest (5 / 10 / 20 bps), regime-conditional Sharpe heatmap,
Deflated Sharpe Ratio. The CNN-LSTM survives 20 bps with Sharpe ≈ +0.42 — the
only baseline that does.

```{image} ../reports/figures/etape6/etape6_equity_curves.png
:alt: Equity curves
:align: center
```

```{image} ../reports/figures/etape6/etape6_drawdowns.png
:alt: Drawdowns
:align: center
```

```{image} ../reports/figures/etape6/etape6_cost_sensitivity.png
:alt: Cost sensitivity
:align: center
```

```{image} ../reports/figures/etape6/etape6_regime_heatmap.png
:alt: Regime heatmap
:align: center
```

```{image} ../reports/figures/etape6/etape6_final_summary.png
:alt: Final summary
:align: center
```

---

## Étape 7 — Risk layer (4 figures)

Causal VaR / ES (parametric + historical) and a 3-state risk regime built from
TRAIN+VAL quantiles. Kupiec POF passes (5.80 %, p=0.27), Christoffersen
independence rejected — honest negative finding documented in
`outputs/etape7/report.md`.

```{image} ../reports/figures/etape7/etape7_vol_cone.png
:alt: Volatility cone (GARCH)
:align: center
```

```{image} ../reports/figures/etape7/etape7_var_breaches.png
:alt: VaR breaches (Kupiec)
:align: center
```

```{image} ../reports/figures/etape7/etape7_mdd_comparison.png
:alt: MDD comparison
:align: center
```

```{image} ../reports/figures/etape7/etape7_return_distribution.png
:alt: Return distribution
:align: center
```

---

## Étape 8 — Combined strategies (5 figures, HMM-gate wins)

Seven strategies compared (raw signal, HMM-gate, risk-gate, VaR-budget,
combinations). **HMM-gate is the winner** : Sharpe +1.71, MDD −6 %, DSR 0.997
under the historical protocol.

```{image} ../reports/figures/etape8/etape8_dsr_summary.png
:alt: DSR summary
:align: center
```

```{image} ../reports/figures/etape8/etape8_equity_curves.png
:alt: Equity curves (étape 8)
:align: center
```

```{image} ../reports/figures/etape8/etape8_drawdowns.png
:alt: Drawdowns (étape 8)
:align: center
```

```{image} ../reports/figures/etape8/etape8_regime_heatmap.png
:alt: Regime heatmap (étape 8)
:align: center
```

```{image} ../reports/figures/etape8/etape8_sharpe_mdd_scatter.png
:alt: Sharpe vs MDD scatter
:align: center
```

---

## Étape 9 — Robustness (5 figures, 5 axes)

Temporal stability (P1 ≈ P2 ≈ 1.7), cost robustness up to 20 bps, HMM
threshold insensitivity, and Jobson-Korkie-Memmel pairwise tests.

```{image} ../reports/figures/etape9/etape9_subperiod_sharpe.png
:alt: Subperiod Sharpe (P1 vs P2)
:align: center
```

```{image} ../reports/figures/etape9/etape9_cost_sensitivity.png
:alt: Cost sensitivity (étape 9)
:align: center
```

```{image} ../reports/figures/etape9/etape9_dynamic_hmm_threshold.png
:alt: Dynamic HMM threshold
:align: center
```

```{image} ../reports/figures/etape9/etape9_jkm_heatmap.png
:alt: Jobson-Korkie-Memmel heatmap
:align: center
```

```{image} ../reports/figures/etape9/etape9_robustness_scorecard.png
:alt: Robustness scorecard
:align: center
```

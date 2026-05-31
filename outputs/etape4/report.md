# ÉTAPE 4 — HMM Regime Detection Report (v2)
## MASI Hybrid Forecasting System
**Generated:** 2026-05-21
**Input:** `outputs/etape3/features/masi_features_{train,val,test}.csv`
**Script:** `scripts/04_hmm_regimes.py`
**Notebook:** `notebooks/04_hmm_regimes.ipynb`

> **v2 — comparative observation study.** v1 (observations = `[log_return,
> garch_vol]`) produced persistent but **DESCRIPTIVE_ONLY** regimes: they
> separated volatility, not return direction. v2 tests **5 observation sets** and
> selects the best on TRAIN+VAL (TEST held out). The winner — **momentum +
> volatility** — yields directionally **consistent** regimes: verdict **USABLE**.
> Honest caveats are in §6 — the gain is real but partly momentum continuation.

---

## 1. Why a comparative study

A HMM's regimes are only as directional as the series it observes. **Raw daily
return is ≈ noise** → v1 regimes were volatility regimes with no stable next-day
signal (Bull−Bear spread flipped sign train↔test). A **smoothed momentum** signal
(`roll_mean_21`) is persistent and directional, so it should let the HMM form
genuine trend regimes. v2 tests this directly.

### Anti-leakage compliance (unchanged from v1)

| Rule | Status | Evidence |
|------|--------|----------|
| L1 — obs scaler on TRAIN only | ✅ ENFORCED | per-spec TRAIN mean/std |
| L2 — HMM fit on TRAIN only | ✅ ENFORCED | EM on 3,297 train rows; forward-applied |
| L3 — causal regime feature | ✅ ENFORCED | **causal forward-filter**; Viterbi excluded from features |
| L4 / L6 — target & splits | ✅ INHERITED | Étapes 1 & 3 |
| L8 — GARCH per window | 🟡 PARTIAL (by design) | `garch_vol` = Étape 3 TRAIN-frozen; per-window refit → Étape 5 |

**The causal forward-filter** (scaled forward algorithm, `filtered_t =
P(state_t | obs_0..t)`) is what keeps `regime_t` leakage-free. Viterbi /
`predict_proba` decode over the whole sequence → would leak the future → used
only for plots. Causal-vs-Viterbi agreement on the winner: 91.8 / 93.5 / 91.1%.

---

## 2. Comparative study — 5 observation sets

3-state Gaussian HMM, full covariance, 12 EM restarts each.

| Spec | Observations | Persistence | BIC | Bull−Bear next-day spread (train / val / test) | Directional? |
|------|--------------|-------------|-----|-----------------------------------------------|--------------|
| S1 | log_return + garch_vol | 0.897 | 10,106 | −0.00036 / +0.00013 / +0.00217 | ❌ sign flips |
| S2 | log_return | 0.898 | 8,034 | +0.00346 / **−0.00516** / −0.00148 | ❌ overfits TRAIN |
| **S3** | **roll_mean_21 + garch_vol** | **0.938** | 9,846 | **+0.00067 / +0.00043 / +0.00148** | ✅ **consistent** |
| S4 | roll_mean_5 + garch_vol | 0.878 | 8,859 | −0.00026 / −0.00119 / +0.00193 | ❌ sign flips |
| S5 | roll_mean_21 + roll_vol_21 | 0.972 | 9,483 | +0.00075 / +0.00007 / +0.00023 | ✅ consistent (weak) |

### Selection (D2 — TRAIN+VAL only, TEST held out)

- Eligibility: persistence ≥ 0.85 → all 5 pass.
- Winner rule: positive Bull−Bear spread on **both** train **and** val → only
  **S3** and **S5** qualify; S3 wins on the larger VAL spread.
- **TEST was not used to select.** It is reported afterwards as the honest
  out-of-sample check — and S3's TEST spread came out **+0.00148 (positive)**,
  confirming the train/val selection.

### Two key honest observations

1. **S2 (raw daily return) is a cautionary tale** — best TRAIN spread (+0.00346)
   but **−0.00516 on VAL**: it overfits the training noise and fails OOS. This
   confirms raw daily return is not a usable regime signal.
2. **Two momentum specs (S3, S5) both give consistent directional regimes** →
   the result is corroborated, not a lone fluke from one lucky specification.

---

## 3. Winner — S3: momentum (`roll_mean_21`) + volatility (`garch_vol`)

### 3.1 Transition matrix & persistence

| from \ to | Bear | Neutral | Bull |
|-----------|------|---------|------|
| **Bear** | **0.947** | 0.019 | 0.033 |
| **Neutral** | 0.034 | **0.921** | 0.046 |
| **Bull** | 0.030 | 0.025 | **0.945** |

Mean diagonal **0.938**. Expected durations: Bear **19.0 d**, Neutral 12.6 d,
Bull **18.2 d**. Regimes are strongly persistent — genuine multi-week states, not
day-to-day flicker.

### 3.2 Regime characteristics (TRAIN) — now genuinely directional

| Regime | Share | Mean return (ann.) | Volatility (ann.) | Duration |
|--------|-------|--------------------|--------------------|----------|
| **Bear** | 37.5% | **−12.6%** | 9.6% | 19.0 d |
| **Neutral** | 23.0% | +2.7% | **19.2%** | 12.6 d |
| **Bull** | 39.5% | **+7.9%** | 8.4% | 18.2 d |

Unlike v1, the regimes now have a **real return ordering** (Bear −12.6% <
Neutral +2.7% < Bull +7.9% annualized). Note **Neutral has the highest
volatility** — it is the *choppy transition* regime between sustained Bull and
Bear trends, which are themselves calmer. This is economically sensible.

---

## 4. Out-of-sample coverage

| Split | Bear | Neutral | Bull |
|-------|------|---------|------|
| TRAIN | 37.5% | 23.0% | 39.5% |
| VAL | 24.7% | 15.7% | 59.6% |
| TEST | 17.7% | 39.1% | 43.1% |

All regimes present in every split (rarest 15.7%). VAL is 60% Bull — the
post-COVID 2020–2022 recovery was a sustained uptrend, correctly identified.

---

## 5. Honest evaluation — verdict USABLE

### 5.1 Regime-conditional next-day return

| Split | Bear | Neutral | Bull | **Bull − Bear** |
|-------|------|---------|------|------------------|
| TRAIN | −0.000412 | +0.000048 | +0.000258 | **+0.000670** |
| VAL | +0.000214 | −0.000554 | +0.000647 | **+0.000432** |
| TEST | −0.000972 | +0.001120 | +0.000508 | **+0.001481** |

**The Bull−Bear spread is positive on all three splits** — the regime now carries
a *consistent* directional signal (v1 flipped sign). Honest nuance: the clean
monotone ordering Bear < Neutral < Bull holds only on **TRAIN**; on VAL/TEST the
**Neutral** regime is out of place (it is the noisy transitional state). The
robust, reproducible signal is **Bear vs Bull (the extremes)**.

### 5.2 Regime-timed strategy (long Bull / short Bear, 5 bps cost)

| Split | Sharpe (ann.) | Max Drawdown | Trades |
|-------|---------------|--------------|--------|
| TRAIN | **+0.69** | −0.15 | 247 |
| VAL | **+0.93** | −0.06 | 40 |
| TEST | **+0.93** | −0.14 | 95 |

**Positive and consistent on all three windows** (v1 lost on 2 of 3). A rule that
works across train, val *and* test is a genuine — if modest — result.

### 5.3 Verdict

| Check | Threshold | Result | Verdict |
|-------|-----------|--------|---------|
| Persistence | mean diag ≥ 0.85 | 0.938 | ✅ PASS |
| Predictive content | Bull−Bear > 0 on train **and** test | +0.00067 / +0.00148 | ✅ PASS |
| OOS coverage | rarest regime ≥ 5% | 15.7% | ✅ PASS |

**OVERALL = USABLE.** The momentum-based HMM produces persistent, leakage-free,
directionally-consistent regimes — a real improvement over the v1
DESCRIPTIVE_ONLY result.

---

## 6. Honest caveats — what "USABLE" does NOT mean

The result is genuine but must not be oversold:

1. **The directionality is momentum continuation.** The regime ≈ "is the 21-day
   trend up or down". MASI shows momentum continuation (return autocorrelation
   confirmed in Étape 0 — an illiquid market diffuses information slowly). The
   HMM works because it captures *trend persistence*, not because it forecasts
   genuine turning points.

2. **Possible redundancy with existing features.** `roll_mean_5` and
   `roll_mean_21` are **already** in the 24-feature CNN-LSTM set (Étape 3). The
   S3 regime largely re-encodes `roll_mean_21` as a smoothed 3-state label. Its
   *marginal* value beyond the raw momentum features is **not yet established** —
   the Étape 5 ablation (with vs without the regime feature) remains mandatory.

3. **Multiple-comparisons risk.** 5 specs were tested and the winner picked.
   Mitigations: selection used TRAIN+VAL only (TEST held out and confirmed);
   two independent momentum specs (S3 and S5) both qualified; S3 was an a-priori
   natural candidate. The result is corroborated but the spread magnitudes
   (0.04–0.15%/day) are small — a frontier-market-sized edge, not a large one.

4. **BIC still prefers 4 states** (BIC: 2→12,044, 3→9,846, 4→8,273). 3 states
   kept for interpretability (C3, project design) — a deliberate choice.

**Bottom line:** the HMM improvement succeeded — verdict moved DESCRIPTIVE_ONLY →
USABLE. But whether the regime feature *adds skill to the CNN-LSTM beyond the
momentum features already present* is an open question that Étape 5 must answer
empirically. Per RULE 6/8, if the ablation shows no gain, the regime is dropped.

---

## 7. Decisions for Étape 5 (CNN-LSTM)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Use the winner regime: one-hot (3) + soft probs (3) from `masi_regimes_*.csv` | S3, the directionally-consistent spec |
| 2 | **Mandatory ablation:** CNN-LSTM with vs without the regime feature | Caveat §6.2 — must prove marginal value |
| 3 | If ablation shows no gain → drop the regime feature | RULE 6/8 — no unjustified complexity |
| 4 | Re-fit GARCH **and** HMM inside each walk-forward window | Completes L8/L2 (current = single TRAIN fit) |
| 5 | Prefer soft probabilities over hard one-hot | smoother near transitions |
| 6 | Expect a **modest** gain at best | honest expectation set by §6 |

---

## 8. Output Artifacts

| Artifact | Path |
|----------|------|
| Regime-augmented features — TRAIN/VAL/TEST | `outputs/etape4/regimes/masi_regimes_{train,val,test}.csv` |
| Winner HMM parameters | `outputs/etape4/regimes/hmm_params.json` |
| Observation scaler (TRAIN-only, L1) | `outputs/etape4/regimes/hmm_obs_scaler_train_only.json` |
| Winner evaluation metrics | `outputs/etape4/regimes/regime_evaluation.json` |
| **5-spec comparison table** | `outputs/etape4/regimes/spec_comparison.json` |
| Plot — regime timeline | `reports/figures/etape4/etape4_regime_timeline.png` |
| Plot — transition matrix | `reports/figures/etape4/etape4_transition_matrix.png` |
| Plot — regime characteristics | `reports/figures/etape4/etape4_regime_characteristics.png` |
| Plot — 5-spec comparison | `reports/figures/etape4/etape4_spec_comparison.png` |
| Plot — n_states selection & coverage | `reports/figures/etape4/etape4_model_selection.png` |

---

*End of Étape 4 Report (v2) — generated by `scripts/04_hmm_regimes.py`. Winner:
S3 (momentum + volatility). Verdict: USABLE — directionally-consistent regimes,
with the honest caveat that the gain is momentum continuation and the Étape 5
ablation must confirm marginal value.*

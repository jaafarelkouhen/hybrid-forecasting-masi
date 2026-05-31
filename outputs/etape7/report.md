# ÉTAPE 7 — Couche Risque (VaR / ES / GARCH / régime de risque)
## MASI Hybrid Forecasting System
**Generated:** 2026-05-24
**Inputs:** Prédictions CNN-LSTM base12 (étape 5) · Régimes HMM causaux (étape 4) · Features train/val/test (étape 3) · Sanity check equity curves (étape 6)
**Scripts:** `scripts/07_risk_layer.py` · `notebooks/07_risk_layer.ipynb`

---

## 0. Executive Summary (mémoire-ready)

On ajoute une **couche de protection ex-ante** au-dessus du signal CNN-LSTM `base12` (modèle de production validé étapes 5/6), sans modifier le prédicteur. Trois variantes de filtre sont évaluées :

| Stratégie | Sharpe ann. | MDD | Equity finale | Trades |
|---|---|---|---|---|
| **CNN-LSTM nu (référence étape 6)** | +1.055 | −15.96 % | 1.737 | 221 |
| + Filtre VaR paramétrique 5 % | +1.055 | −15.96 % | 1.737 | 221 |
| + Filtre ES paramétrique 5 % | +1.055 | −15.96 % | 1.737 | 221 |
| **+ Filtre régime de risque (vol GARCH)** | **+1.201** 🏆 | **−6.68 %** 🏆 | 1.422 | 209 |

**Trois findings honnêtes :**

1. Les **filtres VaR/ES sur la prédiction** sont **dégénérés** sur ce modèle : les prédictions CNN-LSTM sont trop conservatrices (|ŷ| de l'ordre de 1e-4) pour jamais déclencher le seuil VaR (≈ −1.2 %). Conclusion méthodologique honnête : pour ce prédicteur, filtrer le signal avec un seuil VaR n'a aucun effet.

2. Le **filtre régime de risque** (stand-aside les jours où σ_GARCH est dans le tertile haut) est la seule couche utile : **MDD divisé par 2** (−15.96 % → −6.68 %), **Sharpe amélioré** (+1.055 → +1.201), au prix d'une equity finale plus basse (1.42 vs 1.74). Trade-off classique risk/return — favorable au régulateur, défavorable au rendement absolu.

3. Les **VaR 5 %** (hist & param) **passent Kupiec POF** (taux observé conforme à 5 %) mais **échouent Christoffersen** (LR_ind p < 0.01) : les breaches forment des **clusters temporels** — limite connue d'un VaR statique en période de stress, à corriger en étape 8 par un seuil dynamique conditionné au régime.

**Position scientifique finale :** la couche risque est une **addition utile mais limitée**. Le seul filtre qui marche est le régime de risque ; les filtres VaR/ES sont rejetés méthodologiquement. La couche sera utilisée en étape 8 dans la **combinaison** CNN-LSTM + HMM directionnel + régime de risque.

---

## 1. Méthodologie

### 1.1 Architecture de la couche

```
        ┌─────────────────────────┐
        │  CNN-LSTM base12        │  (étape 5 — figé)
        │  ŷ_t = f(features_t)    │
        └────────┬────────────────┘
                 │
        ┌────────▼────────────────┐
        │  signal_brut = sign(ŷ)  │  (étape 6)
        └────────┬────────────────┘
                 │
        ┌────────▼─────────────────────────────┐
        │  COUCHE RISQUE (étape 7) — 3 filtres │
        │  ① ŷ ≥ VaR_paramétrique ?           │
        │  ② ŷ ≥ ES_paramétrique  ?           │
        │  ③ régime ≠ high (σ_GARCH q67+) ?   │
        └────────┬─────────────────────────────┘
                 │
        ┌────────▼────────────────┐
        │  signal_filtré → trade  │
        └─────────────────────────┘
```

Le modèle de prédiction n'est PAS retouché. La couche risque est **strictly additive**.

### 1.2 Métriques de risque calculées

| Métrique | Définition | Fenêtre | Source |
|---|---|---|---|
| `var_hist_5` | Quantile empirique 5 % | rolling 252j causal | Jorion (2006) |
| `var_param_5` | μ + σ_GARCH · Φ⁻¹(0.05) | μ rolling 252j ; σ_GARCH causal | normale paramétrique |
| `es_hist_5` | E[r \| r ≤ VaR_hist] | rolling 252j causal | Acerbi-Tasche (2002) |
| `es_param_5` | μ − σ_GARCH · φ(z_α)/α | μ rolling 252j ; σ_GARCH causal | normale paramétrique |
| `vol_garch` | σ conditionnelle GARCH(1,1) | fit TRAIN-only étape 3 | propagée causalement |
| `vol_realized_21` | écart-type rolling 21j | shift(1).rolling(21) | étape 3 |
| `risk_regime` | discrétisation σ_GARCH | seuils q33/q67 figés TRAIN+VAL | bornes ci-dessous |

### 1.3 Seuils figés du régime de risque (anti-fuite L1)

| Seuil | Valeur σ_GARCH | Source |
|---|---|---|
| q33 | 0.00524 | quantile TRAIN+VAL (3775 jours) |
| q67 | 0.00653 | quantile TRAIN+VAL (3775 jours) |

→ Si σ_GARCH_t ≤ q33 : `low`
→ Si σ_GARCH_t ≤ q67 : `normal`
→ Sinon : `high`

**Aucun recalcul sur TEST** — les seuils sont gelés à la fin de VAL. C'est l'application stricte de la règle anti-fuite L1.

### 1.4 Trois variantes de signal protégé

```
signal_var_filter     = sign(ŷ) · 𝟙{ŷ ≥ VaR_param_5}
signal_es_filter      = sign(ŷ) · 𝟙{ŷ ≥ ES_param_5}
signal_risk_regime    = sign(ŷ) · 𝟙{risk_regime ≠ "high"}
```

Coût de transaction : **5 bps one-way** (cohérent avec étape 6 primary cost).

### 1.5 Validation backtest des VaR

| Test | H0 | Statistique | Loi asympt. |
|---|---|---|---|
| Kupiec POF (1995) | taux de breach = α | LR_uc | χ²(1) |
| Christoffersen (1998) | breaches indépendants | LR_ind | χ²(1) |

---

## 2. Anti-fuite (règles L1–L8)

| Règle | Application étape 7 |
|---|---|
| **L1** | Quantiles q33/q67 figés sur TRAIN+VAL — vérifié programmatiquement (`assert quantiles == TRAIN_VAL_Q`) |
| **L2** | HMM régimes consommés depuis étape 4 v2 (causaux) |
| **L3** | `series.shift(1).rolling(252, min_periods=252)` → fenêtre strictement [t−252, t−1] |
| **L4** | `y_true` jamais utilisé pour décider le signal de `t` (sanity check : `assert max\|y_true_preds − target_y_next_regs\| < 1e-6`) |
| **L5** | Signal `t` exécuté contre `y_true_t = ln(P_{t+1}/P_t)` ; cohérent étape 6 |
| **L6** | Gap walk-forward inhérent étape 1 ; étape 7 ne réintroduit pas de données |
| **L7** | Jours zero-volume retirés étape 1 |
| **L8** | GARCH fit TRAIN-only (étape 3 `garch_params_train.json`) ; `garch_vol` causal |

**Assertions programmatiques actives** :
- `len(preds) == len(regs) == 948`
- `max\|y_true − target_y_next\| < 1e-6` (max observé : 2.06e-9)
- `len(slice_test) == 948`
- `out[VaR/ES cols].notna().all()` après warm-up
- `\|ln(eq_finale) − sum(strategy_return)\| < 1e-4` (max observé : 1.24e-9)

---

## 3. Résultats — Validation backtest des VaR

### 3.1 Kupiec POF — taux de breach

| VaR type | Breaches / 948 | Taux observé | LR_uc | p-value | Verdict |
|---|---|---|---|---|---|
| Historique | 58 | 6.12 % | 2.34 | **0.126** | **OK** (≠ 5 % non significatif) |
| Paramétrique | 55 | 5.80 % | 1.22 | **0.269** | **OK** (≠ 5 % non significatif) |

Les deux VaR sont **bien calibrées** sur le taux global. Léger sur-breach (6 % vs 5 % attendu) cohérent avec une distribution leptokurtique typique des marchés frontières (étape 0 a confirmé `kurtosis ≈ 10` sur les retours MASI).

### 3.2 Christoffersen — indépendance des breaches

| VaR type | n00 | n01 | n10 | n11 | LR_ind | p-value | Verdict |
|---|---|---|---|---|---|---|---|
| Historique | 840 | 49 | 49 | 9 | 7.00 | **0.008** | **REJETÉ** |
| Paramétrique | 846 | 46 | 46 | 9 | 8.43 | **0.004** | **REJETÉ** |

**Rejet net** : les breaches forment des clusters temporels. `n11 = 9` (un breach suivi d'un breach) est anormalement élevé vs le `n11` attendu sous indépendance.

**Interprétation économique** : pendant les épisodes de stress (chute concentrée sur quelques jours), un VaR statique fenêtré 252j est **trop lent** à s'ajuster. Limite connue de tout VaR non-conditionnel ; à corriger en étape 8 via un seuil dynamique régime-conditionné (ou un VaR EWMA / Filtered Historical Simulation).

---

## 4. Résultats — Comparaison des stratégies

### 4.1 Métriques agrégées TEST (948 jours, 5 bps one-way)

| Stratégie | Sharpe ann. | MDD | Calmar | Equity finale | Trades |
|---|---|---|---|---|---|
| CNN-LSTM nu | +1.055 | −15.96 % | +0.92 | 1.737 | 221 |
| + Filtre VaR paramétrique 5 % | +1.055 | −15.96 % | +0.92 | 1.737 | 221 |
| + Filtre ES paramétrique 5 % | +1.055 | −15.96 % | +0.92 | 1.737 | 221 |
| **+ Filtre régime risque** | **+1.201** | **−6.68 %** | **+1.80** | 1.422 | 209 |

### 4.2 Pourquoi les filtres VaR/ES sont dégénérés (finding clé)

Sur les 948 jours TEST :
- `|ŷ_pred|` médian ≈ **1.4 × 10⁻⁴** (prédictions CNN-LSTM très conservatrices)
- `VaR_param_5` moyen ≈ **−1.19 × 10⁻²**
- → Condition `ŷ_pred ≥ VaR_param_5` **toujours vraie** → filtre **jamais déclenché**

Idem pour ES. C'est un **résultat honnête mais important** : la conception classique du filtre VaR (« ne tradez pas si la prédiction est sous le VaR ») suppose un prédicteur capable de produire des forecasts extrêmes (par ex. ARIMA dont la variance des prédictions est nettement plus large). Sur un CNN-LSTM régularisé à ~5 000 paramètres, les prédictions restent près de zéro et ce design de filtre est inopérant.

**Implication pour le mémoire** : on documente ce design défaillant comme un *negative result* méthodologique. La conception alternative (à creuser en étape 8) est :
- Filtrer sur `realized_vol` ou `garch_vol` (état du marché), pas sur `ŷ_pred` (état du modèle)
- Ou bien introduire un seuil de **conviction** : `|ŷ_pred| > seuil` pour activer la position (déjà testé infructueusement en étape 5 comme « confidence-threshold »).

### 4.3 Pourquoi le régime de risque marche (finding clé)

| Indicateur | Valeur | Lecture |
|---|---|---|
| Réduction MDD | −15.96 % → **−6.68 %** (−58 %) | la couche évite les drawdowns concentrés dans les jours σ_GARCH ≥ q67 |
| Amélioration Sharpe | +1.055 → **+1.201** (+14 %) | la vol évitée pèse plus dans la moyenne que le rendement perdu |
| Perte equity finale | 1.737 → 1.422 (−18 %) | coût de l'inaction : ~⅔ des jours TEST sont écartés (441/948 = 46 % en `high`) |
| Calmar | +0.92 → +1.80 (+96 %) | ratio rendement/MDD presque doublé |

→ **C'est la couche défensive du mémoire.** Elle confirme l'intuition étape 6 (« les pertes se concentrent dans la vol haute / régime Neutral ») et y répond opérationnellement.

### 4.4 Répartition du régime de risque sur TEST

| Régime | n jours | % | σ_GARCH median |
|---|---|---|---|
| low (≤ q33) | 262 | 27.6 % | ~0.0044 |
| normal (q33 → q67) | 245 | 25.8 % | ~0.0058 |
| **high (> q67)** | **441** | **46.5 %** | ~0.0085 |

La période TEST (2022-06 → 2026-04) est **plus volatile** que TRAIN+VAL (les seuils q33/q67 ont été fixés sur 2007-2022) — cohérent avec le contexte macro post-COVID + tensions géopolitiques. C'est précisément pourquoi le filtre régime apporte de la valeur : il identifie correctement le « risque actuel > risque historique » sans look-ahead.

---

## 5. Synthèse VaR/ES (moyennes TEST)

| Métrique | Valeur |
|---|---|
| VaR_hist_5 moyen | −1.07 % |
| VaR_param_5 moyen | −1.19 % |
| ES_hist_5 moyen | −1.80 % |
| ES_param_5 moyen | −1.50 % |
| Pire journée TEST | −5.81 % |
| Meilleure journée TEST | +4.95 % |

L'**ES historique > ES paramétrique** en magnitude (−1.80 % vs −1.50 %) → la queue empirique est **plus épaisse** que la normale. Confirmé par la kurtosis ≈ 10 observée à l'étape 0. Pour un mémoire, c'est un argument en faveur de l'**ES historique** plutôt que paramétrique, mais ce point n'affecte pas la conclusion (les deux filtres VaR/ES sont dégénérés sur le CNN-LSTM).

---

## 6. Honest Limitations

1. **Filtres VaR/ES sur prédiction = degenerate design** pour CNN-LSTM conservateur. Conclusion méthodologique honnête à corriger en étape 8.
2. **Christoffersen rejeté** : VaR statique ne capture pas les clusters de stress. Étape 8 pourra introduire un VaR conditionnel au régime HMM ou EWMA.
3. **Régime de risque sur 3 quantiles seulement** : un découpage plus fin (5 ou 10 quantiles) donnerait des seuils plus granulaires, mais réduit le n par bucket.
4. **Pas de VaR multi-horizon** (1-jour seulement). Pour un dashboard de production, un VaR 5-jour / 10-jour serait utile (scaling racine du temps approximatif).
5. **Coût uniforme 5 bps** — pas de modèle d'impact de marché. La couche risque réduit le turnover (221 → 209 trades), mais le gain en coût est marginal sur ce petit écart.

---

## 7. Future work (matérialisé étape 8+)

| # | Idée | Étape cible |
|---|---|---|
| 1 | Combiner régime de risque (vol) + régime HMM (direction) pour ne trader qu'en Bear/Bull AVEC vol normale/basse | 8 |
| 2 | VaR conditionnel au régime (Filtered Historical Simulation, EWMA, GARCH-Filtered) — corrige Christoffersen | 9 (robustesse) |
| 3 | Position sizing proportionnel inverse à σ_GARCH (Kelly fractionnaire) plutôt que filtre binaire | 8 (variante) |
| 4 | Combiner avec un seuil de conviction sur \|ŷ\| (déjà testé étape 5, transfert TEST échoué) | reporté |

---

## 8. Output Artifacts

| Artifact | Path |
|---|---|
| **Fichier canonique étape 6 (base API/dashboard)** | `outputs/etape6/etape6_final_predictions.csv` |
| Métriques risque + signaux filtrés + strategy_returns | `outputs/etape7/risk_metrics_test.csv` |
| Tests Kupiec/Christoffersen + comparaison stratégies | `outputs/etape7/risk_validation.json` |
| Plot — cône de vol GARCH vs réalisée | `reports/figures/etape7/etape7_vol_cone.png` |
| Plot — breaches du VaR paramétrique | `reports/figures/etape7/etape7_var_breaches.png` |
| Plot — distribution des retours avec VaR/ES | `reports/figures/etape7/etape7_return_distribution.png` |
| Plot — drawdowns comparatifs des 4 stratégies | `reports/figures/etape7/etape7_mdd_comparison.png` |
| Script principal | `scripts/07_risk_layer.py` |
| Notebook reproduisible | `notebooks/07_risk_layer.ipynb` |

---

## 9. Schéma du fichier canonique `etape6_final_predictions.csv`

948 lignes × 7 colonnes :

| Colonne | Type | Description |
|---|---|---|
| `date` | date (YYYY-MM-DD) | jour TEST |
| `actual_return` | float | `y_true = ln(P_{t+1}/P_t)`, log-retour réalisé |
| `predicted_return` | float | `ŷ` CNN-LSTM base12 |
| `signal` | int ∈ {−1, 0, +1} | `sign(ŷ)` |
| `regime` | int ∈ {0, 1, 2} | régime HMM Bear/Neutral/Bull (causal) |
| `regime_name` | str | "Bear" / "Neutral" / "Bull" |
| `strategy_return` | float | `signal · y_true − cost si position change` |

**Sanity check passé :** `Σ strategy_return = +0.5521 = ln(1.737) = ln(equity finale CNN-LSTM étape 6)`. Écart numérique : 1.24e-9.

C'est ce fichier qui servira de **single source of truth** pour l'étape 8 (combinaisons) et plus tard pour l'API / dashboard.

---

*End of Étape 7 Report — generated by `scripts/07_risk_layer.py`.*
*Recommandation mémoire : la couche risque utile est le **filtre régime de vol** ; les filtres VaR/ES sur prédiction sont rejetés méthodologiquement (résultat honnête à documenter). Le régime de risque sera combiné au régime HMM directionnel en étape 8.*

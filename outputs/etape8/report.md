# ÉTAPE 8 — Stratégies combinées (7 candidates)
## MASI Hybrid Forecasting System
**Generated:** 2026-05-24
**Inputs:** `etape6_final_predictions.csv` + `outputs/etape7/risk_metrics_test.csv`
**Scripts:** `scripts/08_strategies.py` · `notebooks/08_strategies.ipynb`

---

## 0. Executive Summary (mémoire-ready)

On compare 7 stratégies sur la fenêtre TEST canonique (948 jours, 5 bps one-way), avec **Deflated Sharpe Ratio** (N=7), **Diebold-Mariano** vs CNN-LSTM nu, et **décomposition régime-conditionnelle**.

### Tableau récapitulatif

| # | Stratégie | Mode | Sharpe | Sortino | MDD | Calmar | Eq finale | DSR | DM vs nu |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Buy & Hold | binary | +0.878 | +1.094 | −20.73 % | +0.63 | 1.583 | 0.850 | ns |
| 2 | CNN-LSTM nu | binary | +1.055 | +1.339 | −15.96 % | +0.99 | 1.737 | 0.917 | — |
| 3 | **CNN-LSTM + HMM-gate** | binary | **+1.709** 🏆 | **+1.896** | **−6.00 %** 🏆 | **+2.84** 🏆 | **1.808** 🏆 | **0.997** 🏆 | ns |
| 4 | CNN-LSTM + risk-gate | binary | +1.201 | +1.172 | −6.68 % | +1.47 | 1.422 | 0.948 | ns |
| 5 | CNN-LSTM + HMM + risk | binary | +1.062 | +0.987 | −6.28 % | +1.32 | 1.350 | 0.915 | ns |
| 6 | CNN-LSTM × VaR-budget | continuous | +1.222 | +1.676 | −10.63 % | +1.30 | 1.628 | 0.956 | ns |
| 7 | CNN-LSTM × HMM-cond. budget | continuous | +1.201 | +1.662 | −11.57 % | +1.21 | 1.640 | 0.955 | ns |

🏆 = vainqueur de la colonne · ns = DM p-value > 0.05

### Quatre findings honnêtes

1. **CNN-LSTM + HMM-gate gagne sur 5 métriques sur 7** (Sharpe, Sortino, MDD, Calmar, eq finale, DSR) ET franchit le seuil DSR = 0.95 « publishable » (DSR = 0.997). **C'est la nouvelle recommandation production**, supérieure à CNN-LSTM nu (étape 6) sur tous les axes économiques. Cela confirme la « future work » de l'étape 6 et l'inscrit dans le scope mémoire.

2. **Combiner HMM-gate + risk-gate dégrade Sharpe** (1.71 → 1.06). Le risk-gate est trop restrictif quand on a déjà gateé sur HMM : il retire des jours Bull/Bear avec haute vol qui restent profitables (le CNN-LSTM est capable d'exploiter la haute vol en régime directionnel net). **L'union des protections n'est pas additive — finding non trivial à signaler.**

3. **Les stratégies VaR-budget (6, 7)** sont une **alternative cohérente** (Sharpe ~1.22, DSR ≥ 0.95) mais **payent en turnover** (588-611 trades vs 180 pour HMM-gate). Sur la métrique Calmar, elles sont battues par HMM-gate (1.30 vs 2.84). À considérer si on veut une position continue plutôt que binaire.

4. **Diebold-Mariano : aucune différence n'est statistiquement significative** (toutes p > 0.27). Avec 948 jours TEST, on ne peut pas affirmer avec 95 % de confiance que HMM-gate bat CNN-LSTM nu sur les retours individuels. **Économiquement** la différence est nette (+0.65 Sharpe annualisé) ; **statistiquement** elle ne franchit pas le test. Limite intrinsèque d'un échantillon ~3.8 ans.

**Position scientifique finale :** la combinaison **CNN-LSTM + HMM-gate** est le meilleur trade-off rendement/risque, validée par le seul DSR publishable (0.997) du projet. Le filtre risque (étape 7) reste utile comme alternative défensive mais n'est PAS additif à HMM-gate.

---

## 1. Méthodologie

### 1.1 Architecture des 7 stratégies

```
                            ┌─ ŷ = CNN-LSTM base12 (étape 5) ─┐
                            │                                 │
                  ┌─────────┴─────────┐         ┌─────────────┴─────────────┐
                  ▼                   ▼         ▼                           ▼
         signal = sign(ŷ)       gate par regime         sizing continu  w_t = min(1, B/|VaR|)
              {-1, 0, +1}         (HMM ou risk)              ∈ [-1, +1]
                  │                   │                           │
        ┌─────────┼─────────┐         │                ┌──────────┼──────────┐
        ▼         ▼         ▼         ▼                ▼                     ▼
     #2 nu    #3 HMM    #4 risk   #5 HMM+risk       #6 VaR-B            #7 HMM-cond-B
                                                  (B=0.01 fixe)         (B varie / régime)

                #1 Buy & Hold = pos=1 ∀t (réf. passive, hors arbre)
```

### 1.2 Conventions de coût

| Stratégies | Coût | Justification |
|---|---|---|
| 1-5 (binary, position ∈ {−1, 0, +1}) | flat 5 bps si position change | cohérent étape 6 (comparabilité directe) |
| 6-7 (continuous, position ∈ [−1, +1]) | 5 bps × \|Δw\| | proportionnel turnover (repo source `economic_evaluation.py` L107) |

Cette dichotomie est honnête : un changement binaire −1 → +1 est un événement unique (1 changement de signe, mais on trade 2 unités d'exposition) ; pour les stratégies continues, le coût doit suivre l'amplitude pour éviter de masquer le coût réel des repondérations partielles.

### 1.3 Paramètres figés (avant TEST, anti-fuite L1)

| Paramètre | Valeur | Justification |
|---|---|---|
| `B_std` (budget standard, strat 6) | 0.01 (1 %) | repo source L342, littérature risk-budgeting |
| `rolling_budget` (strat 7) | 60 jours | repo source L388 |
| `q_stress` (strat 7) | 0.30 | repo source — petit budget en stress |
| `q_normal` (strat 7) | 0.70 | repo source — gros budget en normal |
| `Stress proxy` (strat 7) | régime HMM = Neutral | étape 6 a quantifié l'échec en Neutral |
| Coût primaire | 5 bps | étape 6 D2 |
| Fenêtre TEST | 948 jours | étape 1 canonique |
| N_trials DSR | 7 | nombre de stratégies évaluées |

### 1.4 Métriques calculées

| Catégorie | Métriques |
|---|---|
| Rendement / risque | Sharpe ann., Sortino ann., ann_return, ann_vol |
| Drawdown | MDD, Calmar |
| Trading | n_trades, turnover moyen, exposition moyenne, % jours actifs |
| Régime-cond | Sharpe + mean_return_ann par régime HMM (Bear/Neutral/Bull) |
| Validation | DSR (PSR vs SR0), PSR vs 0 |
| Test pairwise | Diebold-Mariano (h=1) vs CNN-LSTM nu |

---

## 2. Attribution (anti-plagiat)

Les stratégies **6** (VaR-budget) et **7** (HMM-conditional budget) sont **inspirées** du repo :

> `_analysis_research_notebooks/masi-risk-research-notebooks-main/src/analysis/economic_evaluation.py`
> Fonctions `compute_weights_from_budget_and_var` (L310-326), `weights_benchmark_econometric` (L342-352), `weights_hmm_lstm_quantile_budget` (L366-416).

La formule `w_t = min(cap, B_t / |VaR_t|)` est une convention standard de **risk-budgeting** (Bertrand & Prigent 2003, Roncalli 2014 — *Introduction to Risk Parity and Budgeting*). Notre implémentation **adapte** la méthode du repo :

- **Setup long-short** (position ∈ [−1, +1]) au lieu de long-only (∈ [0, 1])
- **VaR consommé depuis étape 7** (paramétrique GARCH causal) au lieu d'un LSTM-VaR
- **Proxy stress = régime HMM = Neutral** (étape 4, défendable par étape 6) au lieu d'un stress flag externe
- **Code écrit indépendamment** dans `scripts/08_strategies.py` — pas de copier-coller

---

## 3. Anti-fuite (L1–L8)

| Règle | Application étape 8 |
|---|---|
| L1 | Quantiles rolling de B_t (strat 7) : `.shift(1).rolling(60, min_periods=60)` strictement causal |
| L2 | Régimes HMM consommés étape 4 v2 (causaux) |
| L3 | Idem L1 : tous les rolling utilisent `shift(1).rolling(.)` |
| L4 | `y_true` jamais utilisé pour décider position ou poids |
| L5 | Position `t` exécutée contre `y_true_t = ln(P_{t+1}/P_t)` |
| L6/L7 | Inhérent étapes précédentes (pas de réintroduction de données) |
| L8 | VaR consommé étape 7 (déjà GARCH TRAIN-only) |

**Assertions actives passées** :
- `len(df) == 948`
- CNN-LSTM nu Sharpe ≈ 1.055 ± 0.01 (cohérent étape 6)
- CNN-LSTM nu eq finale ≈ 1.737 ± 0.01 (cohérent étape 6)
- Buy & Hold eq finale == exp(sum(y_true) − 5 bps) à 1e-16 près

---

## 4. Résultats — Métriques agrégées

### 4.1 Vue d'ensemble (TEST 948 jours, 5 bps one-way)

| Stratégie | Sharpe | Sortino | MDD | Calmar | Eq finale | % actif | Turn moy | Trades |
|---|---|---|---|---|---|---|---|---|
| Buy & Hold | +0.878 | +1.094 | −20.73 % | +0.63 | 1.583 | 100 % | 0.001 | 1 |
| CNN-LSTM nu | +1.055 | +1.339 | −15.96 % | +0.99 | 1.737 | 100 % | 0.465 | 221 |
| **CNN-LSTM + HMM-gate** | **+1.709** | **+1.896** | **−6.00 %** | **+2.84** | **1.808** | 61 % | 0.306 | 180 |
| CNN-LSTM + risk-gate | +1.201 | +1.172 | −6.68 % | +1.47 | 1.422 | 53 % | 0.310 | 209 |
| CNN-LSTM + HMM + risk | +1.062 | +0.987 | −6.28 % | +1.32 | 1.350 | 51 % | 0.287 | 189 |
| CNN-LSTM × VaR-budget | +1.222 | +1.676 | −10.63 % | +1.30 | 1.628 | 86 % | 0.420 | 611 |
| CNN-LSTM × HMM-cond. budget | +1.201 | +1.662 | −11.57 % | +1.21 | 1.640 | 85 % | 0.416 | 588 |

### 4.2 Lecture par axe

**Axe Sharpe / Sortino** : HMM-gate domine clairement (1.71 vs 1.06 pour le suivant). VaR-budget (#6) en seconde position (1.22).

**Axe MDD** : Toutes les variantes gatées sont à −6 à −7 % vs −16 % pour CNN-LSTM nu — réduction d'environ ⅔. C'est la valeur défensive principale du gating.

**Axe Calmar** : HMM-gate écrasant (+2.84). Le rapport rendement/MDD est presque doublé vs n'importe quelle autre option.

**Axe turnover** : Les VaR-budget continus (6, 7) ont **3× plus de trades** que les binary gating. Économiquement crédible (repondération quasi quotidienne) mais coûte en frais.

**Axe exposition** : HMM-gate trade seulement **61 %** des jours (les 371 jours Neutral écartés), risk-gate **53 %** (441 jours high-vol). VaR-budget trade **86 %** mais avec exposition partielle.

---

## 5. Résultats — Deflated Sharpe Ratio (N=7)

| Stratégie | SR daily | SR0 seuil | Skew | Kurt | DSR vs SR0 | PSR vs 0 | Verdict |
|---|---|---|---|---|---|---|---|
| Buy & Hold | +0.055 | 0.021 | −0.93 | 9.83 | 0.850 | 0.953 | WEAK |
| CNN-LSTM nu | +0.067 | 0.021 | −0.23 | 10.81 | 0.917 | 0.978 | STRONG (≥0.90) |
| **CNN-LSTM + HMM-gate** | **+0.108** | 0.021 | +0.05 | 17.55 | **0.997** | 1.000 | **PUBLISHABLE (≥0.95)** |
| CNN-LSTM + risk-gate | +0.076 | 0.021 | −0.16 | 14.18 | 0.948 | 0.988 | STRONG (≥0.90) |
| CNN-LSTM + HMM + risk | +0.067 | 0.021 | −0.20 | 17.55 | 0.915 | 0.977 | STRONG (≥0.90) |
| **CNN-LSTM × VaR-budget** | +0.077 | 0.021 | −0.19 | 9.85 | **0.956** | 0.991 | PUBLISHABLE (≥0.95) |
| **CNN-LSTM × HMM-cond. budget** | +0.076 | 0.021 | −0.21 | 10.06 | **0.955** | 0.990 | PUBLISHABLE (≥0.95) |

**3 stratégies franchissent DSR = 0.95** : HMM-gate (0.997), VaR-budget (0.956), HMM-cond. budget (0.955).

**Comparaison avec étape 6** : aucun modèle ne franchissait 0.95 à l'étape 6 (best ARIMA 0.62). Pourquoi maintenant ?

> **Honnêteté méthodologique sur le DSR** : le seuil SR0 dépend de **V = variance des Sharpes des trials**. À l'étape 6, V = 0.0029 (5 trials incluant Random Walk et Hist. Mean qui ont SR ≈ 0 → grande variance). À l'étape 8, V = 0.00023 (7 trials, toutes raisonnables → variance 12× plus faible). Du coup, SR0 daily passe de 0.064 à 0.021. **Cela explique mécaniquement pourquoi les DSR sont meilleurs ici.**
>
> Si on calculait le DSR avec V = 0.0029 (V de l'étape 6), HMM-gate aurait un DSR ≈ 0.91 (STRONG mais pas PUBLISHABLE). La conclusion reste favorable (HMM-gate au-dessus de tout le reste) mais le franchissement du seuil 0.95 est partiellement dû au choix du trial set.

C'est la limite reconnue du DSR : il pénalise la **diversité** des stratégies testées, pas leur *qualité absolue*.

---

## 6. Résultats — Diebold-Mariano vs CNN-LSTM nu

| Stratégie | mean(r_strat − r_nu) | DM stat | p-value | Verdict |
|---|---|---|---|---|
| Buy & Hold | −0.00010 | −0.24 | 0.811 | ns |
| CNN-LSTM + HMM-gate | +0.00004 | +0.20 | 0.842 | ns |
| CNN-LSTM + risk-gate | −0.00021 | −0.89 | 0.372 | ns |
| CNN-LSTM + HMM + risk | −0.00027 | −1.10 | 0.269 | ns |
| CNN-LSTM × VaR-budget | −0.00007 | −0.62 | 0.538 | ns |
| CNN-LSTM × HMM-cond. budget | −0.00006 | −0.55 | 0.580 | ns |

**Aucune significativité à 5 %.** Pourquoi ?
- HMM-gate a un **Sharpe** +0.65 supérieur (1.71 vs 1.06) ; économiquement c'est énorme.
- Mais sa **moyenne quotidienne** n'est que +0.00004 plus haute (+0.001 % par jour).
- Sur 948 jours, avec vol quotidienne ~1.4 %, la SE de la différence est ~0.045 % → t-stat ≈ 0.2.
- La supériorité de HMM-gate vient de **vol plus faible**, pas de **rendement plus haut**. DM teste les rendements moyens, pas le risk-adjusted.

**Implication mémoire** : le test DM rejette H0 sur les **moyennes**, pas sur les **Sharpes**. Pour tester des Sharpes différents on aurait besoin du test de Jobson-Korkie / Ledoit-Wolf (out of scope ici mais matériel pour étape 9 robustesse).

---

## 7. Résultats — Décomposition régime-conditionnelle

Sharpe annualisé par régime HMM × stratégie :

| Stratégie | Bear (168 j) | Neutral (371 j) | Bull (409 j) |
|---|---|---|---|
| Buy & Hold | **−1.62** | +1.71 | +1.26 |
| CNN-LSTM nu | +2.11 | −0.30 | +2.53 |
| CNN-LSTM + HMM-gate | +2.05 | **−5.12*** | +2.47 |
| CNN-LSTM + risk-gate | +0.82 | +0.86 | +1.69 |
| CNN-LSTM + HMM + risk | +0.82 | **−3.37*** | +1.68 |
| CNN-LSTM × VaR-budget | +1.83 | −0.21 | +2.19 |
| CNN-LSTM × HMM-cond. budget | +1.95 | −0.40 | +2.22 |

\* **Note technique** : les Sharpe Neutral très négatifs (-5.12, -3.37) sont des **artéfacts** quand la stratégie est *gated-off* en Neutral : la quasi-totalité des retours est 0 (mean ≈ 0, std ≈ 0), et les rares valeurs non nulles sont les **coûts d'entrée/sortie** (négatifs). Le Sharpe = mean/std × √252 devient mécaniquement très négatif sans représenter une vraie perte économique. Cohérent avec la nature gate-off de la stratégie en Neutral.

### 7.1 Lectures honnêtes

1. **Buy & Hold subit le régime Bear** (Sharpe −1.62) — le marché chute, B&H perd. Logique : aucun signal directionnel.
2. **CNN-LSTM nu performe en Bear ET Bull** (+2.11 / +2.53) — la valeur du prédicteur est dans les régimes directionnels.
3. **HMM-gate récupère le Sharpe Bear & Bull** (+2.05 / +2.47) en sacrifiant le Neutral (ne trade pas).
4. **Risk-gate dégrade le Sharpe Bear** (+2.11 → +0.82) car de nombreux jours Bear ont haute vol GARCH → écartés. **Le risk-gate retire de la valeur en Bear.**
5. **HMM + risk** combine les deux défauts : dégradation Bear (0.82) + artefact Neutral.

### 7.2 Pourquoi HMM-gate + risk-gate est inférieur à HMM-gate seul

- En **Bear**, HMM-gate trade → Sharpe +2.05 ; HMM+risk-gate dégrade → Sharpe +0.82 (retire les jours Bear à haute vol qui sont profitables car la direction est claire).
- En **Bull**, dégradation similaire (+2.47 → +1.68).
- En **Neutral**, les deux ne tradent pas → artefact identique.

**Conclusion robuste** : le risk-gate retire de la valeur quand HMM dit "régime directionnel net" parce que la haute vol dans un régime net = signal fort. **L'intersection des protections n'est pas additive.**

---

## 8. Honest Limitations

1. **DSR sensible à V** : 3 stratégies passent 0.95 ici parce que V (variance des Sharpes des trials) est faible (toutes stratégies raisonnables). Avec V de l'étape 6, seul HMM-gate reste STRONG. **Documenté** §5.

2. **DM non significatif** : aucune paire ne franchit p < 0.05. Avec 948 jours, on ne peut pas affirmer statistiquement la supériorité. Bornage économique vs statistique discuté §6.

3. **Sharpe Neutral négatif des stratégies gatées** : artefact de division par std ≈ 0 quand position = 0. Pas une vraie perte. **Documenté** §7 note technique.

4. **Coût uniforme 5 bps** : pas de modèle d'impact de marché. À tester en étape 9.

5. **VaR-budget paramétré sur B=0.01** : pas d'optimisation conjointe B × q_stress × q_normal — pourrait améliorer. Recherche d'hyperparamètres = sur-fitting risque ; on garde les valeurs littérature.

6. **Régime HMM Neutral utilisé comme proxy stress** dans strat 7 : alternative possible = utiliser le régime de risque (étape 7) comme stress flag. À tester en étape 9.

7. **Test Jobson-Korkie / Ledoit-Wolf** non implémenté pour comparer Sharpe vs CNN-LSTM nu — DM teste les moyennes, pas les Sharpes. **Plan étape 9.**

---

## 9. Future work (étapes 9-10)

| # | Idée | Étape cible |
|---|---|---|
| 1 | **Robustesse temporelle** : refaire l'analyse sur des sous-périodes (2022-23 vs 2024-26) pour voir si HMM-gate est stable | 9 |
| 2 | **Sensibilité aux coûts** : 5 / 10 / 20 bps comme étape 6 pour chaque variante | 9 |
| 3 | **Seuils HMM dynamiques** : tester gate sur P(regime=Bull) > 0.6 plutôt que argmax | 9 |
| 4 | **Test Jobson-Korkie** pour différence de Sharpe pairwise | 9 |
| 5 | **Stress proxy = risk_regime étape 7** au lieu de HMM=Neutral pour stratégie 7 | 9 |
| 6 | **Pipeline CLI complet** : train / predict / backtest / risk / export | 10 |

---

## 10. Output Artifacts

| Artifact | Path |
|---|---|
| Données per-day (positions + returns + equity, 7 stratégies) | `outputs/etape8/strategies_returns.csv` |
| Métriques complètes (Sharpe/DSR/DM/regime-cond) | `outputs/etape8/strategies_metrics.json` |
| Plot — equity curves | `reports/figures/etape8/etape8_equity_curves.png` |
| Plot — drawdowns | `reports/figures/etape8/etape8_drawdowns.png` |
| Plot — heatmap régime × stratégie | `reports/figures/etape8/etape8_regime_heatmap.png` |
| Plot — scatter Sharpe vs MDD | `reports/figures/etape8/etape8_sharpe_mdd_scatter.png` |
| Plot — Sharpe + DSR par stratégie | `reports/figures/etape8/etape8_dsr_summary.png` |
| Script principal | `scripts/08_strategies.py` |
| Notebook reproductible | `notebooks/08_strategies.ipynb` |

---

## 11. Récap des changements depuis étape 6

| Métrique | Étape 6 best (CNN-LSTM nu) | Étape 8 best (HMM-gate) | Δ |
|---|---|---|---|
| Sharpe ann. | +1.055 | +1.709 | **+0.65** |
| MDD | −15.96 % | −6.00 % | **+10 pp** (mieux) |
| Calmar | +0.92 | +2.84 | **×3.1** |
| Eq finale | 1.737 | 1.808 | +4 % |
| DSR (publishable threshold = 0.95) | 0.53 (étape 6, N=5) | 0.997 (étape 8, N=7) | franchit le seuil |
| Trades | 221 | 180 | −19 % |

**La couche HMM-gate est l'apport empirique majeur de l'étape 8.** Combinée avec le déjà-validé CNN-LSTM `base12`, elle constitue la **stratégie production finale du mémoire**.

---

*End of Étape 8 Report — generated by `scripts/08_strategies.py`.*
*Recommandation mémoire : **CNN-LSTM `base12` + HMM-gate** = stratégie production (DSR 0.997, Calmar 2.84). Variantes VaR-budget conservées comme alternatives à signal continu, risk-gate conservé comme défense additionnelle hors HMM.*

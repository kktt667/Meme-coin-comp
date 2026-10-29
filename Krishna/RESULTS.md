# Experiment log

All numbers come from the walk-forward harness over 37 days (2026-09-20 .. 2026-10-26). Each day trains
only on the official window: the 30 days before it, minus the last 2h. PnL uses the official payout.
"14d" is the rolling 14-day sum, the same shape as the final score. "exBest" is the total without the best day.

**Noise warning:** the same config with a different seed moves the 37-day total by about ±50.
Only compare bagged (5-seed) runs, and only trust a difference that passes `paired()`.

## Setup
- Features (`src/features.py`, cached per day in `data/cache/`):
  - **launch:** insider share of the curve, full-curve self-buy, bundle tx/tips/BEQ, dev SOL, metadata host, legacy token program
  - **curve:** trades, failed tx (rate, last 60s, wallets), duration, late-10m share, largest buy, top1/top5/gini of buyers, round-trippers, funded wallets, router share
  - **holders at migration:** count, top1/top10 share of the curve supply, creator's remaining share, gini
  - **first 60s on the AMM:** flow per 0-15/15-30/30-60s window, late/early ratio, largest buy, top1 share, ret60, at_high, curve holders dumping, creator selling, funded share, slot clustering, depth, migration pop
- Reputation (`src/reputation.py`): creator history, early-buyer history (curve first 30 + AMM buyers), shrunk hit rates relative to market, elite wallet counts, market 5x rate over the last resolved 24h. Every feature uses the cutoff day_start - 2h, so train and test features are equally stale.
- Model (`src/lab.py`): P(max_x >= L) for L in {3,4,5,7,10,15,25}, forced to fall as L rises. Expected PnL = P(L - 1.3) + (1 - P) x (population miss payoff). tp = argmax, and we buy if expected PnL > margin.

## Results

| run | margin | total | exBest | 14d worst | 14d median | takes/day | hit | notes |
|---|---|---|---|---|---|---|---|---|
| baseline 3 seeds (exp_baseline) | 0.25 | +106.7 | +41.8 | -29.2 | +5.8 | 17.8 | 20.9% | |
| baseline 3 seeds | 0.5 | +103.2 | +57.4 | -6.4 | +25.4 | 7.0 | 17.8% | 96% of 14d blocks positive; paired P(>0)=0.91 vs skip |
| baseline 3 seeds | 0.75 | +68.4 | +20.5 | -14.0 | +18.0 | 2.8 | 16.2% | |
| base 1 seed | 0.5 | +53.4 | +7.5 | -9.4 | +21.8 | 7.2 | 18.7% | the seed alone costs ~50 |
| all history instead of 30d, 1 seed | 0.5 | +29.6 | +9.3 | -27.5 | +25.9 | 6.3 | 17.2% | paired vs base30: -23.8, P=0.36, REJECT; keep the official 30d window |
| base 5 seeds (lab02 base5) | 0.25 | +53.9 | -13.6 | -29.8 | -2.9 | 17.3 | 20.0% | |
| **base 5 seeds** | **0.5** | **+69.5** | **+18.9** | **-10.9** | **+27.4** | 6.7 | 17.8% | **current reference** |
| base 5 seeds | 0.75 | +65.4 | +22.2 | -14.0 | +19.6 | 2.9 | 15.7% | |
| regularized (7 leaves, mcs 100, lr .02) | 0.5 | +24.3 | -31.9 | -26.8 | -2.0 | 5.0 | 12.0% | paired vs base5: -45, P=0.06, REJECT. It picks tp 10-25 too often |
| stacked: one model, level as a monotone input | 0.5 | +12.5 | +1.3 | -3.0 | -0.5 | 1.6 | 25.9% | -57, P=0.17, REJECT; too timid, only tp 3-4 |

### Calibration (base5, P(reach 5x), all eval days pooled)
The level means match well (5x: pred 0.055, real 0.057). **The top buckets are overconfident, and those are exactly the tokens we buy:**
pred 0.171 -> real 0.161 | pred 0.235 -> real 0.168 | pred 0.344 -> real 0.235.
Next: calibrate the top tail, e.g. isotonic/Platt fit on held-out days inside the window, or a shrink factor.

## Lab 03: v2 features, Platt calibration (2026-10-29)

| run | margin | total | exBest | 14d worst | 14d median | takes/day | hit | notes |
|---|---|---|---|---|---|---|---|---|
| v2 features (129), 5 seeds | 0.25 | +60.2 | +9.7 | -35.3 | -6.1 | 17.8 | 21.1% | |
| v2 features | 0.5 | +53.7 | +14.4 | -8.8 | +13.2 | 7.0 | 17.7% | paired vs v1: -15.8, P=0.26, REJECT |
| v2 features | 0.75 | +32.0 | -1.8 | -8.3 | +4.9 | 3.0 | 15.2% | |

v2 adds router/bot program shares, burner-wallet buyers, sniping in the first 2s, multi-sig, tip and v3-instruction shares,
and curve flow in the last 60s / last 5m. **Not accepted.** The cache is now v2, so v1 is pinned as `X_all_v1.parquet`.

**Platt calibration** (per level, fit on eval days D-15..D-2, compared on the 29 days that have that history):
the predictions do get calibrated (top bucket pred 0.167 -> real 0.171), but the policy gets **worse at every margin**
(v1 m=0.5: raw +27.8 -> cal +1.3). Calibrating each level separately lifts P at 15x/25x, so argmax moves to tp 15-25 and the hit rate falls to about 9%. REJECT.

**The sobering part:** on the last 29 days only (2026-09-28 onward), the reference (v1 base5, m=0.5) makes **+27.8**,
not +69.5. About 60% of the 37-day total came from the first 8 days. Lower margins lose money on those 29 days
(m=0.25: -42.8). The edge is real but thin, and most of it is in the top-confidence tail.

### Next ideas
1. Cap tp or drop the high levels (the EV policy picks tp 10-15 often; those are lottery tickets with high variance).
2. Use the model as an exclusion filter plus a fixed low tp (3-5), not a full EV argmax.
3. Rank-based depth fixed from training (like the official notebooks: threshold = the score that the top x% of past days cleared).
4. Feature pruning: 98-129 features on ~16k rows with ~900 positives at 5x is a lot.

## Lab 04: decision policy on cached v1 base5 predictions (no refits)

| policy | 37d total | 37d 14d worst | 29d total | 29d 14d worst | takes/day (29d) | hit (29d) |
|---|---|---|---|---|---|---|
| **EV argmax, no cap, m=0.5 (reference)** | **+69.5** | **-10.9** | **+27.8** | **-10.9** | 4.0 | 15.7% |
| EV, tp cap 10, m=0.5 | +21.8 | -9.9 | +10.8 | -9.9 | 3.8 | 16.5% |
| EV, tp cap 7, m=0.5 | +22.6 | -5.9 | +15.1 | -5.9 | 2.7 | 20.3% |
| EV, tp cap 5, m=0.5 | +4.2 | -11.0 | -0.4 | -2.0 | 1.3 | 21.6% |
| fixed tp=3, P>=0.4 | +49.2 | -6.4 | -9.4 | -6.4 | 3.1 | 31.1% |
| fixed tp=5, P>=0.3 | +13.2 | -2.0 | -0.5 | -2.0 | 0.8 | 20.8% |
| fixed tp=7, P>=0.2 | +21.8 | -5.6 | +12.5 | -5.6 | 1.5 | 18.6% |

**Capping tp only costs money.** The profit comes from the model's few tp 10-15 picks that land. Fixed low-tp policies
are break-even on the last 29 days. The edge is in spotting high-upside tokens, not in scalping 3x. Keep uncapped EV with m=0.5.
Next: make the high-level probabilities better (more seeds/bagging for the rare classes, feature pruning, a
max_x regression as a second opinion), not cap them away.

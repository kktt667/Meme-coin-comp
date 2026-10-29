# Krishna: development

My model work. The shared scoring code and data stay in `../harness/`. Everything here imports
`harness/src/common.py` through `src/paths.py`, so scripts run from any directory.

NOTES
before lab 6 - look at what consitutes a failing token and types of scams. For example a rugpull leaning token will more likely react differently when an unaffiliated retail wallet pops up - this is a key indicator. Also need better feature engineering for wallet reputation etc, we can also look at contract type because looking at contract details for hidden clauses and pool unlock codes immediately revels a token that is not community acked. Social indicators etc are also important prelaunch and token airdrops and supply checks important as well. Check and add details to features before lab 6. And quickly categorise features, so can test independednt ones as well. Current features have been described by claude in teh FEATURES.md
ONLY WORRY ABT TOP 30 all 98 have lower stability and bring teh accuracy down. 
Note - do not overfit on winners, look for loosing signals first then we look at what are similarities in winnnf tokens. 

| file | what it is |
|---|---|
| `src/features.py` | per-token features from the launch record and the curve + first-60s AMM tapes; `VERSION` names the cache |
| `src/reputation.py` | creator / early-buyer / market-regime history, cutoff day_start - 2h (leak-free) |
| `src/lab.py` | experiment loop: cached feature matrix, day-parallel LightGBM, cached out-of-day probabilities, `evaluate` / `summary` / `calib` |
| `src/exp_*.py` | one file per experiment; the results are logged in `RESULTS.md` |
| `cache/` | git-ignored: feature parquets, `X_all_v<N>.parquet`, `oof_<name>.pkl` |


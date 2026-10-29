# Krishna: development

My model work. The shared scoring code and data stay in `../harness/`. Everything here imports
`harness/src/common.py` through `src/paths.py`, so scripts run from any directory.

```
harness/.venv/bin/python Krishna/src/features.py      # build Krishna/cache/feat_<day>.parquet (about 4 min, once per data update)
harness/.venv/bin/python Krishna/src/exp_lab03.py     # run an experiment
```

| file | what it is |
|---|---|
| `src/features.py` | per-token features from the launch record and the curve + first-60s AMM tapes; `VERSION` names the cache |
| `src/reputation.py` | creator / early-buyer / market-regime history, cutoff day_start - 2h (leak-free) |
| `src/lab.py` | experiment loop: cached feature matrix, day-parallel LightGBM, cached out-of-day probabilities, `evaluate` / `summary` / `calib` |
| `src/exp_*.py` | one file per experiment; the results are logged in `RESULTS.md` |
| `cache/` | git-ignored: feature parquets, `X_all_v<N>.parquet`, `oof_<name>.pkl` |

Run experiments under `caffeinate -i` so the Mac doesn't sleep partway through (system sleep is set to 1 minute).

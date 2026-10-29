# Local test harness: memecoins-pump-or-dump

Test ideas locally instead of spending submissions. Each leaderboard score is one day
of a few heavy-tailed trades, so a single score says very little.
Ported from the S6E8 setup (`playground_season_6/src/common.py`).

```
cd harness
python src/exp_template.py        # ~1s; copy it to src/exp_<idea>.py per experiment
```

## Payout (from the Evaluation page)

- `tp = 0`: skip, pnl 0.
- `3 <= tp <= 100`: pnl = `tp - 1.3` if `max_x >= tp`, otherwise `max(final_x - 1.3, -1)`.
  `0.3` is the trading cost, and you can never lose more than 1.
- Any other value (e.g. 1.5, 2.9, 101) gets the whole file rejected. `pnl()` and `write_sub()` raise on it.
- Entry is the last trade at or before `mt + 60s`. `max_x` is the trimmed max over the
  next hour and `final_x` is the last price of that hour, both relative to entry.

## How it works

- **Walk-forward days that copy the official rotation.** For each eval day, `get_days()` trains on
  the 30 calendar days before it, minus the last 2 hours (their label hour overlaps the test
  day). It scores every day that has 30 full days of history before it: 37 days right now,
  2026-09-20 .. 2026-10-26, using the archive. `check_window()` confirms that rebuilding the window for
  the real test day gives exactly `train.csv`. For an experiment that also trains on
  older history, use `get_days(tr, train_days=None)`. The official notebook can do the same
  if it loads `train_archive.csv` itself.
- **A strategy is one function:** `strategy(train_df, eval_df) -> tp per row`.
  `eval_df` has `max_x`/`final_x` removed, so it can't peek at the answers.
- **`pnl()` turns tp into profit** using the payout above.
- **`report()`** also prints the worst and median **14-day block** sums. The final score is the
  sum of 14 days (1-14 Dec) with a frozen notebook, so consistency is what gets paid.
- **`paired(a, b)` replaces the 5/5 rule.** It accepts only if the variant wins more
  days than it loses, still gains after dropping its single best day, and a
  bootstrap over days gives P(gain > 0) ≥ 0.95.
- **`write_sub(te, tp)`** writes `mint,tp` after checking the mint set and tp values.

Baselines on the 37 days (tp=5): skip all = 0; buy all = -12114 (loses every day).
**Skip-all is the bar to beat.** Only ~13% of tokens reach 3x, ~5% reach 5x, and ~74% end the hour
below 0.3x (so they hit the -1 floor).

## What gets submitted

A **notebook that trains inside the run** on whatever 30+1 day window is mounted. No pre-trained
weights or hard-coded coefficients. The league bot re-runs it daily (4 CPU, 31 GB, 1 h wall time,
no network). Keep strategy code self-contained so the same function runs here and in the notebook.

## Leak rules for tape features

- The tapes are already cut at `mt + 60s`, so per-token tape features are safe as shipped.
- **Cross-token features that use labels** (creator history, wallet hit-rate, market regime)
  may only use tokens with `mt <= this_token.mt - 3600`, computed cumulatively. Never use
  `groupby().transform()` over the whole file. Test: a token's feature must not change if
  every later token is deleted.
- Curve tape: filter `failed == 0` before counting trades, volume or price.
  Keep the file order, and never re-sort by `ts`.

`data/` holds the small CSVs and is git-ignored. Put the day parquets there too.

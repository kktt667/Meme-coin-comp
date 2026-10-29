"""Shared loading, P&L simulator, and walk-forward harness (memecoins-pump-or-dump).

Payout, from the competition's Evaluation page:
  tp == 0          -> skip, pnl 0
  3 <= tp <= 100   -> pnl = tp - 1.3                  if max_x >= tp
                      pnl = max(final_x - 1.3, -1)    otherwise
  anything else is rejected by the scorer.
Official rotation: train on the 30 days before the test day, minus its last 2 hours; score one day.
Final score = sum of 14 consecutive daily PnLs (1-14 Dec), frozen notebook retrained each day.
"""
import pandas as pd, numpy as np

DATA = 'data'
SEED = 42
COST = 0.3                # fees + slippage, charged on every take
TP_MIN, TP_MAX = 3.0, 100.0
TRAIN_DAYS = 30           # official labelled window length
GAP_HOURS = 2             # official: last 2h before the test day are dropped from training
FINAL_DAYS = 14           # final score = sum of this many consecutive days
LABELS = ['max_x', 'final_x']

def load(data=DATA):
    """Archive + train as one labelled frame sorted by time, plus test. 'day' is UTC."""
    tr = pd.concat([pd.read_csv(f'{data}/train_archive.csv'), pd.read_csv(f'{data}/train.csv')])
    tr = tr.drop_duplicates('mint').sort_values('mt').reset_index(drop=True)
    te = pd.read_csv(f'{data}/test.csv')
    for d in (tr, te):
        d['day'] = pd.to_datetime(d['mt'], unit='s').dt.strftime('%Y-%m-%d')
    return tr, te

def check_tp(tp):
    """Raise if any tp is outside {0} U [3, 100] - the scorer rejects the whole file."""
    tp = np.asarray(tp, float)
    bad = ~((tp == 0) | ((tp >= TP_MIN) & (tp <= TP_MAX)))
    if bad.any():
        raise ValueError(f"{bad.sum()} tp values not in {{0}} U [{TP_MIN:g}, {TP_MAX:g}], e.g. {tp[bad][:5]}")
    return tp

def pnl(tp, max_x, final_x):
    """Per-token P&L under the official payout. tp == 0 means skip (P&L 0)."""
    tp = check_tp(tp)
    hit = np.where(np.asarray(max_x) >= tp, tp - 1 - COST,
                   np.maximum(np.asarray(final_x) - 1 - COST, -1.0))
    return np.where(tp > 0, hit, 0.0)

def train_index(tr, day, train_days=TRAIN_DAYS, gap_hours=GAP_HOURS):
    """Rows the official rotation would train on for test day `day`: the `train_days`
    calendar days before it, minus tokens migrating in the last `gap_hours` before it
    (their label hour overlaps the test day). train_days=None uses all earlier history."""
    start = pd.Timestamp(day).timestamp()
    lo = -np.inf if train_days is None else start - train_days * 86400
    m = (tr['mt'].values >= lo) & (tr['mt'].values < start - gap_hours * 3600)
    return np.flatnonzero(m)

def get_days(tr, n_eval=None, train_days=TRAIN_DAYS, gap_hours=GAP_HOURS):
    """Walk-forward splits - identical across every experiment so results are comparable.
    Each eval day is trained exactly like the official rotation (see train_index).
    n_eval=None scores every day that has a full train_days of history before it."""
    days = sorted(tr['day'].unique())
    first = pd.Timestamp(days[0]) + pd.Timedelta(days=train_days or 1)
    eligible = [d for d in days if pd.Timestamp(d) >= first]
    if n_eval is not None:
        eligible = eligible[-n_eval:]
    return [(d, train_index(tr, d, train_days, gap_hours),
             np.flatnonzero((tr['day'] == d).values)) for d in eligible]

def check_window(tr, te, data=DATA):
    """Sanity check: rebuilding the window for the real test day must give exactly train.csv."""
    got = set(tr['mint'].values[train_index(tr, te['day'].iloc[0])])
    want = set(pd.read_csv(f'{data}/train.csv', usecols=['mint'])['mint'])
    assert got == want, f"window mismatch: {len(got - want)} extra, {len(want - got)} missing"

def run(strategy, tr, splits):
    """strategy(train_df, eval_df) -> tp per eval row. eval_df has the labels removed,
    so a strategy cannot peek. Returns per-day results and out-of-day tp for every row."""
    rows, tp_oof = [], np.zeros(len(tr))
    for day, ti, vi in splits:
        tp = np.asarray(strategy(tr.iloc[ti], tr.iloc[vi].drop(columns=LABELS)), float)
        assert tp.shape == (len(vi),), f"{day}: strategy returned {tp.shape}, need ({len(vi)},)"
        p = pnl(tp, tr['max_x'].values[vi], tr['final_x'].values[vi])
        rows.append((day, p.sum(), int((tp > 0).sum()), len(vi)))
        tp_oof[vi] = tp
    res = pd.DataFrame(rows, columns=['day', 'pnl', 'takes', 'tokens']).set_index('day')
    return res, tp_oof

def report(name, res):
    p, t = res['pnl'], res['takes'].sum()
    best = p.max()
    print(f"\n{'='*62}\n{name}")
    print(f"  total {p.sum():+.2f} | mean/day {p.mean():+.3f} +/- {p.std():.3f} | worst {p.min():+.2f} best {best:+.2f}")
    print(f"  days positive {(p > 0).sum()}/{len(p)} | takes/day {t/len(p):.1f} | pnl/take {p.sum()/max(t,1):+.4f}")
    # Heavy tails: a 'good' strategy can be one lucky token. Show the total without the best day.
    print(f"  total excl. best day {p.sum() - best:+.2f}")
    # The final score is one 14-day sum, so show how every 14-day block in the backtest would have scored.
    if len(p) >= FINAL_DAYS:
        blk = p.rolling(FINAL_DAYS).sum().dropna()
        print(f"  {FINAL_DAYS}-day blocks: worst {blk.min():+.2f} median {blk.median():+.2f} | positive {(blk > 0).mean():.0%}")
    print('='*62)
    return p.sum()

def paired(a, b, name_a='variant', name_b='baseline', n_boot=10000, show_days=True):
    """Paired day-by-day verdict, a - b. Replaces S6E8's 5/5 rule: with ~30 noisy,
    heavy-tailed days, require (1) more winning days than losing, (2) the gain survives
    removing the single best day, (3) day-bootstrap P(total gain > 0) >= 0.95."""
    delta = (a['pnl'] - b['pnl']).values
    days = a.index
    print(f"\n{'#'*60}\nPAIRED VERDICT ({name_a} - {name_b}):")
    if show_days:
        for d, x, y, dl in zip(days, a['pnl'], b['pnl'], delta):
            tag = 'WIN' if dl > 0 else 'LOSS' if dl < 0 else 'tie'
            print(f"  {d}: {x:+8.2f} vs {y:+8.2f}  {dl:+8.2f} {tag}")
    wins, losses = (delta > 0).sum(), (delta < 0).sum()
    boot = np.random.default_rng(SEED).choice(delta, (n_boot, len(delta))).sum(1)
    p_pos = (boot > 0).mean()
    lo, hi = np.percentile(boot, [5, 95])
    excl_best = delta.sum() - delta.max()
    print(f"  total delta {delta.sum():+.2f}  (90% CI {lo:+.2f} .. {hi:+.2f})  P(>0) {p_pos:.3f}")
    print(f"  days won {wins} / lost {losses} / tied {len(delta)-wins-losses}  | delta excl. best day {excl_best:+.2f}")
    ok = wins > losses and excl_best > 0 and p_pos >= 0.95
    print("  VERDICT:", "ACCEPT" if ok else "REJECT")
    print('#'*60)
    return ok

def write_sub(te, tp, path='submission.csv', data=DATA):
    """Write mint,tp after the same checks the scorer applies: exact mint set, valid tp."""
    sub = pd.DataFrame({'mint': te['mint'], 'tp': check_tp(tp)})
    ss = pd.read_csv(f'{data}/sample_submission.csv')
    assert set(sub.mint) == set(ss.mint) and len(sub) == len(ss), "mints differ from sample_submission"
    sub.to_csv(path, index=False)
    print(f"wrote {path}: {len(sub)} rows, {(sub.tp > 0).sum()} takes")

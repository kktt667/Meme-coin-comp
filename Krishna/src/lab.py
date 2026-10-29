"""Fast experiment loop: cached feature matrix, day-parallel model fits, cached out-of-day probabilities.

    from lab import *
    X = build_X()                                   # tape + launch + reputation, one row per mint
    P = oof('base', FEATS)                          # {day: (n_eval, n_levels)} P(reach level), cached
    res = evaluate(P, margin=0.25)                  # per-day PnL frame for report()/paired()

Each eval day's models see only the official window (get_days), so P is honest out-of-day.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from common import *
from features import load_features, load_edges, CACHE, VERSION
from reputation import reputation

LEVELS = np.array([3, 4, 5, 7, 10, 15, 25], float)
NON_FEATS = {'mint', 'creator', 'host', 'created_ts', 'a_entry', 'mt', 'day', 'max_x', 'final_x'}
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=1)
WORKERS = 9

tr, te = load()
SPLITS = get_days(tr)

def build_X(force=False):
    path = f'{CACHE}/X_all_v{VERSION}.parquet'
    if os.path.exists(path) and not force:
        return pd.read_parquet(path)
    allx = pd.concat([tr, te], ignore_index=True)
    creators = pd.read_csv(f'{DATA}/pumpfun_token_creation.csv', usecols=['mint', 'creator']).set_index('mint')['creator'].to_dict()
    rep = reputation(allx[['mint', 'mt', 'day']], tr, load_edges(), creators)
    X = allx[['mint', 'mt', 'day']].merge(load_features(allx['mint']), on='mint', how='left').merge(rep, on='mint', how='left')
    X['age'] = X['mt'] - X['created_ts']
    X['hour'] = (X['mt'] // 3600) % 24
    X = X.set_index('mint')
    X.to_parquet(path)
    return X

X = build_X()
FEATS = [c for c in X.columns if c not in NON_FEATS and X[c].dtype != object]

def _fit_day(job):
    import lightgbm as lgb
    Xt, mx, Xe, feats, levels, params, seeds, w = job
    P = np.zeros((len(Xe), len(levels)))
    if params.get('topk'):
        # in-window feature selection: rank by gain from quick fits at 3x/5x/10x, keep the top k
        params = dict(params); k = params.pop('topk')
        gain = np.zeros(len(feats))
        for L in (3, 5, 10):
            m = lgb.LGBMClassifier(**params, random_state=99, importance_type='gain').fit(Xt[feats], (mx >= L).astype(int))
            gain += m.feature_importances_ / max(m.feature_importances_.sum(), 1e-9)
        feats = [feats[i] for i in np.argsort(-gain)[:k]]
    if params.get('stacked'):
        # one model over (token, level) pairs; P(reach) is forced to fall as the level rises
        p = {k: v for k, v in params.items() if k != 'stacked'}
        A = np.log(levels)
        Xs = pd.concat([Xt[feats].assign(log_level=a) for a in A], ignore_index=True)
        ys = np.concatenate([(mx >= L).astype(int) for L in levels])
        ws = None if w is None else np.tile(w, len(levels))
        mono = [0] * len(feats) + [-1]
        for s in seeds:
            m = lgb.LGBMClassifier(**p, random_state=s, monotone_constraints=mono).fit(Xs, ys, sample_weight=ws)
            for j, a in enumerate(A):
                P[:, j] += m.predict_proba(Xe[feats].assign(log_level=a))[:, 1] / len(seeds)
        return np.minimum.accumulate(P, axis=1)
    for j, L in enumerate(levels):
        y = (mx >= L).astype(int)
        for s in seeds:
            m = lgb.LGBMClassifier(**params, random_state=s).fit(Xt[feats], y, sample_weight=w)
            P[:, j] += m.predict_proba(Xe[feats])[:, 1] / len(seeds)
    return np.minimum.accumulate(P, axis=1)

def oof(name, feats, levels=LEVELS, params=PARAMS, seeds=(0,), splits=None, weight=None, force=False):
    """P(reach level) for every eval day, models fit on that day's window only. Cached by name.
    weight(train_df, day) -> sample weights, optional."""
    path = f'{CACHE}/oof_{name}.pkl'
    if os.path.exists(path) and not force:
        return pickle.load(open(path, 'rb'))
    splits = splits or SPLITS
    jobs = []
    for day, ti, vi in splits:
        t, v = tr.iloc[ti], tr.iloc[vi]
        w = weight(t, day) if weight else None
        jobs.append((X.loc[t['mint']], t['max_x'].values, X.loc[v['mint']], list(feats), np.asarray(levels), params, list(seeds), w))
    with ProcessPoolExecutor(WORKERS) as ex:
        out = dict(zip([d for d, _, _ in splits], ex.map(_fit_day, jobs)))
    pickle.dump(out, open(path, 'wb'))
    return out

def miss_payoff(train, levels=LEVELS):
    """Population payoff when a level is missed, from the training window."""
    mx, hold = train['max_x'].values, np.maximum(train['final_x'].values - 1 - COST, -1)
    return np.array([hold[mx < L].mean() for L in levels])

def expected(P, train, levels=LEVELS):
    return P * (np.asarray(levels) - 1 - COST) + (1 - P) * miss_payoff(train, levels)

def evaluate(P_by_day, margin=0.25, levels=LEVELS, splits=None, tp_cap=None, ev_fn=None):
    """Per-day PnL for the policy tp = argmax expected PnL if it clears `margin`."""
    splits = splits or SPLITS
    levels = np.asarray(levels)
    rows, tps = [], []
    for day, ti, vi in splits:
        E = ev_fn(P_by_day[day], tr.iloc[ti], day) if ev_fn else expected(P_by_day[day], tr.iloc[ti], levels)
        if tp_cap is not None:
            E = np.where(levels <= tp_cap, E, -np.inf)
        tp = np.where(E.max(1) > margin, levels[E.argmax(1)], 0.0)
        mx, fx = tr['max_x'].values[vi], tr['final_x'].values[vi]
        p = pnl(tp, mx, fx)
        rows.append((day, p.sum(), int((tp > 0).sum()), len(vi), int(((mx >= tp) & (tp > 0)).sum())))
        tps.append(tp[tp > 0])
    res = pd.DataFrame(rows, columns=['day', 'pnl', 'takes', 'tokens', 'hits']).set_index('day')
    res.attrs['tps'] = np.concatenate(tps)
    return res

def calib(P_by_day, levels=LEVELS, splits=None):
    """Mean predicted vs realised reach rate per level, and by predicted-probability bucket at 5x."""
    splits = splits or SPLITS
    P = np.vstack([P_by_day[d] for d, _, _ in splits]); vi = np.concatenate([v for _, _, v in splits])
    mx = tr['max_x'].values[vi]
    print('  level   pred   real')
    for j, L in enumerate(levels): print(f"  {L:5.0f} {P[:, j].mean():6.3f} {np.mean(mx >= L):6.3f}")
    j5 = list(levels).index(5)
    b = pd.cut(P[:, j5], [0, .05, .1, .15, .2, .3, .5, 1])
    print(pd.DataFrame({'pred': P[:, j5], 'real': mx >= 5}).groupby(b, observed=True).agg(n=('real', 'size'), pred=('pred', 'mean'), real=('real', 'mean')).round(3).to_string())

def summary(name, res):
    """One line: total, 14-day blocks, hit rate, takes/day, tp mix."""
    p = res['pnl']; blk = p.rolling(FINAL_DAYS).sum().dropna()
    tps = {int(k): int(v) for k, v in pd.Series(res.attrs.get('tps', [])).value_counts().sort_index().items()}
    print(f"{name:<34} total {p.sum():+8.2f} | exBest {p.sum()-p.max():+7.2f} | days+ {(p>0).sum():2d}/{len(p)} "
          f"| 14d worst {blk.min():+7.2f} med {blk.median():+7.2f} | takes/d {res.takes.mean():5.1f} "
          f"| hit {res.hits.sum()/max(res.takes.sum(),1):5.1%} | tp {tps}")

ZERO = None
def zero():
    global ZERO
    if ZERO is None:
        ZERO = pd.DataFrame({'pnl': 0.0}, index=[d for d, _, _ in SPLITS])
    return ZERO

def platt_calibrators(P_by_day, lookback=14, min_days=7, levels=LEVELS, splits=None):
    """For each eval day D, per-level Platt fits on the out-of-day predictions of eval days
    D-lookback-1 .. D-2 (all resolved before D's cutoff). Days without min_days of history get None.
    A notebook can reproduce this with an inner walk-forward over the end of its own window."""
    from sklearn.linear_model import LogisticRegression
    splits = splits or SPLITS
    days = [d for d, _, _ in splits]
    vis = {d: vi for d, _, vi in splits}
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    cal = {}
    for i, d in enumerate(days):
        prev = [x for x in days[max(0, i - lookback - 1):max(0, i - 1)]]
        if len(prev) < min_days:
            cal[d] = None; continue
        P = np.vstack([P_by_day[x] for x in prev]); mx = tr['max_x'].values[np.concatenate([vis[x] for x in prev])]
        fits = []
        for j, L in enumerate(levels):
            y = (mx >= L).astype(int)
            fits.append(LogisticRegression(C=1.0).fit(lg(P[:, [j]]), y) if 0 < y.sum() < len(y) else None)
        cal[d] = fits
    def apply(P, d):
        if cal.get(d) is None: return P
        Q = np.column_stack([f.predict_proba(lg(P[:, [j]]))[:, 1] if f is not None else P[:, j] for j, f in enumerate(cal[d])])
        return np.minimum.accumulate(Q, axis=1)
    return cal, apply

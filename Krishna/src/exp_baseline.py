"""Baseline: tape + launch + reputation features, one LightGBM per tp level, tp = argmax expected PnL.
Buy only if the best expected PnL clears MARGIN. Margins are compared from the same fitted models.
Run:  python Krishna/src/exp_baseline.py   (needs Krishna/cache from features.py)"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from common import *
from features import load_features, load_edges
from reputation import reputation

LEVELS = np.array([3, 4, 5, 7, 10, 15, 25], float)
MARGINS = [0.0, 0.1, 0.25, 0.5, 0.75]
SEEDS = [0, 1, 2]
DROP = {'mint', 'creator', 'host', 'created_ts', 'a_entry', 'mt', 'day', 'max_x', 'final_x'}
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=4)

tr, te = load()
check_window(tr, te)
allx = pd.concat([tr, te], ignore_index=True)
feat = load_features(allx['mint'])
creators = pd.read_csv(f'{DATA}/pumpfun_token_creation.csv', usecols=['mint', 'creator']).set_index('mint')['creator'].to_dict()
rep = reputation(allx[['mint', 'mt', 'day']], tr, load_edges(), creators)
X_all = allx[['mint', 'mt']].merge(feat, on='mint', how='left').merge(rep, on='mint', how='left')
X_all['age'] = X_all['mt'] - X_all['created_ts']
X_all = X_all.set_index('mint')
FEATS = [c for c in X_all.columns if c not in DROP]
print(f"{len(FEATS)} features; missing tape features for {X_all['c_n'].isna().sum()} tokens")

def expected_pnl(train, ev):
    """(n_eval, n_levels) matrix of expected PnL per level, from models fit on `train` only."""
    Xt, Xe = X_all.loc[train['mint'], FEATS], X_all.loc[ev['mint'], FEATS]
    mx = train['max_x'].values
    P = np.zeros((len(ev), len(LEVELS)))
    for j, L in enumerate(LEVELS):
        y = (mx >= L).astype(int)
        for s in SEEDS:
            P[:, j] += lgb.LGBMClassifier(**PARAMS, random_state=s).fit(Xt, y).predict_proba(Xe)[:, 1] / len(SEEDS)
    P = np.minimum.accumulate(P, axis=1)                 # P(reach L) cannot rise with L
    hold = np.maximum(train['final_x'].values - 1 - COST, -1)
    miss = np.array([hold[mx < L].mean() for L in LEVELS])   # population payoff when the level is missed
    return P * (LEVELS - 1 - COST) + (1 - P) * miss

def policy(E, margin):
    j = E.argmax(1)
    return np.where(E.max(1) > margin, LEVELS[j], 0.0)

splits = get_days(tr)
print(f"scoring {len(splits)} days: {splits[0][0]} .. {splits[-1][0]}")
E_by_day = {}
for day, ti, vi in splits:
    E_by_day[day] = expected_pnl(tr.iloc[ti], tr.iloc[vi].drop(columns=LABELS))
    print('.', end='', flush=True)

results = {}
for m in MARGINS:
    rows = []
    for day, ti, vi in splits:
        tp = policy(E_by_day[day], m)
        p = pnl(tp, tr['max_x'].values[vi], tr['final_x'].values[vi])
        rows.append((day, p.sum(), int((tp > 0).sum()), len(vi), int((tr['max_x'].values[vi] >= tp)[tp > 0].sum())))
    res = pd.DataFrame(rows, columns=['day', 'pnl', 'takes', 'tokens', 'hits']).set_index('day')
    results[m] = res
    report(f"EV margin {m}", res)
    tps = np.concatenate([policy(E_by_day[d], m) for d, _, _ in splits]); tps = tps[tps > 0]
    print(f"  hit rate {res.hits.sum()/max(res.takes.sum(),1):.1%} | tp used: {pd.Series(tps).value_counts().sort_index().to_dict()}")

base = results[MARGINS[0]] * 0
best = max(MARGINS, key=lambda m: results[m]['pnl'].sum())
paired(results[best], base, f'margin {best}', 'skip_all', show_days=False)

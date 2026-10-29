"""Lab 06: reputation v2 (smart money in vs out). Same v1 tape/launch features and the same model;
only the reputation block changes: AMM sellers and last-5m curve buyers added, elite in/out counts."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import lab
from lab import *
from features import load_edges
from reputation import reputation
import pickle

SEEDS5 = (0, 1, 2, 3, 4)
REP_PREFIX = ('wc_', 'wa_', 'cr_', 'm_')

if __name__ == '__main__':
    path = f'{CACHE}/X_v1_rep2.parquet'
    if not os.path.exists(path):
        X1 = pd.read_parquet(f'{CACHE}/X_all_v1.parquet')
        base = X1[[c for c in X1.columns if not c.startswith(REP_PREFIX)]]
        allx = pd.concat([tr, te], ignore_index=True)
        creators = pd.read_csv(f'{DATA}/pumpfun_token_creation.csv', usecols=['mint', 'creator']).set_index('mint')['creator'].to_dict()
        rep = reputation(allx[['mint', 'mt', 'day']], tr, load_edges(), creators).set_index('mint')
        base.join(rep).to_parquet(path)
    X2 = pd.read_parquet(path)
    lab.X = X2
    F2 = [c for c in X2.columns if c not in NON_FEATS and X2[c].dtype != object]
    new = [c for c in F2 if c.startswith(('ws_', 'wl_', 'elite'))]
    print(f'{len(F2)} features, new: {new}')
    d = tr.set_index('mint').join(X2[['elite_net', 'ws_elite', 'wa_elite', 'ws_s5_wavg']])
    for c in ('elite_net', 'ws_elite', 'ws_s5_wavg'):
        q = pd.qcut(d[c].rank(method='first'), 5, labels=False)
        print(f'  reach-5x by quintile of {c}:', (d.groupby(q)['max_x'].apply(lambda x: (x >= 5).mean()) * 100).round(1).tolist())

    ref = pickle.load(open(f'{CACHE}/oof_base5.pkl', 'rb'))
    ref30 = pickle.load(open(f'{CACHE}/oof_v1_top30.pkl', 'rb'))
    P = oof('rep2_base5', F2, seeds=SEEDS5)
    P30 = oof('rep2_top30', F2, params=dict(PARAMS, topk=30), seeds=SEEDS5)
    LATE = [s for s in SPLITS if s[0] >= '2026-09-28']
    for n, Q in (('ref all', ref), ('rep2 all', P), ('ref top30', ref30), ('rep2 top30', P30)):
        for m in (0.25, 0.5, 0.75):
            summary(f'{n} m={m}', evaluate(Q, m))
        summary(f'{n} m=0.5 29d', evaluate(Q, 0.5, splits=LATE))
    paired(evaluate(P, 0.5), evaluate(ref, 0.5), 'rep2 all', 'ref all', show_days=False)
    paired(evaluate(P30, 0.5), evaluate(ref30, 0.5), 'rep2 top30', 'ref top30', show_days=False)

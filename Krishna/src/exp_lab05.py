"""Lab 05: in-window feature selection (top-k by gain, picked inside each day's training window).
v1 features (the current reference). Compares k = 30, 50 against all 98."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import lab
from lab import *
import pickle

SEEDS5 = (0, 1, 2, 3, 4)

if __name__ == '__main__':
    # pin the v1 matrix: the reference was fit on it
    X1 = pd.read_parquet(f'{CACHE}/X_all_v1.parquet')
    lab.X = X1
    F1 = [c for c in X1.columns if c not in NON_FEATS and X1[c].dtype != object]
    ref = pickle.load(open(f'{CACHE}/oof_base5.pkl', 'rb'))
    runs = {'all98': ref}
    for k in (30, 50):
        runs[f'top{k}'] = oof(f'v1_top{k}', F1, params=dict(PARAMS, topk=k), seeds=SEEDS5)
    LATE = [s for s in SPLITS if s[0] >= '2026-09-28']
    for n, P in runs.items():
        for m in (0.25, 0.5, 0.75):
            summary(f'{n} m={m} 37d', evaluate(P, m))
        summary(f'{n} m=0.5 29d', evaluate(P, 0.5, splits=LATE))
    for k in (30, 50):
        paired(evaluate(runs[f'top{k}'], 0.5), evaluate(ref, 0.5), f'top{k}', 'all98', show_days=False)

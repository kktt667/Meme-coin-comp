"""Lab 03: v2 features vs v1, and Platt calibration of the top tail (on days that have calibration history)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
from lab import *
import pickle

SEEDS5 = (0, 1, 2, 3, 4)

if __name__ == '__main__':
    v1 = pickle.load(open(f'{CACHE}/oof_base5.pkl', 'rb'))
    V1_FEATS = [c for c in pd.read_parquet(f'{CACHE}/X_all_v1.parquet').columns if c in X.columns and c not in NON_FEATS]
    v2 = oof('base5_v2', FEATS, seeds=SEEDS5)
    print(f'v1 {len(V1_FEATS)} feats, v2 {len(FEATS)} feats')
    for m in [0.25, 0.5, 0.75]:
        summary(f'v1 m={m}', evaluate(v1, m)); summary(f'v2 m={m}', evaluate(v2, m))
    paired(evaluate(v2, 0.5), evaluate(v1, 0.5), 'v2', 'v1', show_days=False)
    calib(v2)
    # calibration, compared on the days that have >= 7 days of calibration history
    for name, P in (('v1', v1), ('v2', v2)):
        cal, apply = platt_calibrators(P)
        days_ok = [d for d, c in cal.items() if c is not None]
        sp = [s for s in SPLITS if s[0] in days_ok]
        Pc = {d: apply(P[d], d) for d in P}
        for m in [0.0, 0.1, 0.25, 0.5]:
            summary(f'{name} raw  m={m} ({len(sp)}d)', evaluate(P, m, splits=sp))
            summary(f'{name} cal  m={m} ({len(sp)}d)', evaluate(Pc, m, splits=sp))
        print(name, 'calibrated'); calib(Pc, splits=sp)

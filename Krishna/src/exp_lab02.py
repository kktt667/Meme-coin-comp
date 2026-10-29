"""Lab 02: 5-seed bags. base vs stronger regularisation vs one stacked model over levels."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
from lab import *

SEEDS5 = (0, 1, 2, 3, 4)
REG = dict(PARAMS, n_estimators=500, learning_rate=0.02, num_leaves=7, min_child_samples=100, colsample_bytree=0.5, reg_lambda=5.0)
STK = dict(REG, stacked=True)

if __name__ == '__main__':
    runs = {'base5': oof('base5', FEATS, seeds=SEEDS5),
            'reg5': oof('reg5', FEATS, params=REG, seeds=SEEDS5),
            'stk5': oof('stk5', FEATS, params=STK, seeds=SEEDS5)}
    for n, P in runs.items():
        for m in [0.25, 0.5, 0.75]: summary(f'{n} m={m}', evaluate(P, m))
    for n, P in runs.items():
        print(n); calib(P)
    paired(evaluate(runs['reg5'], 0.5), evaluate(runs['base5'], 0.5), 'reg5', 'base5', show_days=False)
    paired(evaluate(runs['stk5'], 0.5), evaluate(runs['base5'], 0.5), 'stk5', 'base5', show_days=False)

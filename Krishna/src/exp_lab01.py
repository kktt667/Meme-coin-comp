"""Lab 01: reproduce the baseline (1 seed, parallel) and test the training-window length."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
from lab import *

if __name__ == '__main__':
    P = oof('base', FEATS)
    for m in [0.25, 0.5, 0.75]: summary(f'base30 m={m}', evaluate(P, m))
    # all history before the cutoff (archive included) instead of the official 30 days
    allhist = get_days(tr, train_days=None)
    allhist = [(d, ti, vi) for (d, ti, vi) in allhist if d in {s[0] for s in SPLITS}]
    Pa = oof('allhist', FEATS, splits=allhist)
    for m in [0.25, 0.5, 0.75]: summary(f'allhist m={m}', evaluate(Pa, m))
    paired(evaluate(Pa, 0.5), evaluate(P, 0.5), 'allhist', 'base30', show_days=False)

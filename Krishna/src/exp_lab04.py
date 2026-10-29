"""Lab 04: decision policy only (no refits), on cached v1 base5 predictions.
tp caps, and fixed tp with a P(reach) threshold. Scored on all 37 days and on the last 29."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
from lab import *
import pickle

P = pickle.load(open(f'{CACHE}/oof_base5.pkl', 'rb'))
LATE = [s for s in SPLITS if s[0] >= '2026-09-28']

def fixed(P_by_day, L, thr, splits):
    """tp = L whenever P(reach L) >= thr."""
    j = list(LEVELS).index(L)
    rows, tps = [], []
    for day, ti, vi in splits:
        tp = np.where(P_by_day[day][:, j] >= thr, L, 0.0)
        mx, fx = tr['max_x'].values[vi], tr['final_x'].values[vi]
        rows.append((day, pnl(tp, mx, fx).sum(), int((tp > 0).sum()), len(vi), int(((mx >= tp) & (tp > 0)).sum())))
        tps.append(tp[tp > 0])
    res = pd.DataFrame(rows, columns=['day', 'pnl', 'takes', 'tokens', 'hits']).set_index('day')
    res.attrs['tps'] = np.concatenate(tps)
    return res

if __name__ == '__main__':
    for tag, sp in (('37d', SPLITS), ('29d', LATE)):
        print(f'\n--- {tag} ---')
        for cap in (None, 10, 7, 5):
            for m in (0.25, 0.5):
                summary(f'EV cap={cap} m={m}', evaluate(P, m, splits=sp, tp_cap=cap))
        for L, thrs in ((3, (0.40, 0.45, 0.5)), (4, (0.30, 0.35, 0.4)), (5, (0.20, 0.25, 0.30)), (7, (0.12, 0.15, 0.2))):
            for t in thrs:
                summary(f'fixed tp={L} P>={t}', fixed(P, L, t, sp))

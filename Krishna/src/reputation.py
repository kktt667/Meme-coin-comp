"""Creator / early-buyer reputation and market regime, leak-free by construction.

A token migrating on day D only sees outcomes of tokens with mt < D 00:00 - 2h: the same
cutoff the official run has for the test day (train.csv ends there). Training rows use their own
day's cutoff, so train-time and test-time features are equally stale. The history may include
train_archive.csv; the official notebook is allowed to load it.
"""
import numpy as np, pandas as pd

GAP = 7200
PRIOR_N = 10          # shrinkage strength (pseudo-tokens at the population rate)

def _day_start(day):
    return pd.Timestamp(day).timestamp()

def reputation(tokens, labelled, edges, creators):
    """tokens: mint, mt, day for every row needing features (train + eval).
    labelled: mint, mt, max_x, final_x (any labelled history). edges: mint, wallet, sol, role.
    creators: mint -> creator. Returns one row per token mint."""
    lab = labelled[['mint', 'mt', 'max_x', 'final_x']].sort_values('mt').reset_index(drop=True)
    lab['h3'] = (lab['max_x'] >= 3).astype(float)
    lab['h5'] = (lab['max_x'] >= 5).astype(float)
    lab['lmx'] = np.log(lab['max_x'].clip(lower=1e-3))
    lab['creator'] = lab['mint'].map(creators)
    # a wallet's track record comes from tokens it bought early (roles 0, 1, 3), never from selling
    e_lab = edges[edges['role'] != 2].drop_duplicates(['mint', 'wallet']).merge(lab[['mint', 'h3', 'h5', 'lmx']], on='mint')

    tok = tokens[['mint', 'mt', 'day']].copy()
    tok['creator'] = tok['mint'].map(creators)
    out = []
    w_stats = None; c_stats = None; added = 0
    lab_mt = lab['mt'].values
    for day in sorted(tok['day'].unique()):
        cut = _day_start(day) - GAP
        upto = np.searchsorted(lab_mt, cut, side='left')       # labelled tokens resolved by the cutoff
        if upto > added:
            new = lab.iloc[added:upto]
            ne = e_lab[e_lab['mint'].isin(set(new['mint']))]
            ws = ne.groupby('wallet').agg(n=('h5', 'size'), h3=('h3', 'sum'), h5=('h5', 'sum'), lmx=('lmx', 'sum'))
            cs = new.groupby('creator').agg(n=('h5', 'size'), h3=('h3', 'sum'), h5=('h5', 'sum'), lmx=('lmx', 'sum'))
            w_stats = ws if w_stats is None else w_stats.add(ws, fill_value=0)
            c_stats = cs if c_stats is None else c_stats.add(cs, fill_value=0)
            added = upto
        hist = lab.iloc[:upto]
        base5 = hist['h5'].mean() if upto else 0.05
        base3 = hist['h3'].mean() if upto else 0.13
        recent = hist[hist['mt'] >= cut - 86400]
        t = tok[tok['day'] == day]
        f = pd.DataFrame({'mint': t['mint'].values})
        # market regime over the last resolved 24h
        f['m_rate5_24h'] = recent['h5'].mean() if len(recent) else base5
        f['m_rate3_24h'] = recent['h3'].mean() if len(recent) else base3
        f['m_n_24h'] = len(recent)
        # creator history
        cs = c_stats.reindex(t['creator'].values) if c_stats is not None else None
        n = cs['n'].fillna(0).values if cs is not None else np.zeros(len(t))
        f['cr_n'] = n
        f['cr_h5'] = ((cs['h5'].fillna(0).values if cs is not None else 0) + PRIOR_N * base5) / (n + PRIOR_N)
        f['cr_lmx'] = np.where(n > 0, (cs['lmx'].fillna(0).values if cs is not None else 0) / np.maximum(n, 1), np.nan)
        # early-buyer history
        te = edges[edges['mint'].isin(set(t['mint']))]
        if w_stats is not None and len(te):
            s = w_stats.reindex(te['wallet'].values)
            wn = s['n'].fillna(0).values
            te = te.assign(known=(wn > 0).astype(float), wn=wn,
                           s5=(s['h5'].fillna(0).values + PRIOR_N * base5) / (wn + PRIOR_N),
                           s3=(s['h3'].fillna(0).values + PRIOR_N * base3) / (wn + PRIOR_N),
                           slmx=np.where(wn > 0, s['lmx'].fillna(0).values / np.maximum(wn, 1), np.nan))
            te['elite'] = ((te['wn'] >= 3) & (te['s5'] > 2 * base5)).astype(float)
            te['ws5'] = te['s5'] * te['sol']
            for role, tag in ((0, 'wc'), (1, 'wa'), (2, 'ws'), (3, 'wl')):
                r = te[te['role'] == role]
                g = r.groupby('mint').agg(known=('known', 'mean'), s5_mean=('s5', 'mean'), s5_max=('s5', 'max'),
                                          s3_mean=('s3', 'mean'), lmx=('slmx', 'mean'), elite=('elite', 'sum'),
                                          ws5=('ws5', 'sum'), sol=('sol', 'sum'), wn_med=('wn', 'median'))
                g['s5_wavg'] = g['ws5'] / g['sol'].clip(lower=1e-9)
                g = g.drop(columns=['ws5', 'sol'])
                g.columns = [f'{tag}_{c}' for c in g.columns]
                # relative to the market, so drifting base rates do not move the feature
                for c in (f'{tag}_s5_mean', f'{tag}_s5_max', f'{tag}_s5_wavg'):
                    g[c] = g[c] / base5
                f = f.merge(g, left_on='mint', right_index=True, how='left')
        # smart money in vs out during the first minute
        if 'wa_elite' in f and 'ws_elite' in f:
            f['elite_net'] = f['wa_elite'].fillna(0) - f['ws_elite'].fillna(0)
            f['elite_sell_share'] = f['ws_elite'].fillna(0) / (f['wa_elite'].fillna(0) + f['ws_elite'].fillna(0) + 1)
        out.append(f)
    return pd.concat(out, ignore_index=True)

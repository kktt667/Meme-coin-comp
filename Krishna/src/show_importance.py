"""Which features the reach-3x/5x/10x models lean on, fit on the latest official window (v1 features)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import lab
from lab import *
import lightgbm as lgb

X1 = pd.read_parquet(f'{CACHE}/X_all_v1.parquet')
F1 = [c for c in X1.columns if c not in NON_FEATS and X1[c].dtype != object]
day, ti, vi = SPLITS[-1]
t = tr.iloc[ti]; Xt = X1.loc[t['mint'], F1]
imp = {}
for L in (3, 5, 10):
    m = lgb.LGBMClassifier(**dict(PARAMS, n_jobs=2), random_state=0, importance_type='gain').fit(Xt, (t['max_x'] >= L).astype(int))
    imp[f'{L}x'] = m.feature_importances_ / m.feature_importances_.sum() * 100
df = pd.DataFrame(imp, index=F1); df['avg'] = df.mean(1)
print(f'window for {day}: {len(t)} tokens, {int((t.max_x>=5).sum())} reached 5x')
print(df.sort_values('avg', ascending=False).head(20).round(1).to_string())
print('\nshare of total gain by family:')
fam = lambda c: 'reputation' if c.startswith(('wc_','wa_','cr_','m_')) else 'launch' if c in ('insider_pct','full_curve_buy','bundle_size','bundled_buys_count','dev_sol','gas_used','n_instr','n_lookup_r','bs_tips','bs_other','bs_beq','bs_unknown','legacy_token','cashback','direct_pf','host_ipfs','name_len','sym_len','name_non_ascii','sym_upper','age') else 'holders' if c.startswith('h_') else 'curve' if c.startswith('c_') else 'first 60s' if c.startswith(('a_','mig')) else 'other'
print(df.groupby(df.index.map(fam))['avg'].sum().sort_values(ascending=False).round(1).to_string())

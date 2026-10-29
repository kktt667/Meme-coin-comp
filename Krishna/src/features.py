"""Per-token features from the launch record and the two tapes (curve + first 60s on the AMM).

Everything here uses only the token's own data up to mt + 60s, so it is leak-free by construction
and can be cached per migration day. Features that need other tokens' outcomes (creator / wallet
reputation) live in reputation.py and must be computed inside the strategy.

    python Krishna/src/features.py  # builds Krishna/cache/feat_<day>.parquet + edges_<day>.parquet
"""
import os, glob, sys
import numpy as np, pandas as pd
import paths
from paths import CACHE
from common import DATA
VERSION = 2                          # bump when tape features change; the cache is rebuilt
CURVE_SUPPLY = 793.1e6 * 1e6        # tokens sold on the bonding curve, base units
CURVE_TOK = 793.1e6                 # same, in whole tokens (the tapes' `tok` unit)

CURVE_COLS = ['mint', 'wallet', 'fee_payer', 'direction', 'sol', 'tok', 'ts_rel', 'failed',
              'new_price', 'slot', 'parent_program', 'min_amount_out', 'max_amount_in', 'tip_index',
              'num_signatures', 'instruction_type']
AMM_COLS = ['mint', 'signing_wallet', 'fee_payer', 'direction', 'sol', 'tok', 'ts_rel', 'new_price',
            'slot', 'pool', 'coin_creator', 'pool_quote_token_reserves_after', 'virtual_quote_reserves',
            'parent_program', 'num_signatures', 'user_quote_token_reserves', 'is_exact_quote']
# router / bot programs that carry most routed flow (prefixes of parent_program)
PROGS = ['FLASHX8', '6Vo3245', 'DF1ow4', 'term9YP', 'JUP6', 'GMgn', 'ZEPHYR', 'BmGwq']

def _prog_shares(pp, sol, tag):
    pp = pp.fillna('none'); tot = max(sol.sum(), 1e-9)
    d = {f'{tag}_prog_{p[:6]}': sol[pp.str.startswith(p)].sum() / tot for p in PROGS}
    d[f'{tag}_prog_none'] = sol[pp == 'none'].sum() / tot
    return d

def tape_path(day, table):
    return f'{DATA}/day_{day}_{table}_all_swaps.parquet'

def available_days():
    return sorted(os.path.basename(f)[4:14] for f in glob.glob(f'{DATA}/day_*_pumpswap_all_swaps.parquet'))

# ---------------------------------------------------------------- launch record
def launch_features(creation):
    """One row per mint from pumpfun_token_creation.csv. No outcome data involved."""
    c = creation
    bs = c['bundle_structure'].fillna('')
    host = c['url'].fillna('').str.extract(r'https?://([^/]+)')[0].fillna('none')
    f = pd.DataFrame({
        'mint': c['mint'],
        'created_ts': c['created_ts'],
        'creator': c['creator'],
        'insider_pct': c['bundled_buys'] / CURVE_SUPPLY,               # share of the curve bought at creation
        'full_curve_buy': (c['bundled_buys'] >= 0.99 * CURVE_SUPPLY).astype(int),
        'bundle_size': c['bundle_size'],
        'bundled_buys_count': c['bundled_buys_count'],
        'dev_sol': np.log1p(c['dev_balance'] / 1e9),
        'gas_used': c['gas_used'],
        'n_instr': c['amount_of_instructions'],
        'n_lookup_r': c['amount_of_lookup_reads'],
        'bs_tips': bs.str.count(r'\dT(?::|\b)'),
        'bs_other': bs.str.count(r'\dO\b'),
        'bs_beq': bs.str.count(r'BEQ'),
        'bs_unknown': bs.str.count(r'UNKNOWN'),
        'legacy_token': c['token_program'].str.startswith('Tokenkeg').astype(int),
        'cashback': c['is_cashback_enabled'],
        'direct_pf': c['direct_pf_invocation'],
        'host_ipfs': (host == 'ipfs.io').astype(int),
        'host': host,                                                  # categorical, encoded in the model
        'name_len': c['name'].fillna('').str.len(),
        'sym_len': c['symbol'].fillna('').str.len(),
        'name_non_ascii': c['name'].fillna('').map(lambda s: any(ord(ch) > 127 for ch in s)).astype(int),
        'sym_upper': (c['symbol'].fillna('') == c['symbol'].fillna('').str.upper()).astype(int),
    })
    return f

# ---------------------------------------------------------------- tapes
def _gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if n == 0 or x.sum() <= 0: return np.nan
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())

def _shares(v, k):
    v = np.sort(np.asarray(v, float))[::-1]; s = v.sum()
    return v[:k].sum() / s if s > 0 else np.nan

def curve_token(g, creator):
    """g: one mint's curve rows, file order. Returns dict of features."""
    ok = g[g['failed'] == 0]; bad = g[g['failed'] != 0]
    buy = ok[ok['direction'] == 'buy']; sell = ok[ok['direction'] == 'sell']
    bsol = buy['sol'].sum(); ssol = sell['sol'].sum()
    t0 = -ok['ts_rel'].min() if len(ok) else np.nan
    w_buy = buy.groupby('wallet')['sol'].sum()
    # wallet distribution at migration: net tokens held per wallet from curve trading
    net = ok.assign(s=np.where(ok['direction'] == 'buy', ok['tok'], -ok['tok'])).groupby('wallet')['s'].sum()
    hold = net[net > 0]
    sellers = set(sell['wallet']); buyers = set(buy['wallet'])
    late = buy[buy['ts_rel'] > -600]['sol'].sum()
    d = {
        'c_n': len(ok), 'c_failed': len(bad), 'c_fail_rate': len(bad) / max(len(g), 1),
        'c_fail_last60': (bad['ts_rel'] > -60).sum(),
        'c_fail_uw': bad['wallet'].nunique(),
        'c_uw': ok['wallet'].nunique(), 'c_buyers': len(buyers),
        'c_buy_sol': bsol, 'c_sell_sol': ssol, 'c_sell_share': ssol / max(bsol, 1e-9),
        'c_dur': t0, 'c_late10m_share': late / max(bsol, 1e-9),
        'c_buy_max': buy['sol'].max() if len(buy) else 0.0,
        'c_top1': _shares(w_buy, 1), 'c_top5': _shares(w_buy, 5), 'c_gini': _gini(w_buy),
        'c_roundtrip': len(buyers & sellers) / max(len(buyers), 1),
        'c_funded': (ok['fee_payer'] != ok['wallet']).mean() if len(ok) else np.nan,
        'c_router': ok['parent_program'].notna().mean() if len(ok) else np.nan,
        'h_n': len(hold),
        'h_top1': hold.max() / CURVE_TOK if len(hold) else np.nan,
        'h_top10': np.sort(hold.values)[::-1][:10].sum() / CURVE_TOK if len(hold) else np.nan,
        'h_gini': _gini(hold.values),
        'h_creator': max(net.get(creator, 0.0), 0.0) / CURVE_TOK,
        'c_creator_sold': sell.loc[sell['wallet'] == creator, 'sol'].sum(),
        'c_tip_share': (ok['tip_index'] >= 0).mean() if len(ok) else np.nan,
        'c_multisig': (ok['num_signatures'] > 1).mean() if len(ok) else np.nan,
        'c_v3_share': ok['instruction_type'].str.endswith('v3').mean() if len(ok) else np.nan,
        'c_last60_buy_sol': buy.loc[buy['ts_rel'] > -60, 'sol'].sum(),
        'c_last5m_buyers': buy.loc[buy['ts_rel'] > -300, 'wallet'].nunique(),
        'c_fail_buy_last5m': (bad.loc[bad['ts_rel'] > -300, 'direction'] == 'buy').sum(),
    }
    d.update(_prog_shares(buy['parent_program'], buy['sol'], 'c'))
    return d, net

def amm_token(g, net_curve, creator):
    """g: one mint's AMM rows (first 60s), file order."""
    g = g[g['new_price'] > 0]
    if not len(g):
        return {'a_n': 0}
    buy = g[g['direction'] == 'B']; sell = g[g['direction'] == 'S']
    p = g['new_price'].values; entry = p[-1]
    win = lambda lo, hi, x: x[(x['ts_rel'] >= lo) & (x['ts_rel'] < hi)]['sol'].sum()
    b0, b1, b2 = win(0, 15, buy), win(15, 30, buy), win(30, 61, buy)
    s0, s1, s2 = win(0, 15, sell), win(15, 30, sell), win(30, 61, sell)
    w_vol = g.groupby('signing_wallet')['sol'].sum()
    holders = set(net_curve[net_curve > 0].index)
    curve_dump = sell[sell['signing_wallet'].isin(holders)]
    eff_q = (g['pool_quote_token_reserves_after'] + g['virtual_quote_reserves']).values[-1] / 1e9
    return {
        'a_n': len(g), 'a_uw': g['signing_wallet'].nunique(),
        'a_buy_sol': buy['sol'].sum(), 'a_sell_sol': sell['sol'].sum(),
        'a_net_sol': buy['sol'].sum() - sell['sol'].sum(),
        'a_buy_share': buy['sol'].sum() / max(g['sol'].sum(), 1e-9),
        'a_b0': b0, 'a_b1': b1, 'a_b2': b2, 'a_net2': b2 - s2,
        'a_late_over_early': (b2 + 1e-3) / (b0 + 1e-3),
        'a_late_buy_share': b2 / max(b2 + s2, 1e-9),
        'a_buy_max': buy['sol'].max() if len(buy) else 0.0,
        'a_top1': _shares(w_vol, 1), 'a_gini': _gini(w_vol),
        'a_ret60': entry / p[0], 'a_at_high': entry / p.max(), 'a_low': p.min() / entry,
        'a_vol': np.std(np.diff(np.log(p))) if len(p) > 2 else 0.0,
        'a_new_wallets': 1 - np.mean(g['signing_wallet'].isin(set(net_curve.index))),
        'a_curve_dump_sol': curve_dump['sol'].sum(),
        'a_curve_dump_n': curve_dump['signing_wallet'].nunique(),
        'a_creator_sold': sell.loc[sell['signing_wallet'] == creator, 'sol'].sum(),
        'a_funded': (g['fee_payer'] != g['signing_wallet']).mean(),
        'a_slot_cluster': g.duplicated('slot', keep=False).mean(),
        'a_pools': g['pool'].nunique(),
        'a_depth_sol': eff_q,
        'a_entry': entry,
        'a_snipe_sol': buy.loc[buy['ts_rel'] <= 2, 'sol'].sum(),
        'a_snipe_n': buy.loc[buy['ts_rel'] <= 2, 'signing_wallet'].nunique(),
        'a_multisig': (g['num_signatures'] > 1).mean(),
        'a_exact_quote': g['is_exact_quote'].mean(),
        'a_burner_share': (buy['user_quote_token_reserves'] < 1e7).mean() if len(buy) else np.nan,
        'a_buyer_bal_med': np.log1p(buy['user_quote_token_reserves'].median() / 1e9) if len(buy) else np.nan,
        'a_buyer_bal_sum': np.log1p(buy['user_quote_token_reserves'].sum() / 1e9) if len(buy) else np.nan,
        **_prog_shares(buy['parent_program'], buy['sol'], 'a'),
    }

def build_day(day, creators):
    """Features + early-buyer edges for every token migrating on `day`."""
    cv = pd.read_parquet(tape_path(day, 'pumpfun'), columns=CURVE_COLS)
    am = pd.read_parquet(tape_path(day, 'pumpswap'), columns=AMM_COLS)
    rows, edges = [], []
    am_by = dict(tuple(am.groupby('mint', sort=False)))
    for mint, g in cv.groupby('mint', sort=False):
        cr = creators.get(mint)
        d, net = curve_token(g, cr)
        a = am_by.get(mint)
        d.update(amm_token(a, net, cr) if a is not None else {'a_n': 0})
        if 'a_entry' in d and len(g):
            last_curve = g.loc[(g['failed'] == 0) & (g['new_price'] > 0), 'new_price']
            d['mig_pop'] = d['a_entry'] / last_curve.iloc[-1] if len(last_curve) else np.nan
        d['mint'] = mint
        rows.append(d)
        # wallet edges for reputation. role 0: first 30 curve buyers, 1: AMM buyers in the minute,
        # 2: AMM sellers in the minute, 3: curve buyers in the last 5 minutes before migration
        ok = g[(g['failed'] == 0) & (g['direction'] == 'buy')]
        first = ok.groupby('wallet', sort=False)['sol'].sum().head(30)
        edges.append(pd.DataFrame({'mint': mint, 'wallet': first.index, 'sol': first.values, 'role': 0}))
        late = ok[ok['ts_rel'] > -300].groupby('wallet', sort=False)['sol'].sum()
        edges.append(pd.DataFrame({'mint': mint, 'wallet': late.index, 'sol': late.values, 'role': 3}))
        if a is not None:
            ab = a[a['direction'] == 'B'].groupby('signing_wallet', sort=False)['sol'].sum()
            edges.append(pd.DataFrame({'mint': mint, 'wallet': ab.index, 'sol': ab.values, 'role': 1}))
            sl = a[a['direction'] == 'S'].groupby('signing_wallet', sort=False)['sol'].sum()
            edges.append(pd.DataFrame({'mint': mint, 'wallet': sl.index, 'sol': sl.values, 'role': 2}))
    feat = pd.DataFrame(rows)
    feat['c_n_total'] = feat['c_n'] + feat['a_n']
    return feat, pd.concat(edges, ignore_index=True)

def build_cache(days=None, force=False):
    os.makedirs(CACHE, exist_ok=True)
    creators = pd.read_csv(f'{DATA}/pumpfun_token_creation.csv', usecols=['mint', 'creator']).set_index('mint')['creator'].to_dict()
    for day in days or available_days():
        out = f'{CACHE}/feat_{day}.parquet'
        if os.path.exists(out) and not force:
            continue
        if not os.path.exists(tape_path(day, 'pumpfun')):
            print('missing curve tape for', day); continue
        feat, edges = build_day(day, creators)
        feat.to_parquet(out); edges.to_parquet(f'{CACHE}/edges_{day}.parquet')
        print(day, len(feat), 'tokens', len(edges), 'edges', flush=True)

def load_features(mints=None):
    """All cached tape features joined to launch features, one row per mint."""
    f = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{CACHE}/feat_*.parquet'))], ignore_index=True)
    lf = launch_features(pd.read_csv(f'{DATA}/pumpfun_token_creation.csv'))
    f = f.merge(lf, on='mint', how='left')
    return f if mints is None else f[f['mint'].isin(set(mints))]

def load_edges():
    return pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{CACHE}/edges_*.parquet'))], ignore_index=True)

if __name__ == '__main__':
    build_cache(sys.argv[1:] or None)

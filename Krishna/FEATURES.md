# Features: what the best model so far uses

This covers the **v1 feature set (98 features)** used by the reference model (`oof_base5`) and by lab05 top-30.
Lab05 top-30 picks the 30 strongest of these inside each training window. Code:
[src/features.py](src/features.py) (tape and launch) and [src/reputation.py](src/reputation.py) (history).
Results are in [RESULTS.md](RESULTS.md).

## How to read the tables

- **gain%**: the share of the model's total split gain that goes to this feature. It is averaged over the 3x/5x/10x models
  and 6 training windows spread over the backtest. It tells you how much the model *uses* the feature.
- **top30**: in how many of those 6 windows the feature made the top 30. A 6 means it is always picked.
- **Q1 / Q5**: the % of tokens that reach 5x among the 20% with the *lowest* (Q1) and *highest* (Q5) values of the
  feature, across all 38.9k labelled tokens. The base rate is about 5%. A big gap means the feature separates
  winners from losers on its own.
- **rho**: Spearman correlation with log(max_x). Positive means higher values tend to come with higher peaks.

Two cautions:
1. **Gain undersells features that overlap others.** If two features measure the same thing (holder
   concentration and curve-buyer concentration), the trees pick one and ignore the other. A low gain doesn't
   mean the idea is useless. It may already be covered.
2. **The Q1/Q5 numbers are descriptive** (computed on all labelled data). Don't use them to pick features for the model:
   that would leak the future into the backtest. The model picks its own features inside each window.

## The pipeline in one paragraph

For each token we stand at **migration + 60s**. We see the launch record, every bonding-curve trade
(including failed ones), and the first 60 seconds on the AMM pool. [features.py](src/features.py) turns that into one row per
token and caches it per day. [reputation.py](src/reputation.py) adds history (creators, wallets, market), using only tokens
whose outcomes were known before **the token's day start minus 2h**. Seven LightGBM classifiers predict P(reach 3x, 4x, 5x,
7x, 10x, 15x, 25x). We pick the tp with the best expected PnL and buy only if it beats 0.5.

---

## 1. Wallet and creator reputation (about 36% of the model's gain)

**What it means:** some wallets keep showing up early in tokens that run (snipers with good taste, insiders, KOL
followers). Others keep showing up in rugs. If we know the track record of the wallets buying *this* token, we know a lot.

**How it's computed:** for each token we record two groups of wallets (the "edges"):
- **wc**: the first 30 distinct buyers on the bonding curve, with the SOL each put in
- **wa**: every buyer in the first 60s on the AMM, with the SOL each put in

For every wallet we keep a running record over the tokens it bought early that have *already resolved*: how many tokens,
how many hit 3x and 5x, and the average log peak. Hit rates are **shrunk** toward the market rate:
`(hits + 10 * base_rate) / (n + 10)`. That way a wallet with 1 lucky token doesn't look like a genius. The `s5` scores are then
divided by the market 5x rate, so 1.0 means "average wallet" and 2.0 means "twice as good as average".

| feature | meaning | gain% | top30 | Q1 | Q5 | rho |
|---|---|---|---|---|---|---|
| `wa_lmx` | first-minute AMM buyers: average log peak multiple of the tokens they bought before | **7.85** | 6 | 0.8 | 9.4 | **+0.200** |
| `wa_s5_mean` | AMM buyers: mean shrunk 5x hit rate (relative to the market) | 4.09 | 6 | 1.0 | 10.2 | +0.115 |
| `wa_s5_wavg` | the same, weighted by SOL bought (whales count more) | 3.63 | 6 | 1.5 | 10.1 | +0.100 |
| `wa_s3_mean` | AMM buyers: mean shrunk 3x hit rate | 3.29 | 6 | 0.6 | 9.6 | +0.153 |
| `wa_s5_max` | the best single AMM buyer's 5x score | 1.99 | 6 | 1.3 | 9.2 | +0.116 |
| `wa_known` | share of AMM buyers with any history | 1.58 | 6 | 3.8 | 3.3 | -0.003 |
| `wa_wn_med` | median number of past tokens per AMM buyer (how "professional" the crowd is) | 1.23 | 3 | 3.5 | 5.4 | +0.043 |
| `wa_elite` | count of AMM buyers with 3+ past tokens and more than 2x the market 5x rate | 1.22 | 3 | 1.4 | 9.5 | +0.118 |
| `wc_lmx` | first 30 curve buyers: average log peak of their past tokens | 2.19 | 6 | 3.2 | 8.9 | +0.089 |
| `wc_s3_mean` / `wc_s5_mean` / `wc_s5_wavg` / `wc_s5_max` | the same scores for the early curve buyers | 0.8-1.4 | 0-5 | 1.5-3.3 | 7.9-8.4 | |
| `wc_known`, `wc_wn_med`, `wc_elite` | coverage / experience / elite count for early curve buyers | 0.2-0.8 | 0 | | | |
| `cr_h5` | creator's shrunk 5x hit rate over their earlier launches | 1.55 | 5 | 5.0 | 5.6 | +0.034 |
| `cr_n` | number of earlier resolved launches by this creator | 0.32 | 0 | 3.4 | 5.9 | -0.022 |
| `cr_lmx` | creator's average log peak (NaN for first-time creators) | 0.68 | 0 | 5.9 | 5.7 | +0.039 |
| `m_rate5_24h` / `m_rate3_24h` | market: share of tokens reaching 5x / 3x in the last resolved 24h | 0.8-0.9 | 0 | 3.7 | 6.5-6.7 | |
| `m_n_24h` | market: tokens in the last resolved 24h (activity level) | 1.06 | 3 | 4.0 | 5.8 | +0.025 |

**Takeaway:** the *AMM crowd's* history (`wa_*`) beats the early curve buyers' (`wc_*`). The people who show up in the
first minute after migration are the best single signal we have. Q1 to Q5 goes from about 1% to about 10%.

**Leak rule:** a token on day D only sees outcomes of tokens with `mt < D 00:00 - 2h`. Training rows use the same
rule for their own day, so the features are equally stale in training and at test time. Never compute these with a
`groupby` over the whole file.

---

## 2. First 60 seconds on the AMM (about 32%)

**What it means:** migration is the debut. In 60 seconds you see whether real demand shows up, whether it's
accelerating or fading, and whether a few wallets are running the show. Trades are split into time windows
**0-15s / 15-30s / 30-60s**, because the *shape* matters more than the total.

| feature | meaning | gain% | top30 | Q1 | Q5 | rho |
|---|---|---|---|---|---|---|
| `a_at_high` | entry price / highest price in the minute (1 = we're buying at the top) | 1.59 | 6 | 6.8 | **2.6** | -0.037 |
| `a_buy_max` | largest single buy in SOL | 1.54 | 5 | 5.1 | **1.0** | -0.094 |
| `a_b0` | buy SOL in 0-15s (the opening rush) | 1.52 | 6 | 5.6 | 1.4 | -0.104 |
| `a_net_sol` | buy SOL - sell SOL over the minute | 1.48 | 5 | 6.7 | 1.6 | -0.055 |
| `a_gini` | inequality of volume across wallets (1 = one wallet does everything) | 1.43 | 4 | 6.1 | 2.8 | -0.058 |
| `a_vol` | volatility of log price changes trade to trade | 1.42 | 6 | 5.5 | 1.8 | -0.096 |
| `a_funded` | share of trades where the fee payer is not the signer (someone paying for other wallets: bot farms) | 1.41 | 5 | 3.0 | 5.6 | +0.004 |
| `a_ret60` | entry price / first AMM price (how far it already ran) | 1.35 | 3 | 6.7 | 1.3 | -0.056 |
| `a_late_over_early` | buy SOL in 30-60s / buy SOL in 0-15s (accelerating vs fading) | 1.33 | 5 | 2.1 | **6.3** | **+0.119** |
| `a_top1` | top wallet's share of volume | 1.32 | 4 | 6.6 | 1.8 | -0.076 |
| `a_low` | lowest price in the minute / entry (a deep dip that recovered) | 1.31 | 5 | 1.4 | 6.1 | +0.087 |
| `a_late_buy_share` | buys / (buys + sells) in 30-60s | 1.26 | 4 | 5.4 | 3.0 | -0.038 |
| `a_net2` | net SOL in 30-60s | 1.26 | 4 | 6.3 | 5.7 | -0.002 |
| `a_buy_share` | buy share of all volume | 1.19 | 4 | 6.2 | 2.6 | -0.031 |
| `a_new_wallets` | share of AMM trades by wallets never seen on the curve | 1.17 | 1 | 6.0 | 3.8 | -0.041 |
| `a_b2`, `a_b1` | buy SOL in 30-60s, 15-30s | 1.1-1.2 | 1-4 | 3.5-3.7 | 7.5-7.9 | +0.05..0.09 |
| `a_slot_cluster` | share of trades landing in the same slot as another (bots and bundles) | 1.15 | 4 | 4.4 | 5.9 | +0.047 |
| `a_uw`, `a_n` | unique wallets and trade count | 0.8-1.0 | 0-2 | | | |
| `a_depth_sol` | pool SOL (effective reserves) at the snapshot | 0.91 | 1 | 6.7 | 1.3 | -0.055 |
| `a_curve_dump_sol` / `_n` | SOL sold in the minute by wallets holding from the curve, and how many | 0.5-0.9 | 0 | 2.4 | 7.5 | +0.055 |
| `a_sell_sol`, `a_buy_sol` | total sell / buy SOL | 0.8-0.9 | 0 | | | |
| `a_creator_sold` | creator's SOL sold in the minute | 0.41 | 0 | 4.3 | 5.2 | |
| `mig_pop` | first-minute end price / last curve price (the jump at migration) | 0.86 | 1 | 6.7 | 1.3 | -0.056 |
| `a_pools` | number of pools traded (always 1 in this universe, so it's useless) | 0 | 0 | | | |

**Takeaways:**
- **Already pumped = bad.** High `a_ret60`, `mig_pop` or `a_at_high` (buying the top) gives about 1-2.5% vs 6-7%.
- **One whale or one wallet = bad.** A big `a_buy_max`, `a_top1` or `a_gini` usually means someone is setting up the exit.
- **Late money = good.** `a_late_over_early` and `a_b2`: money still arriving at 30-60s beats a big opening rush (`a_b0` is negative).
- **Curve holders selling into the debut is *good*** (`a_curve_dump_sol` Q5 7.5%). It looks like healthy rotation from
  curve buyers to new buyers. Counterintuitive, and worth digging into.

---

## 3. Bonding-curve phase (about 16%)

**What it means:** the curve is the audition before migration. How long did it take to fill, how broad was the
crowd, how much demand failed to get in, and how concentrated was the buying.
Executed trades only (`failed == 0`) unless the name says "fail".

| feature | meaning | gain% | top30 | Q1 | Q5 | rho |
|---|---|---|---|---|---|---|
| `c_fail_rate` | failed / all curve transactions (crowding, slippage, bot collisions) | 1.23 | 3 | 3.0 | 5.9 | +0.014 |
| `c_router` | share of curve trades sent through a router or bot program | 1.16 | 3 | 2.4 | 5.7 | +0.020 |
| `c_fail_uw` | distinct wallets with a failed transaction | 1.07 | 2 | 3.0 | 5.1 | |
| `c_buy_max` | largest curve buy (SOL) | 1.06 | 1 | 6.1 | 4.0 | -0.044 |
| `c_gini` | inequality of buy SOL across wallets | 1.06 | 1 | 2.7 | 5.8 | |
| `c_n_total` | executed curve trades + AMM trades | 1.04 | 2 | 4.2 | 5.3 | |
| `c_funded` | share of curve trades with fee payer != signer | 1.03 | 2 | 2.1 | 4.6 | |
| `c_creator_sold` | creator's SOL sold on the curve | 0.88 | 1 | 2.9 | 6.4 | |
| `c_top1`, `c_top5` | top 1 / top 5 buyers' share of curve buy SOL | 0.8 | 0 | 5.0 | 4.0 | |
| `c_fail_last60` | failed transactions in the last 60s before migration | 0.69 | 0 | 3.2 | 6.5 | +0.046 |
| `c_roundtrip` | share of buyers who also sold on the curve (flippers) | 0.68 | 0 | 1.3 | 5.3 | |
| `c_dur` | seconds from first curve trade to migration | 0.65 | 0 | **1.1** | 4.8 | |
| `c_failed`, `c_n`, `c_uw`, `c_buyers` | raw counts | 0.3-0.8 | 0 | | | |
| `c_buy_sol`, `c_sell_sol`, `c_sell_share` | curve volume and the sold fraction | 0.1-0.5 | 0 | | | |
| `c_late10m_share` | share of curve buy SOL in the last 10 minutes | 0.50 | 0 | 4.6 | 4.9 | |

**Takeaways:** a **curve filled instantly** (`c_dur` Q1 1.1%) or by **very few wallets** is a trap. A broad crowd with
selling, flipping and failed attempts is a *healthy* market. Low gain here partly reflects overlap with the AMM and holder features.

---

## 4. Holders at migration (about 5%, overlaps curve concentration)

**What it means:** your "wallet distribution" idea. We replay the curve tape and add up each wallet's net tokens
(bought minus sold). Whatever is left at migration is who holds the supply. The 793.1M tokens sold on the curve are the denominator.

| feature | meaning | gain% | top30 | Q1 | Q5 | rho |
|---|---|---|---|---|---|---|
| `h_gini` | inequality of holdings | 1.04 | 1 | 2.8 | 6.3 | +0.054 |
| `h_top1` | largest holder's share of curve supply | 0.96 | 1 | 5.4 | 4.0 | -0.036 |
| `h_n` | number of wallets holding at migration | 0.83 | 0 | 2.8 | 4.9 | |
| `h_creator` | creator's remaining share | 0.73 | 0 | 6.3 | 4.0 | -0.061 |
| `h_top10` | top 10 holders' share | 0.64 | 0 | 5.3 | 3.9 | |

**Takeaway:** in the earlier 30-day check, the most concentrated fifth (`h_top1`, `h_creator`) reached 5x **0.1%** of the
time. These are very strong *exclusion* signals, but the model mostly reaches them through `c_top1`/`a_top1`.
Holders are measured from the curve only; transfers outside the curve and the first-minute AMM trades aren't included yet.

---

## 5. Launch record (about 9%)

**What it means:** how the coin was born, from `pumpfun_token_creation.csv`.

| feature | meaning | gain% | top30 | Q1 | Q5 | rho |
|---|---|---|---|---|---|---|
| `dev_sol` | log of the creator wallet's SOL balance before creating | 1.84 | 6 | 5.8 | 3.7 | -0.047 |
| `gas_used` | compute units of the creation transaction (fingerprints the launch tool or bot) | 1.72 | 6 | 5.9 | 5.2 | |
| `hour` | UTC hour of migration | 1.30 | 5 | | | |
| `insider_pct` | share of curve supply bought inside the creation bundle | 1.10 | 1 | 6.3 | 4.0 | -0.036 |
| `name_len`, `sym_len`, `sym_upper`, `name_non_ascii` | name / ticker shape | 0-1.0 | 0 | | | |
| `n_lookup_r`, `n_instr` | lookup-table reads and instruction count of the creation tx | 0.5-0.6 | 0 | | | |
| `age` | seconds from creation to migration | 0.42 | 0 | **1.1** | 4.8 | |
| `bs_tips`, `bs_beq`, `bs_other`, `bs_unknown` | counts of tip / buy-exact-quote / other / unknown parts in `bundle_structure` | 0.1-0.4 | 0 | | | |
| `bundle_size`, `bundled_buys_count` | transactions in the creation bundle, buys inside it | 0.2-0.3 | 0 | | | |
| `full_curve_buy` | the bundle bought the whole curve (self-graduation) | 0.01 | 0 | | | -0.054 |
| `legacy_token`, `cashback`, `direct_pf`, `host_ipfs` | flags | ~0 | 0 | | | |

**Takeaways:** **self-graduated launches** (`insider_pct` near 1, `age` near 0) are dead on arrival: about 40% of all
tokens, with 1-4% reaching 5x. Flags like `full_curve_buy` get ~0 gain because `insider_pct` and `age` already cover them.
`dev_sol` and `gas_used` probably fingerprint *which tool or team* launched the coin. That's worth decoding.

---

## Tested and not adopted

**v2 tape features** (lab03, paired -15.8, P=0.26):
- router program shares (`*_prog_FLASHX/6Vo324/DF1ow4/term9Y/JUP6/GMgn/...`)
- `a_burner_share` (buyers left with < 0.01 SOL), `a_buyer_bal_med/sum`
- `a_snipe_sol/n` (buys in the first 2s), `a_multisig`, `a_exact_quote`
- `c_tip_share`, `c_multisig`, `c_v3_share`, `c_last60_buy_sol`, `c_last5m_buyers`, `c_fail_buy_last5m`

They're still computed in the cache (VERSION 2). Some may help in a pruned model. Not adopting them only means "not better on its own".

**Reputation v2, written but not yet tested** (`reputation.py` + `exp_lab06.py`): new wallet edges for **AMM sellers in
the first minute (role 2)** and **curve buyers in the last 5 minutes (role 3)**. New features: `ws_*` (track record of
first-minute sellers), `wl_*` (late curve buyers), `elite_net` (elite buyers minus elite sellers) and `elite_sell_share`.
The idea: smart money *leaving* during the debut should be a warning. Run `Krishna/src/exp_lab06.py` to test it.

---

## Adding a new feature

1. **Per-token tape or launch feature** (uses only the token's own data up to mt+60s):
   - add it to `curve_token()`, `amm_token()` or `launch_features()` in [features.py](src/features.py)
   - bump `VERSION`
   - rebuild with `harness/.venv/bin/python Krishna/src/features.py` (about 4 min, under `caffeinate -i`)
2. **Cross-token feature** (anything using other tokens' outcomes or behaviour): add it to [reputation.py](src/reputation.py), inside the
   day loop. It may only read `lab.iloc[:upto]` (tokens resolved before the cutoff). Test: the value must not change if
   you delete every token that migrated later.
3. **Test it:** copy an `exp_lab0X.py` and compare against `oof_base5` (all features) or `oof_v1_top30` with `paired()`, using 5 seeds.
   Log the result in RESULTS.md. Expect noise of about +-50. Also look at the 14-day worst, exBest and days+, not just the total.

## Raw columns not used yet (room to expand)

- **Curve tape:** `min_amount_out` / `max_amount_in` (slippage tolerance: panic buyers vs patient bots), `provided_gas_fee`,
  `consumed_gas`, `tip_index` (Jito tips = sniper bots), `signers`, `holder_rewards`, `instruction_type` by wallet,
  per-wallet timing (buy then sell within N seconds), the reverted rows' wallets and directions in more depth.
- **AMM tape:** `user_base_token_reserves` (tokens the buyer holds after the trade: are they accumulating?),
  `fees_retained_in_pool`, `coin_creator_fees`, `top_level_transfers_json` (SOL funding flows inside the trade), `lookup_tables`.
- **Migrations file:** `user` (who triggered the migration), `pool_migration_fee`.
- **Creation:** `bundle_structure` amounts per transaction (who got how much at launch), `url` host as a category (`host` is
  in the data but not used by the model yet), `version`.
- **Wallet graph:** wallets funded by the same fee payer, wallets that always buy together, and creator-linked wallets
  (the creator's fee payer showing up in the AMM buyers is a classic trap tell).

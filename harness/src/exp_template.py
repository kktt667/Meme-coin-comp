"""Paired walk-forward test: hour-of-day filter vs skip-everything.
Copy this file per experiment (exp_<idea>.py) and change ONE thing. Runs from any directory."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *

TP = 5.0     # legal tp is 0 (skip) or 3..100

def skip_all(train, ev):
    """The honest baseline: P&L 0 every day. Anything that can't beat this is losing money."""
    return np.zeros(len(ev))

def buy_all(train, ev):
    """Sanity check: buying every token loses on every day."""
    return np.full(len(ev), TP)

def hour_filter(train, ev):
    """Toy learned strategy: only trade launch hours whose past P&L at TP was positive.
    Uses train labels only - the pattern every real model should follow."""
    hr = lambda d: pd.to_datetime(d['mt'], unit='s').dt.hour.values
    p = pnl(np.full(len(train), TP), train['max_x'].values, train['final_x'].values)
    good = pd.Series(p).groupby(hr(train)).mean()
    good = set(good[good > 0].index)
    return np.where(np.isin(hr(ev), list(good)), TP, 0.0)

tr, te = load()
check_window(tr, te)      # our walk-forward window rebuilds train.csv exactly
splits = get_days(tr)
print(f"{len(tr)} labelled tokens, {tr['day'].nunique()} days; scoring last {len(splits)} days")

base, _ = run(skip_all, tr, splits);    report("SKIP ALL", base)
alln, _ = run(buy_all, tr, splits);     report(f"BUY ALL @ {TP}", alln)
var, _  = run(hour_filter, tr, splits); report(f"HOUR FILTER @ {TP}", var)
paired(var, base, 'hour_filter', 'skip_all')

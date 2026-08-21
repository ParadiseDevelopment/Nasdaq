#!/usr/bin/env python3
"""
Bar-replay backtest for NAS100_Overnight.mq5.

The strategy: go long the index at a fixed server time in the evening, flat at
a fixed server time before the cash open. Nothing is predicted. It harvests the
overnight drift in equity indices, which on the measured feed accounted for
essentially the whole year's move.

Unlike the ML bot's backtest this one is nearly assumption-free — the entry and
exit times are fixed, so the only modelled quantities are the spread, the
overnight financing charge and an intrabar protective stop.

    python3 tools/backtest_overnight.py --bars NAS100.s_M15.csv --swap-points 4

IMPORTANT: --swap-points is the single most important input and it is
broker-specific. The position is deliberately held across the 00:00 rollover,
so you pay financing every night. Read it from your terminal
(right-click the symbol -> Specification -> Swap long) and convert to index
points. Getting it wrong invalidates the result.

Requires numpy and pandas.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import atr, load_bars  # noqa: E402

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def run(df, cfg, spec):
    """One long trade per night. Entry and exit are pure clock events."""
    df = df.copy()
    df["atr"] = atr(df, 14)
    point = spec["point"]
    vpu = spec["tick_value"] / spec["tick_size"]

    bars = df.index
    entry_mask = (bars.hour == cfg.entry_hour) & (bars.minute == cfg.entry_minute)
    exit_mask = (bars.hour == cfg.exit_hour) & (bars.minute == cfg.exit_minute)
    entry_times = bars[entry_mask]

    skip = {WEEKDAYS.index(d) for d in cfg.skip_days} if cfg.skip_days else set()

    balance = cfg.deposit
    peak = cfg.deposit
    halted = False
    trades = []

    for t0 in entry_times:
        if halted:
            break
        if t0.dayofweek in skip:
            continue

        i0 = bars.get_loc(t0)
        atr_v = df["atr"].iloc[i0]
        if not np.isfinite(atr_v) or atr_v <= 0:
            continue

        spread_pts = df["spread"].iloc[i0]
        if spread_pts > cfg.max_spread_points:
            continue

        # find this night's exit: the first exit-time bar strictly after entry
        later = bars[(bars > t0) & exit_mask]
        if len(later) == 0:
            continue
        t1 = later[0]
        if (t1 - t0) > pd.Timedelta(hours=cfg.max_hold_hours):
            continue                      # weekend or data gap - skip the night
        i1 = bars.get_loc(t1)

        entry = df["open"].iloc[i0]
        stop_dist = cfg.stop_atr * atr_v if cfg.stop_atr > 0 else 0.0
        sl = entry - stop_dist if stop_dist > 0 else None

        # position size from the money at risk; with no stop, size off a
        # nominal distance so risk per trade stays comparable
        risk_dist = stop_dist if stop_dist > 0 else cfg.nominal_risk_atr * atr_v
        risk_money = balance * cfg.risk_percent / 100.0
        raw = risk_money / (risk_dist * vpu)
        step = spec["volume_step"]
        lots = min(np.floor(raw / step) * step, spec["volume_max"], cfg.max_lots)
        if lots < spec["volume_min"]:
            continue

        # walk the night bar by bar so the stop is honoured intrabar
        exit_price, reason = df["open"].iloc[i1], "time"
        if sl is not None:
            seg = df.iloc[i0 + 1: i1 + 1]
            hit = seg.index[seg["low"] <= sl]
            if len(hit):
                exit_price, reason = sl, "stop"

        gross = (exit_price - entry) * lots * vpu
        # spread arrives in broker points, swap is quoted in index points
        cost = (spread_pts * point + cfg.swap_points) * lots * vpu
        cost += cfg.commission * lots
        net = gross - cost

        balance += net
        peak = max(peak, balance)
        if (peak - balance) / peak * 100.0 >= cfg.max_drawdown_pct:
            halted = True

        trades.append(dict(entry_time=t0, exit_time=t1, entry=entry, exit=exit_price,
                           lots=lots, points=exit_price - entry, profit=net,
                           reason=reason, dow=WEEKDAYS[t0.dayofweek], balance=balance))

    return pd.DataFrame(trades), balance, halted


def report(tr, final, halted, cfg, df):
    print("=" * 70)
    print("NAS100 Overnight - bar replay")
    print("=" * 70)
    print(f"period        : {df.index[0]:%Y-%m-%d}  ->  {df.index[-1]:%Y-%m-%d}")
    print(f"session       : long {cfg.entry_hour:02d}:{cfg.entry_minute:02d} "
          f"-> flat {cfg.exit_hour:02d}:{cfg.exit_minute:02d} (server time)")
    print(f"costs         : spread from data + {cfg.swap_points} pts swap "
          f"+ {cfg.commission:.2f}/lot commission")
    print(f"stop          : {cfg.stop_atr} x ATR" if cfg.stop_atr > 0 else "stop          : none")
    print()

    if tr.empty:
        print("NO TRADES. Check --entry-hour against the data's server time.")
        return

    ret = (final / cfg.deposit - 1.0) * 100.0
    eqc = pd.Series(tr.balance.values, index=tr.exit_time)
    dd = (eqc.cummax() - eqc) / eqc.cummax() * 100.0
    wins = tr[tr.profit > 0]

    print(f"RETURN        : {ret:+.2f}%   ({cfg.deposit:,.0f} -> {final:,.0f})")
    print(f"max drawdown  : {dd.max():.2f}%")
    print(f"nights traded : {len(tr)}")
    print(f"win rate      : {len(wins)/len(tr)*100:.1f}%")
    pf = wins.profit.sum() / max(-tr[tr.profit <= 0].profit.sum(), 1e-9)
    print(f"profit factor : {pf:.2f}")
    print(f"avg night     : {tr.profit.mean():+,.2f}  ({tr.points.mean():+.1f} pts gross)")
    print(f"best / worst  : {tr.profit.max():+,.2f} / {tr.profit.min():+,.2f}")
    if halted:
        print("\n!! drawdown kill switch fired - trading stopped early")

    print("\nby weekday (entry night):")
    for d, g in tr.groupby("dow", sort=False):
        print(f"  {d}  n={len(g):>3}  mean {g.profit.mean():+9.2f}  total {g.profit.sum():+10.2f}")

    print("\nby month:")
    m = tr.set_index("exit_time").profit.resample("1ME").sum()
    for k, v in m.items():
        bar = "#" * min(30, int(abs(v) / max(m.abs().max(), 1e-9) * 30))
        print(f"  {k:%Y-%m}  {v:+10.2f}  {bar}")

    half = len(tr) // 2
    print(f"\nfirst half    : {tr.profit[:half].mean():+.2f} per night")
    print(f"second half   : {tr.profit[half:].mean():+.2f} per night")
    print("  (both should be positive; if only one is, do not trade this)")

    bh = (df.close.iloc[-1] / df.close.iloc[0] - 1.0) * 100.0
    print(f"\nbuy and hold  : {bh:+.2f}%  (24h exposure vs this bot's "
          f"~{(cfg.exit_hour - cfg.entry_hour) % 24}h/day)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bars", required=True)
    p.add_argument("--from", dest="dt_from", default=None)
    p.add_argument("--to", dest="dt_to", default=None)
    p.add_argument("--deposit", type=float, default=10000.0)
    p.add_argument("--entry-hour", type=int, default=23)
    p.add_argument("--entry-minute", type=int, default=0)
    p.add_argument("--exit-hour", type=int, default=4)
    p.add_argument("--exit-minute", type=int, default=0)
    p.add_argument("--max-hold-hours", type=int, default=12)
    p.add_argument("--swap-points", type=float, default=4.0,
                   help="overnight financing in index points - BROKER SPECIFIC")
    p.add_argument("--commission", type=float, default=0.0)
    p.add_argument("--risk-percent", type=float, default=0.5)
    p.add_argument("--stop-atr", type=float, default=3.0, help="0 disables the stop")
    p.add_argument("--nominal-risk-atr", type=float, default=3.0)
    p.add_argument("--max-lots", type=float, default=5.0)
    p.add_argument("--max-spread-points", type=float, default=400.0)
    p.add_argument("--max-drawdown-pct", type=float, default=25.0)
    p.add_argument("--skip-days", nargs="*", default=[],
                   help="e.g. --skip-days Mon   (UNVALIDATED, see docs)")
    p.add_argument("--point", type=float, default=None)
    p.add_argument("--tick-value", type=float, default=None)
    p.add_argument("--tick-size", type=float, default=None)
    cfg = p.parse_args()

    df, spec = load_bars(cfg.bars, {"point": cfg.point, "tick_value": cfg.tick_value,
                                    "tick_size": cfg.tick_size})
    if cfg.dt_from:
        df = df[df.index >= pd.Timestamp(cfg.dt_from)]
    if cfg.dt_to:
        df = df[df.index < pd.Timestamp(cfg.dt_to)]

    tr, final, halted = run(df, cfg, spec)
    report(tr, final, halted, cfg, df)


if __name__ == "__main__":
    main()

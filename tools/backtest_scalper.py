#!/usr/bin/env python3
"""
Bar-replay backtest for XAUUSD_Scalper.mq5 - the gold scalper that moves to the
24/7 book at weekends.

This is a FAITHFUL REPLICA of the EA's decision path: the same indicators, the
same gate order, the same ATR-anchored stops, the same weekend handover. It
replays OHLC bars rather than ticks, so it is the fast first answer, not the
final one - MetaTrader's tester on real ticks is.

It has no dependencies. Standard library only, so it runs wherever python3 does.

    # weekdays only
    python3 tools/backtest_scalper.py --bars XAUUSD.s_M5.csv

    # with the weekend book, which is the whole point of this bot
    python3 tools/backtest_scalper.py --bars XAUUSD.s_M5.csv \
                                      --weekend-bars "XAUUSD24-7.s_M5.csv"

    # no data yet? this generates a random walk and shows what NO EDGE looks like
    python3 tools/backtest_scalper.py --demo

THE LINE THAT MATTERS is the expectancy, in R, with its standard error:

    expectancy   : +0.031 R  (SE 0.038)  t = 0.81   <- indistinguishable from 0

Unlike the ML bot in this repository, these trades do not overlap - the EA holds
one position at a time - so that t statistic is honest. Below about t = 2, with
both halves positive and the random-direction control beaten, there is nothing
here worth trading. A backtest return without that check is decoration.

Input CSV: the output of MQL5/Scripts/ExportBars.mq5, or MetaTrader's own
tab-separated chart export. Export BOTH symbols on the same timeframe.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
from datetime import datetime, timedelta

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MODE_BREAKOUT, MODE_FADE = "breakout", "fade"


# ======================================================================
# Bar loading
# ======================================================================
def default_spec():
    return {"point": 0.01, "tick_value": 0.01, "tick_size": 0.01,
            "volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0,
            "symbol": "?", "timeframe": "?"}


def _parse_time(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"unrecognised timestamp: {s!r}")


def load_bars(path, overrides=None):
    """Read ExportBars.mq5 output or MetaTrader's native chart export."""
    spec = default_spec()
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]

    if not lines:
        raise SystemExit(f"{path}: empty file")

    start = 0
    if lines[0].startswith("#"):
        for tok in lines[0][1:].split():
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            if k in ("symbol", "timeframe"):
                spec[k] = v
            elif k in spec:
                try:
                    spec[k] = float(v)
                except ValueError:
                    pass
        start = 1

    header = lines[start]
    native = header.lstrip().startswith("<DATE>")
    delim = "\t" if native else ","
    cols = [c.strip().strip("<>").lower() for c in header.split(delim)]
    rows = list(csv.reader(lines[start + 1:], delimiter=delim))

    if native and spec["symbol"] == "?":
        stem = os.path.basename(path)
        spec["symbol"] = stem.split("_")[0].split("-")[-1]

    idx = {name: cols.index(name) for name in cols}
    series = {k: [] for k in ("time", "open", "high", "low", "close", "spread")}
    for row in rows:
        if len(row) < len(cols):
            continue
        try:
            if native:
                t = _parse_time(row[idx["date"]] + " " + row[idx["time"]])
            else:
                t = _parse_time(row[idx["time"]])
            o = float(row[idx["open"]])
            h = float(row[idx["high"]])
            lo = float(row[idx["low"]])
            c = float(row[idx["close"]])
        except (ValueError, KeyError, IndexError):
            continue
        sp = 0.0
        if "spread" in idx:
            try:
                sp = float(row[idx["spread"]])
            except ValueError:
                sp = 0.0
        series["time"].append(t)
        series["open"].append(o)
        series["high"].append(h)
        series["low"].append(lo)
        series["close"].append(c)
        series["spread"].append(sp)

    order = sorted(range(len(series["time"])), key=lambda i: series["time"][i])
    for k in series:
        series[k] = [series[k][i] for i in order]

    if overrides:
        spec.update({k: v for k, v in overrides.items() if v is not None})
    if not series["time"]:
        raise SystemExit(f"{path}: no usable bars")
    return series, spec


# ======================================================================
# Indicators - matched to the MetaTrader 5 built-ins the EA calls
# ======================================================================
def rma(values, n, seed_sma=True):
    """Wilder smoothing, seeded with an SMA like MT5's own indicators."""
    out = [None] * len(values)
    if len(values) < n:
        return out
    acc = 0.0
    count = 0
    prev = None
    for i, v in enumerate(values):
        if v is None:
            continue
        if prev is None:
            acc += v
            count += 1
            if count == n:
                prev = acc / n if seed_sma else v
                out[i] = prev
            continue
        prev = prev + (v - prev) / n
        out[i] = prev
    return out


def ema(values, n):
    out = [None] * len(values)
    k = 2.0 / (n + 1.0)
    prev = None
    acc, count = 0.0, 0
    for i, v in enumerate(values):
        if prev is None:
            acc += v
            count += 1
            if count == n:
                prev = acc / n
                out[i] = prev
            continue
        prev = v * k + prev * (1.0 - k)
        out[i] = prev
    return out


def sma(values, n):
    out = [None] * len(values)
    run = 0.0
    for i, v in enumerate(values):
        run += v
        if i >= n:
            run -= values[i - n]
        if i >= n - 1:
            out[i] = run / n
    return out


def stdev_pop(values, n):
    out = [None] * len(values)
    for i in range(n - 1, len(values)):
        w = values[i - n + 1: i + 1]
        m = sum(w) / n
        out[i] = math.sqrt(sum((x - m) ** 2 for x in w) / n)
    return out


def true_range(h, l, c):
    out = [h[0] - l[0]]
    for i in range(1, len(h)):
        pc = c[i - 1]
        out.append(max(h[i] - l[i], abs(h[i] - pc), abs(l[i] - pc)))
    return out


def atr(h, l, c, n):
    return rma(true_range(h, l, c), n)


def rsi(close, n):
    gains, losses = [0.0], [0.0]
    for i in range(1, len(close)):
        d = close[i] - close[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag, al = rma(gains, n), rma(losses, n)
    out = [None] * len(close)
    for i in range(len(close)):
        if ag[i] is None or al[i] is None:
            continue
        out[i] = 100.0 if al[i] == 0 else 100.0 - 100.0 / (1.0 + ag[i] / al[i])
    return out


def adx(h, l, c, n):
    plus_dm, minus_dm = [0.0], [0.0]
    for i in range(1, len(h)):
        up = h[i] - h[i - 1]
        dn = l[i - 1] - l[i]
        plus_dm.append(up if (up > dn and up > 0) else 0.0)
        minus_dm.append(dn if (dn > up and dn > 0) else 0.0)
    tr_n = rma(true_range(h, l, c), n)
    p_n, m_n = rma(plus_dm, n), rma(minus_dm, n)
    dx = [None] * len(h)
    for i in range(len(h)):
        if tr_n[i] in (None, 0) or p_n[i] is None or m_n[i] is None:
            continue
        pdi = 100.0 * p_n[i] / tr_n[i]
        mdi = 100.0 * m_n[i] / tr_n[i]
        s = pdi + mdi
        dx[i] = 0.0 if s == 0 else 100.0 * abs(pdi - mdi) / s
    return rma(dx, n)


def rolling_extreme(values, n, want_max):
    """Extreme of the n bars ENDING at i, aligned so index i covers [i-n+1, i]."""
    out = [None] * len(values)
    for i in range(n - 1, len(values)):
        w = values[i - n + 1: i + 1]
        out[i] = max(w) if want_max else min(w)
    return out


def htf_context(times, closes, htf_minutes, ema_period):
    """For every base bar, the EMA and close of the last CLOSED higher-TF bar.

    Causal by construction: a higher-timeframe bucket only becomes visible once
    a base bar belonging to a later bucket has opened.
    """
    span = htf_minutes * 60
    epoch = datetime(1970, 1, 1)
    n = len(times)
    ema_at = [None] * n
    close_at = [None] * n

    k = 2.0 / (ema_period + 1.0)
    cur_bucket = None
    cur_close = None
    seen = 0
    acc = 0.0
    ema_val = None
    last_closed_ema = None
    last_closed_close = None

    for i in range(n):
        b = int((times[i] - epoch).total_seconds()) // span
        if cur_bucket is None:
            cur_bucket = b
        elif b != cur_bucket:
            # the previous bucket is complete
            seen += 1
            if ema_val is None:
                acc += cur_close
                if seen == ema_period:
                    ema_val = acc / ema_period
            else:
                ema_val = cur_close * k + ema_val * (1.0 - k)
            last_closed_ema = ema_val
            last_closed_close = cur_close
            cur_bucket = b
        cur_close = closes[i]
        ema_at[i] = last_closed_ema
        close_at[i] = last_closed_close
    return ema_at, close_at


def build_indicators(series, cfg):
    h, l, c = series["high"], series["low"], series["close"]
    ind = {}
    ind["atr"] = atr(h, l, c, cfg.atr_period)
    ind["rsi"] = rsi(c, cfg.rsi_period)
    ind["adx"] = adx(h, l, c, cfg.adx_period)
    ind["fast"] = ema(c, cfg.ema_fast)
    ind["slow"] = ema(c, cfg.ema_slow)
    base = sma(c, cfg.bb_period)
    sd = stdev_pop(c, cfg.bb_period)
    ind["bb_up"] = [None if base[i] is None else base[i] + cfg.bb_dev * sd[i]
                    for i in range(len(c))]
    ind["bb_lo"] = [None if base[i] is None else base[i] - cfg.bb_dev * sd[i]
                    for i in range(len(c))]
    # the channel the EA reads spans the bars BEFORE the signal bar, so at
    # signal bar j it is the extreme of [j-donchian, j-1]
    ind["dc_hi"] = rolling_extreme(h, cfg.donchian, True)
    ind["dc_lo"] = rolling_extreme(l, cfg.donchian, False)
    if cfg.use_htf:
        ind["htf_ema"], ind["htf_close"] = htf_context(
            series["time"], c, cfg.htf_minutes, cfg.htf_ema)
    else:
        ind["htf_ema"] = [None] * len(c)
        ind["htf_close"] = [None] * len(c)
    return ind


# ======================================================================
# The signal - index for index with SignalEngine.mqh
# ======================================================================
def evaluate(series, ind, j, cfg):
    """Signal from closed bar j. Returns (dir, mode, stop_dist, atr) or None."""
    a = ind["atr"][j]
    if a is None or a <= 0:
        return None
    for key in ("rsi", "adx", "fast", "slow", "bb_up", "bb_lo"):
        if ind[key][j] is None:
            return None
    if j < cfg.donchian + 1:
        return None
    hh, ll = ind["dc_hi"][j - 1], ind["dc_lo"][j - 1]
    if hh is None or ll is None:
        return None

    o1 = series["open"][j]
    h1 = series["high"][j]
    l1 = series["low"][j]
    c1 = series["close"][j]
    fast, slow = ind["fast"][j], ind["slow"][j]
    adx_v, rsi_v = ind["adx"][j], ind["rsi"][j]

    htf_bias = 0
    if cfg.use_htf:
        he, hc = ind["htf_ema"][j], ind["htf_close"][j]
        if he is None or hc is None:
            return None
        htf_bias = 1 if hc > he else -1

    rng = h1 - l1
    body = abs(c1 - o1)

    def stop_from(direction, by_atr_mult):
        swing = (l1 - 0.10 * a) if direction > 0 else (h1 + 0.10 * a)
        by_atr = (c1 - by_atr_mult * a) if direction > 0 else (c1 + by_atr_mult * a)
        stop = min(swing, by_atr) if direction > 0 else max(swing, by_atr)
        return min(abs(c1 - stop), cfg.max_stop_atr * a)

    if cfg.mode in ("breakout", "both"):
        d = 0
        if c1 > hh and c1 > o1 and fast > slow and c1 > fast:
            d = 1
        elif c1 < ll and c1 < o1 and fast < slow and c1 < fast:
            d = -1
        if d != 0:
            ok = (rng >= cfg.expansion_atr * a and rng > 0
                  and body >= cfg.body_frac * rng
                  and adx_v >= cfg.min_adx
                  and (not cfg.use_htf or htf_bias == d)
                  and abs(c1 - fast) <= cfg.max_stretch_atr * a)
            if ok:
                return d, MODE_BREAKOUT, stop_from(d, cfg.bo_stop_atr), a, cfg.bo_target_r

    if cfg.mode in ("fade", "both"):
        d = 0
        if c1 < ind["bb_lo"][j] and rsi_v <= cfg.fade_rsi:
            d = 1
        elif c1 > ind["bb_up"][j] and rsi_v >= (100.0 - cfg.fade_rsi):
            d = -1
        if d != 0:
            stretch = (fast - c1) if d > 0 else (c1 - fast)
            ok = (adx_v <= cfg.fade_max_adx
                  and stretch >= cfg.fade_stretch_atr * a
                  and (not (cfg.use_htf and cfg.fade_with_htf) or htf_bias == d))
            if ok:
                return d, MODE_FADE, stop_from(d, cfg.fade_stop_atr), a, cfg.fade_target_r
    return None


# ======================================================================
# Weekend routing - the CLOCK model, matching XS_ROUTE_CLOCK in the EA
# ======================================================================
class Calendar:
    """Spot market open from week_start (dow, sec) to week_end (dow, sec)."""

    def __init__(self, start_dow, start_sec, end_dow, end_sec):
        self.f = start_dow * 86400 + start_sec
        self.t = end_dow * 86400 + end_sec
        if self.t <= self.f:
            self.t += 604800

    @staticmethod
    def week_seconds(t: datetime) -> int:
        # datetime.weekday(): Monday = 0. The EA uses Sunday = 0.
        dow = (t.weekday() + 1) % 7
        return dow * 86400 + t.hour * 3600 + t.minute * 60 + t.second

    def spot_open(self, t: datetime) -> bool:
        ws = self.week_seconds(t)
        return self.f <= ws < self.t or self.f <= ws + 604800 < self.t

    def seconds_to_handover(self, t: datetime, weekend: bool) -> int:
        ws = self.week_seconds(t)
        edge = self.t if not weekend else self.f
        d = edge - ws
        while d < 0:
            d += 604800
        return d


# ======================================================================
# The replay
# ======================================================================
class Trade:
    """One completed round turn."""

    __slots__ = ("symbol", "book", "mode", "dir", "entry_time", "entry", "exit_time",
                 "exit", "lots", "stop_dist", "risk_money", "net", "r", "reason", "bars",
                 "sl", "tp", "atr", "cost", "book_index")


def _in_session(hour, start, end):
    if start == end:
        return True
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def run(books, cfg, cal, rng=None, random_direction=False, zero_spread=False):
    """Replay every book on one merged timeline. One position at a time.

    The EA holds a single position across both symbols, so the replay does too -
    which is also what makes the per-trade statistics independent enough to test.
    """
    events = []
    for b_i, b in enumerate(books):
        for i in range(len(b["series"]["time"])):
            events.append((b["series"]["time"][i], b_i, i))
    events.sort(key=lambda e: (e[0], e[1]))

    balance = cfg.deposit
    peak = balance
    trades = []
    pos = None

    day_key = None
    day_start_balance = balance
    trades_today = 0
    halted_day = False
    halted = False
    loss_streak = 0
    cooldown = 0

    for t, b_i, i in events:
        b = books[b_i]
        s, ind, spec = b["series"], b["ind"], b["spec"]
        weekend_book = (b["book"] == "weekend")
        vpu = spec["tick_value"] / spec["tick_size"]
        point = spec["point"]

        dk = (t.year, t.month, t.day)
        if dk != day_key:
            day_key = dk
            day_start_balance = balance
            trades_today = 0
            halted_day = False

        #--- which book is live right now, and how close the handover is
        spot = cal.spot_open(t)
        live = (spot != weekend_book)
        handover_soon = cal.seconds_to_handover(t, weekend_book) <= cfg.flatten_minutes * 60

        o, h, l, c = s["open"][i], s["high"][i], s["low"][i], s["close"][i]

        # ------------------------------------------------------------------
        # 1. entry at this bar's open, from the signal on the bar before it
        # ------------------------------------------------------------------
        if (pos is None and live and not handover_soon and not halted and not halted_day
                and cooldown <= 0 and i >= 1
                and (cfg.max_trades_per_day <= 0 or trades_today < cfg.max_trades_per_day)
                and _in_session(t.hour,
                                cfg.we_session_start if weekend_book else cfg.session_start,
                                cfg.we_session_end if weekend_book else cfg.session_end)):
            sig = evaluate(s, ind, i - 1, cfg)
            if sig is not None:
                d, mode, stop_dist, a, target_r = sig
                spread_pts = 0.0 if zero_spread else s["spread"][i]
                spread_price = spread_pts * point
                cap_pts = cfg.we_max_spread_points if weekend_book else cfg.max_spread_points
                cap_frac = cfg.we_max_spread_frac if weekend_book else cfg.max_spread_frac
                gate_abs = cap_pts <= 0 or spread_pts <= cap_pts
                gate_rel = cap_frac <= 0 or spread_price <= cap_frac * stop_dist
                if stop_dist > 0 and gate_abs and gate_rel:
                    risk_pct = cfg.risk_percent * (cfg.weekend_risk_mult if weekend_book else 1.0)
                    step = spec["volume_step"]
                    raw = (balance * risk_pct / 100.0) / (stop_dist * vpu)
                    lots = min(math.floor(raw / step) * step, spec["volume_max"], cfg.max_lots)
                    if lots >= spec["volume_min"]:
                        if random_direction:
                            d = 1 if rng.random() < 0.5 else -1
                        #--- a long is filled at the ask, a short at the bid, and
                        #--- both are closed on the other side of the book. Charging
                        #--- the spread once at entry and reading exits off the raw
                        #--- series is therefore exact, not an approximation.
                        pos = Trade()
                        pos.symbol = spec["symbol"]
                        pos.book = b["book"]
                        pos.book_index = b_i
                        pos.mode = mode
                        pos.dir = d
                        pos.entry_time = t
                        pos.entry = o + spread_price if d > 0 else o
                        pos.lots = lots
                        pos.stop_dist = stop_dist
                        pos.risk_money = stop_dist * lots * vpu
                        pos.atr = a
                        pos.bars = 0
                        pos.sl = pos.entry - stop_dist * d
                        pos.tp = pos.entry + target_r * stop_dist * d
                        pos.cost = spread_price * lots * vpu + cfg.commission * lots
                        trades_today += 1

        # ------------------------------------------------------------------
        # 2. manage the open position, on its own book's bars
        # ------------------------------------------------------------------
        if pos is not None and pos.book_index == b_i:
            d = pos.dir
            pos.bars += 1
            exit_price, reason = None, None

            #--- stop first: when both levels sit inside one bar we cannot know
            #--- the order, so assume the worse one
            if d > 0:
                if l <= pos.sl:
                    exit_price, reason = pos.sl, "stop"
                elif h >= pos.tp:
                    exit_price, reason = pos.tp, "target"
            else:
                if h >= pos.sl:
                    exit_price, reason = pos.sl, "stop"
                elif l <= pos.tp:
                    exit_price, reason = pos.tp, "target"

            if exit_price is None:
                #--- break even and the ATR trail, moved on CLOSES only. The EA
                #--- moves them tick by tick; confirming on the close is the
                #--- pessimistic reading and keeps the replay from inventing
                #--- exits that the bar's shape cannot justify.
                r_mult = ((c - pos.entry) * d / pos.stop_dist) if pos.stop_dist > 0 else 0.0
                new_sl = pos.sl
                if cfg.be_trigger_r > 0 and r_mult >= cfg.be_trigger_r:
                    be = pos.entry + cfg.be_offset_r * pos.stop_dist * d
                    new_sl = max(new_sl, be) if d > 0 else min(new_sl, be)
                if cfg.trail_start_r > 0 and r_mult >= cfg.trail_start_r:
                    tr = c - cfg.trail_atr * pos.atr * d
                    new_sl = max(new_sl, tr) if d > 0 else min(new_sl, tr)
                pos.sl = new_sl

                if cfg.max_bars_in_trade > 0 and pos.bars >= cfg.max_bars_in_trade:
                    exit_price, reason = c, "time"
                elif handover_soon:
                    exit_price, reason = c, "handover"

            if exit_price is not None:
                gross = (exit_price - pos.entry) * d * pos.lots * vpu
                pos.net = gross - pos.cost
                pos.exit = exit_price
                pos.exit_time = t
                pos.reason = reason
                pos.r = pos.net / pos.risk_money if pos.risk_money > 0 else 0.0

                balance += pos.net
                peak = max(peak, balance)
                trades.append(pos)

                if pos.net < 0:
                    loss_streak += 1
                    if loss_streak >= cfg.loss_streak:
                        cooldown = cfg.cooldown_bars
                        loss_streak = 0
                else:
                    loss_streak = 0
                if day_start_balance > 0 and (day_start_balance - balance) \
                        / day_start_balance * 100.0 >= cfg.max_daily_loss:
                    halted_day = True
                if peak > 0 and (peak - balance) / peak * 100.0 >= cfg.max_drawdown:
                    halted = True
                pos = None

        if cooldown > 0 and live:
            cooldown -= 1

    return trades, balance, halted


# ======================================================================
# Reporting
# ======================================================================
def summarise(trades, final, cfg, halted):
    if not trades:
        return None
    rs = [t.r for t in trades]
    n = len(rs)
    mean_r = sum(rs) / n
    sd = statistics.pstdev(rs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("inf")
    tstat = mean_r / se if se > 0 else 0.0
    wins = [t for t in trades if t.net > 0]
    gross_win = sum(t.net for t in wins)
    gross_loss = -sum(t.net for t in trades if t.net <= 0)
    peak, dd = cfg.deposit, 0.0
    bal = cfg.deposit
    for t in trades:
        bal += t.net
        peak = max(peak, bal)
        dd = max(dd, (peak - bal) / peak * 100.0)
    return dict(n=n, mean_r=mean_r, se=se, t=tstat, ret=(final / cfg.deposit - 1) * 100.0,
                win_rate=len(wins) / n * 100.0, dd=dd,
                pf=gross_win / gross_loss if gross_loss > 0 else float("inf"),
                halted=halted)


def report(trades, final, halted, cfg, books, zero_spread_result, control):
    print("=" * 72)
    print("XAUUSD Scalper - bar replay")
    print("=" * 72)
    for b in books:
        s = b["series"]
        sp = sorted(s["spread"])
        med = sp[len(sp) // 2] if sp else 0.0
        print(f"{b['book']:>8} book : {b['spec']['symbol']:<16} {len(s['time']):>7} bars  "
              f"{s['time'][0]:%Y-%m-%d} -> {s['time'][-1]:%Y-%m-%d}  "
              f"median spread {med:.0f} pts")
    htf = f", HTF {cfg.htf_minutes}m EMA{cfg.htf_ema}" if cfg.use_htf else ", no HTF filter"
    print(f"mode         : {cfg.mode} on {cfg.tf_minutes}m{htf}")
    print(f"exits        : stop {cfg.bo_stop_atr:g}xATR, target {cfg.bo_target_r:g}R, "
          f"time stop {cfg.max_bars_in_trade} bars, "
          f"BE {'off' if cfg.be_trigger_r <= 0 else f'{cfg.be_trigger_r:g}R'}, "
          f"trail {'off' if cfg.trail_start_r <= 0 else f'{cfg.trail_start_r:g}R'}")
    print(f"risk         : {cfg.risk_percent}% per trade "
          f"({cfg.weekend_risk_mult:g}x on the weekend book)")
    print(f"costs        : spread from the data + {cfg.commission:.2f}/lot commission")
    print()

    st = summarise(trades, final, cfg, halted)
    if st is None:
        print("NO TRADES.")
        print("The usual causes, in order: the session window does not match this")
        print("feed's server time; the spread gate is refusing everything (raise")
        print("--max-spread-points to see); the history is shorter than the warm-up.")
        return

    print(f"RETURN       : {st['ret']:+.2f}%   ({cfg.deposit:,.0f} -> {final:,.0f})")
    print(f"max drawdown : {st['dd']:.2f}%")
    print(f"trades       : {st['n']}")
    print(f"win rate     : {st['win_rate']:.1f}%")
    print(f"profit factor: {st['pf']:.2f}")
    if halted:
        print("!! the drawdown kill switch fired - the run stopped early")
    print()
    print("-" * 72)
    print(f"expectancy   : {st['mean_r']:+.4f} R  (SE {st['se']:.4f})   t = {st['t']:.2f}")
    if st["t"] >= 2.0:
        print("               t >= 2: the edge survives its own noise on this sample.")
    else:
        print("               t < 2: NOT distinguishable from zero. A positive return")
        print("               above this line is not evidence of anything yet.")
    print("-" * 72)

    # cost sensitivity
    if zero_spread_result is not None:
        zt, zf = zero_spread_result
        zs = summarise(zt, zf, cfg, False)
        if zs:
            print(f"\nat ZERO spread: {zs['ret']:+.2f}%  over {zs['n']} trades  "
                  f"({zs['mean_r']:+.4f} R/trade)")
            if zs["mean_r"] <= 0:
                print("               negative even for free - cost is not what is wrong.")
            else:
                gap = zs["mean_r"] - st["mean_r"]
                print(f"               spread costs {gap:.4f} R per trade "
                      f"({gap / max(zs['mean_r'], 1e-9) * 100:.0f}% of the gross edge)")

    # random-direction control - the empirical zero for this data and geometry
    if control:
        rets = sorted(c[0] for c in control)
        pool = [r for c in control for r in c[1]]
        beat = sum(1 for r in rets if r >= st["ret"])
        print("\nrandom-direction control - same entries, sizes, exits and costs,")
        print(f"only the direction call replaced by a coin flip ({len(rets)} seeds):")
        print(f"  return   median {sorted(rets)[len(rets) // 2]:+.2f}%   "
              f"range {rets[0]:+.2f}% .. {rets[-1]:+.2f}%")
        if pool:
            c_mean = sum(pool) / len(pool)
            c_se = statistics.pstdev(pool) / math.sqrt(len(pool)) if len(pool) > 1 else 0.0
            print(f"  expectancy {c_mean:+.4f} R over {len(pool)} control trades")
            edge = st["mean_r"] - c_mean
            se = math.sqrt(st["se"] ** 2 + c_se ** 2)
            tt = edge / se if se > 0 else 0.0
            print()
            print("=" * 72)
            print(f"EDGE over the control : {edge:+.4f} R   (SE {se:.4f})   t = {tt:.2f}")
            print("=" * 72)
            print("  This is the number to judge the bot by. The control pays the same")
            print("  spread, takes the same stops and suffers the same bar-replay")
            print("  pessimism, so everything except the direction call cancels out.")
            if tt < 2.0:
                print("  t < 2: the direction call has not been shown to add anything.")
        print(f"  {beat}/{len(rets)} coin-flip runs returned at least as much as the strategy")

    # splits
    half = len(trades) // 2
    if half:
        a = sum(t.r for t in trades[:half]) / half
        b = sum(t.r for t in trades[half:]) / (len(trades) - half)
        print(f"\nfirst half   : {a:+.4f} R over {half} trades")
        print(f"second half  : {b:+.4f} R over {len(trades) - half} trades")
        print("  (both positive, or it is one regime and not a strategy)")

    if cfg.folds > 1:
        print(f"\nwalk-forward, {cfg.folds} chronological folds:")
        size = len(trades) / cfg.folds
        for k in range(cfg.folds):
            chunk = trades[int(k * size): int((k + 1) * size)]
            if not chunk:
                continue
            m = sum(t.r for t in chunk) / len(chunk)
            print(f"  fold {k + 1}  {chunk[0].entry_time:%Y-%m-%d} -> "
                  f"{chunk[-1].exit_time:%Y-%m-%d}  n={len(chunk):>4}  {m:+.4f} R")

    # breakdowns
    def group(key, label):
        buckets = {}
        for t in trades:
            buckets.setdefault(key(t), []).append(t)
        print(f"\nby {label}:")
        for k in sorted(buckets, key=lambda x: str(x)):
            g = buckets[k]
            m = sum(x.r for x in g) / len(g)
            tot = sum(x.net for x in g)
            print(f"  {str(k):<12} n={len(g):>4}  {m:+.4f} R   total {tot:+10.2f}")

    group(lambda t: t.book, "book")
    group(lambda t: t.mode, "mechanism")
    group(lambda t: WEEKDAYS[t.entry_time.weekday()], "weekday")
    group(lambda t: f"{t.entry_time.hour:02d}:00", "entry hour (server)")
    group(lambda t: t.reason, "exit reason")
    group(lambda t: f"{t.entry_time:%Y-%m}", "month")


# ======================================================================
# Demo data - a random walk, for exercising the harness with no feed
# ======================================================================
def demo_books(cfg, seed=7):
    rng = random.Random(seed)
    spec = default_spec()
    spec.update({"point": 0.01, "tick_value": 1.0, "tick_size": 0.01,
                 "volume_min": 0.01, "volume_step": 0.01, "volume_max": 50.0})
    step = timedelta(minutes=cfg.tf_minutes)
    t = datetime(2026, 1, 5, 1, 0)
    price = 2400.0
    #--- roughly 1.2% a day, which is what gold has actually been doing
    per_bar_sigma = 0.012 * price / math.sqrt(1440.0 / cfg.tf_minutes)

    weekday = {k: [] for k in ("time", "open", "high", "low", "close", "spread")}
    weekend = {k: [] for k in ("time", "open", "high", "low", "close", "spread")}
    cal = Calendar(cfg.week_start_dow, cfg.week_start_hour * 3600 + cfg.week_start_minute * 60,
                   cfg.week_end_dow, cfg.week_end_hour * 3600 + cfg.week_end_minute * 60)

    for _ in range(int(cfg.demo_bars)):
        o = price
        price *= math.exp(rng.gauss(0.0, per_bar_sigma / price))
        c = price
        wick = abs(rng.gauss(0.0, per_bar_sigma * 0.8))
        h, lo = max(o, c) + wick, min(o, c) - wick
        spot = cal.spot_open(t)
        tgt = weekday if spot else weekend
        tgt["time"].append(t)
        tgt["open"].append(round(o, 2))
        tgt["high"].append(round(h, 2))
        tgt["low"].append(round(lo, 2))
        tgt["close"].append(round(c, 2))
        tgt["spread"].append(20.0 if spot else 60.0)
        t += step

    out = []
    for name, s in (("weekday", weekday), ("weekend", weekend)):
        if not s["time"]:
            continue
        sp = dict(spec)
        sp["symbol"] = "DEMO.XAU" + ("" if name == "weekday" else "24/7")
        out.append({"book": name, "series": s, "spec": sp})
    return out


# ======================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bars", help="weekday symbol CSV (XAUUSD.s)")
    p.add_argument("--weekend-bars", default=None, help="24/7 symbol CSV")
    p.add_argument("--demo", action="store_true",
                   help="generate a random walk instead of reading files")
    p.add_argument("--demo-bars", type=int, default=60000)
    p.add_argument("--from", dest="dt_from", default=None)
    p.add_argument("--to", dest="dt_to", default=None)

    g = p.add_argument_group("strategy")
    g.add_argument("--mode", choices=["breakout", "fade", "both"], default="breakout")
    g.add_argument("--tf-minutes", type=int, default=5)
    g.add_argument("--htf-minutes", type=int, default=60)
    g.add_argument("--htf-ema", type=int, default=50)
    g.add_argument("--no-htf", dest="use_htf", action="store_false", default=True)
    g.add_argument("--donchian", type=int, default=20)
    g.add_argument("--ema-fast", type=int, default=20)
    g.add_argument("--ema-slow", type=int, default=50)
    g.add_argument("--atr-period", type=int, default=14)
    g.add_argument("--rsi-period", type=int, default=14)
    g.add_argument("--adx-period", type=int, default=14)
    g.add_argument("--bb-period", type=int, default=20)
    g.add_argument("--bb-dev", type=float, default=2.0)
    g.add_argument("--expansion-atr", type=float, default=0.80)
    g.add_argument("--body-frac", type=float, default=0.50)
    g.add_argument("--min-adx", type=float, default=18.0)
    g.add_argument("--max-stretch-atr", type=float, default=2.50)
    g.add_argument("--bo-stop-atr", type=float, default=1.00)
    g.add_argument("--bo-target-r", type=float, default=1.60)
    g.add_argument("--fade-rsi", type=float, default=25.0)
    g.add_argument("--fade-stretch-atr", type=float, default=1.00)
    g.add_argument("--fade-max-adx", type=float, default=25.0)
    g.add_argument("--no-fade-htf", dest="fade_with_htf", action="store_false", default=True)
    g.add_argument("--fade-stop-atr", type=float, default=1.20)
    g.add_argument("--fade-target-r", type=float, default=1.00)
    g.add_argument("--max-stop-atr", type=float, default=2.50)

    e = p.add_argument_group("exits")
    e.add_argument("--max-bars-in-trade", type=int, default=24)
    e.add_argument("--be-trigger-r", type=float, default=0.0,
               help="0 = off. Measured cost on a driftless series: ~0.04 R/trade")
    e.add_argument("--be-offset-r", type=float, default=0.05)
    e.add_argument("--trail-start-r", type=float, default=0.0,
               help="0 = off. Measured cost on a driftless series: ~0.06 R/trade")
    e.add_argument("--trail-atr", type=float, default=1.20)

    r = p.add_argument_group("risk and sessions")
    r.add_argument("--deposit", type=float, default=10000.0)
    r.add_argument("--risk-percent", type=float, default=0.25)
    r.add_argument("--weekend-risk-mult", type=float, default=0.50)
    r.add_argument("--commission", type=float, default=0.0,
                   help="per lot, round turn, in account currency")
    r.add_argument("--max-spread-points", type=float, default=40.0)
    r.add_argument("--max-spread-frac", type=float, default=0.12,
                   help="spread cap as a share of the stop distance - the gate that "
                        "decides whether gold scalping clears its own costs")
    r.add_argument("--we-max-spread-points", type=float, default=150.0)
    r.add_argument("--we-max-spread-frac", type=float, default=0.25,
                   help="the same cap on the weekend book, where spreads are wider")
    r.add_argument("--max-daily-loss", type=float, default=2.0)
    r.add_argument("--max-drawdown", type=float, default=12.0)
    r.add_argument("--max-trades-per-day", type=int, default=12)
    r.add_argument("--loss-streak", type=int, default=3)
    r.add_argument("--cooldown-bars", type=int, default=12)
    r.add_argument("--max-lots", type=float, default=5.0)
    r.add_argument("--session-start", type=int, default=9)
    r.add_argument("--session-end", type=int, default=23)
    r.add_argument("--we-session-start", type=int, default=0)
    r.add_argument("--we-session-end", type=int, default=0)
    r.add_argument("--flatten-minutes", type=int, default=20)

    w = p.add_argument_group("weekend clock (BROKER SERVER TIME)")
    w.add_argument("--week-start-dow", type=int, default=1, help="0=Sun .. 6=Sat")
    w.add_argument("--week-start-hour", type=int, default=1)
    w.add_argument("--week-start-minute", type=int, default=5)
    w.add_argument("--week-end-dow", type=int, default=5)
    w.add_argument("--week-end-hour", type=int, default=23)
    w.add_argument("--week-end-minute", type=int, default=45)

    a = p.add_argument_group("analysis")
    a.add_argument("--folds", type=int, default=4)
    a.add_argument("--control-seeds", type=int, default=20,
                   help="random-direction control runs (0 to skip)")
    a.add_argument("--no-zero-spread", dest="zero_spread_run",
                   action="store_false", default=True)

    o = p.add_argument_group("contract overrides (when the CSV has no # header)")
    o.add_argument("--point", type=float, default=None)
    o.add_argument("--tick-value", type=float, default=None)
    o.add_argument("--tick-size", type=float, default=None)

    cfg = p.parse_args()
    if not cfg.demo and not cfg.bars:
        p.error("--bars is required (or use --demo)")

    over = {"point": cfg.point, "tick_value": cfg.tick_value, "tick_size": cfg.tick_size}

    if cfg.demo:
        books = demo_books(cfg)
        print("DEMO MODE: these bars are a random walk with no structure in them.")
        print("A strategy cannot make money here. What this proves is that the")
        print("harness runs and that the reported expectancy lands where it should:")
        print("at roughly minus the trading cost.\n")
    else:
        books = []
        s, spec = load_bars(cfg.bars, over)
        books.append({"book": "weekday", "series": s, "spec": spec})
        if cfg.weekend_bars:
            s2, spec2 = load_bars(cfg.weekend_bars, over)
            books.append({"book": "weekend", "series": s2, "spec": spec2})

    if cfg.dt_from or cfg.dt_to:
        lo = datetime.fromisoformat(cfg.dt_from) if cfg.dt_from else datetime.min
        hi = datetime.fromisoformat(cfg.dt_to) if cfg.dt_to else datetime.max
        for b in books:
            keep = [i for i, t in enumerate(b["series"]["time"]) if lo <= t < hi]
            for k in b["series"]:
                b["series"][k] = [b["series"][k][i] for i in keep]
        books = [b for b in books if b["series"]["time"]]
        if not books:
            raise SystemExit("no bars left after the date filter")

    for b in books:
        b["ind"] = build_indicators(b["series"], cfg)

    cal = Calendar(cfg.week_start_dow,
                   cfg.week_start_hour * 3600 + cfg.week_start_minute * 60,
                   cfg.week_end_dow,
                   cfg.week_end_hour * 3600 + cfg.week_end_minute * 60)

    trades, final, halted = run(books, cfg, cal)

    zero = None
    if cfg.zero_spread_run:
        zt, zf, _ = run(books, cfg, cal, zero_spread=True)
        zero = (zt, zf)

    control = []
    for seed in range(cfg.control_seeds):
        rng = random.Random(1000 + seed)
        ct, cf, _ = run(books, cfg, cal, rng=rng, random_direction=True)
        control.append(((cf / cfg.deposit - 1) * 100.0, [x.r for x in ct]))

    report(trades, final, halted, cfg, books, zero, control)


if __name__ == "__main__":
    main()

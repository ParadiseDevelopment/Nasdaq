#!/usr/bin/env python3
"""
Bar-replay backtest for NAS100_ML_Bot.

This is a FAITHFUL PYTHON REPLICA of the EA, not the EA itself. It reproduces
the same 40 features, the same triple-barrier labelling, the same three online
experts with the same AdamW settings, the same Hedge blending, the same entry
gates and the same risk/exit rules - but it replays OHLC bars, not ticks.

Use it to get a fast, honest first number and to iterate on parameters. The
authoritative result still comes from MetaTrader's Strategy Tester on real
ticks with your broker's execution; see docs/BACKTESTING.md for the
differences that matter.

Input
-----
CSV produced by MQL5/Scripts/ExportBars.mq5:

    #symbol=... tick_value=... tick_size=... volume_min=... ...
    time,open,high,low,close,tick_volume,spread
    2024.01.02 09:00,16543.2,16560.1,16538.0,16552.4,1832,12

The leading '#' line is optional; without it, pass --tick-value / --tick-size
etc. on the command line.

Usage
-----
    python3 tools/backtest.py --bars bars.csv
    python3 tools/backtest.py --bars bars.csv --from 2024-01-01 --to 2025-01-01
    python3 tools/backtest.py --bars bars.csv --commission 4.0 --risk-percent 0.25

Requires numpy and pandas.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_offline import Logistic, Mlp, Rff, log_loss  # noqa: E402

N_FEATURES = 40


# ======================================================================
# Indicators - matched to the MetaTrader 5 built-ins the EA uses
# ======================================================================
def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing, as used by MT5's ATR, RSI and ADX."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"],
                      (df["high"] - pc).abs(),
                      (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return _rma(true_range(df), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = _rma(d.clip(lower=0.0), n)
    loss = _rma((-d).clip(lower=0.0), n)
    rs = gain / loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, slowing: int = 3):
    """MT5 iStochastic(k, d, slowing, MODE_SMA, STO_LOWHIGH)."""
    ll = df["low"].rolling(k).min()
    hh = df["high"].rolling(k).max()
    num = (df["close"] - ll).rolling(slowing).sum()
    den = (hh - ll).rolling(slowing).sum()
    kline = 100.0 * num / den.replace(0.0, np.nan)
    kline = kline.fillna(50.0)
    return kline, kline.rolling(d).mean().fillna(50.0)


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    """MT5's built-in MACD: EMA difference, SMA signal line."""
    main = _ema(close, fast) - _ema(close, slow)
    return main, main.rolling(signal).mean()


def bollinger(close: pd.Series, n: int = 20, dev: float = 2.0):
    base = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return base + dev * sd, base - dev * sd


def adx(df: pd.DataFrame, n: int = 14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr_n = _rma(true_range(df), n)
    plus_di = 100.0 * _rma(pd.Series(plus_dm, index=df.index), n) / tr_n
    minus_di = 100.0 * _rma(pd.Series(minus_dm, index=df.index), n) / tr_n
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return _rma(dx.fillna(0.0), n), plus_di, minus_di


# ======================================================================
# Feature matrix - index-for-index with FeatureEngine.mqh
# ======================================================================
def build_features(df: pd.DataFrame, sessions: dict) -> np.ndarray:
    n = len(df)
    F = np.zeros((n, N_FEATURES))

    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    a14 = atr(df, 14)
    a50 = atr(df, 50)
    r = rsi(c, 14)
    e20, e50, e200 = _ema(c, 20), _ema(c, 50), _ema(c, 200)
    m_main, m_sig = macd(c)
    bb_up, bb_lo = bollinger(c)
    sk, sd = stochastic(df)
    adx_v, di_p, di_m = adx(df)

    A = a14.replace(0.0, np.nan)

    def sd_(x, y, fill=0.0):
        return (x / y).replace([np.inf, -np.inf], np.nan).fillna(fill).to_numpy()

    #--- 0..5 multi-horizon momentum
    for j, lag in enumerate((1, 2, 3, 5, 8, 13)):
        F[:, j] = sd_(c - c.shift(lag), A)

    #--- 6..9 oscillators
    F[:, 6] = ((r - 50.0) / 50.0).to_numpy()
    F[:, 7] = ((r - r.shift(3)) / 50.0).fillna(0.0).to_numpy()
    F[:, 8] = ((sk - 50.0) / 50.0).to_numpy()
    F[:, 9] = ((sk - sd) / 50.0).to_numpy()

    #--- 10..13 trend / band structure
    F[:, 10] = sd_(m_main - m_sig, A)
    F[:, 11] = sd_(m_main, A)
    bw = (bb_up - bb_lo)
    F[:, 12] = np.clip(sd_(c - bb_lo, bw.replace(0.0, np.nan), 0.5) * 2.0 - 1.0, -3, 3)
    F[:, 13] = sd_(bw, A) - 4.0

    #--- 14..19 moving-average distance and slope
    F[:, 14] = sd_(c - e20, A)
    F[:, 15] = sd_(c - e50, A)
    F[:, 16] = sd_(c - e200, A)
    F[:, 17] = sd_(e20 - e20.shift(5), A)
    F[:, 18] = sd_(e50 - e50.shift(10), A)
    F[:, 19] = sd_(e20 - e50, A)

    #--- 20..21 volatility regime
    F[:, 20] = sd_(a14, a50.replace(0.0, np.nan), 1.0) - 1.0
    F[:, 21] = sd_(a14, c) * 1000.0

    #--- 22..25 candle anatomy
    rng = (h - l).replace(0.0, np.nan)
    F[:, 22] = sd_(c - o, rng)
    F[:, 23] = sd_(h - pd.concat([o, c], axis=1).max(axis=1), rng)
    F[:, 24] = sd_(pd.concat([o, c], axis=1).min(axis=1) - l, rng)
    F[:, 25] = sd_(h - l, A) - 1.0

    #--- 26 tick volume against its own 20-bar mean
    v = df["tick_volume"].astype(float)
    F[:, 26] = np.clip(sd_(v, v.shift(1).rolling(20).mean(), 1.0) - 1.0, -3, 5)

    #--- 27..28 directional strength
    F[:, 27] = ((adx_v - 25.0) / 25.0).fillna(0.0).to_numpy()
    F[:, 28] = ((di_p - di_m) / 50.0).fillna(0.0).to_numpy()

    #--- 29 signed streak of consecutive same-direction closes
    up_bar = (c > c.shift(1)).to_numpy()
    streak = np.zeros(n)
    for i in range(1, n):
        k = 0
        for back in range(10):
            j = i - back
            if j - 1 < 0 or up_bar[j] != up_bar[i]:
                break
            k += 1
        streak[i] = (k if up_bar[i] else -k) / 5.0
    F[:, 29] = streak

    #--- 30..32 previous-day reference levels
    day = df.index.normalize()
    daily = df.groupby(day).agg(pdh=("high", "max"), pdl=("low", "min"), pdc=("close", "last"))
    prev = daily.shift(1).reindex(day).to_numpy()
    av = A.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        F[:, 30] = np.clip(np.nan_to_num((c.to_numpy() - prev[:, 0]) / av), -20, 20)
        F[:, 31] = np.clip(np.nan_to_num((c.to_numpy() - prev[:, 1]) / av), -20, 20)
        F[:, 32] = np.clip(np.nan_to_num((c.to_numpy() - prev[:, 2]) / av), -20, 20)

    #--- 33..39 calendar and session context
    hour_frac = df.index.hour + df.index.minute / 60.0
    F[:, 33] = np.sin(2 * np.pi * hour_frac / 24.0)
    F[:, 34] = np.cos(2 * np.pi * hour_frac / 24.0)
    dow = (df.index.dayofweek + 1) % 7          # MQL5 day_of_week: Sunday = 0
    F[:, 35] = np.sin(2 * np.pi * dow / 7.0)
    F[:, 36] = np.cos(2 * np.pi * dow / 7.0)
    hh = df.index.hour
    F[:, 37] = ((hh >= sessions["london_start"]) & (hh < sessions["london_end"])).astype(float)
    F[:, 38] = ((hh >= sessions["ny_start"]) & (hh < sessions["ny_end"])).astype(float)
    F[:, 39] = ((hour_frac >= sessions["ny_start"]) &
                (hour_frac < sessions["ny_start"] + 1.5)).astype(float)

    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(F, -50.0, 50.0)


# ======================================================================
# Streaming standardiser - CScaler
# ======================================================================
class Scaler:
    def __init__(self, n):
        self.n, self.count = n, 0
        self.mean = np.zeros(n)
        self.m2 = np.zeros(n)
        self.std = np.ones(n)

    def observe(self, x):
        self.count += 1
        d = x - self.mean
        self.mean += d / self.count
        self.m2 += d * (x - self.mean)
        if self.count > 1:
            self.std = np.sqrt(np.maximum(self.m2 / (self.count - 1), 1e-10))

    def transform(self, x):
        return np.clip((x - self.mean) / np.maximum(self.std, 1e-8), -5.0, 5.0)


# ======================================================================
# Hedge ensemble - CEnsemble
# ======================================================================
class Ensemble:
    def __init__(self, n, cfg, rng):
        #--- A single random projection / weight init makes the whole result a
        #--- seed lottery. Running several of each and letting Hedge weight
        #--- them averages that variance away instead of gambling on one draw.
        n_rff = getattr(cfg, "n_rff", 1)
        n_mlp = getattr(cfg, "n_mlp", 1)
        self.lin = Logistic(n)
        self.rffs = [Rff(n, cfg.rff_dim, cfg.rff_sigma, rng) for _ in range(n_rff)]
        self.mlps = [Mlp(n, cfg.mlp_hidden, rng) for _ in range(n_mlp)]
        self.experts = [self.lin] + self.rffs + self.mlps
        self.lrs = [cfg.lr] + [cfg.lr] * n_rff + [cfg.lr * 0.6] * n_mlp
        k = len(self.experts)
        self.floor = 0.5 / k
        self.w = np.ones(k) / k
        self.eta = cfg.hedge_eta
        self.l2 = cfg.l2
        self.hits: list[float] = []
        self.lls: list[float] = []
        self.labels: list[float] = []
        self.seen = 0
        self.eval_window = cfg.eval_window
        self.last_p = np.full(len(self.experts), 0.5)

    def predict(self, x):
        self.last_p = np.array([m.predict(x[None, :])[0] for m in self.experts])
        return float(self.w @ self.last_p)

    def disagreement(self):
        return float(self.last_p.max() - self.last_p.min())

    def rolling_accuracy(self):
        if len(self.hits) < 20:
            return 0.5
        return float(np.mean(self.hits[-self.eval_window:]))

    def majority_baseline(self):
        """Accuracy of always calling the more common class over the same window.

        Reporting accuracy without this is how a model that has only learned
        the base rate gets mistaken for one that has learned something."""
        w = self.labels[-self.eval_window:]
        if len(w) < 20:
            return 0.5
        up = float(np.mean(w))
        return max(up, 1.0 - up)

    def rolling_logloss(self):
        if len(self.lls) < 20:
            return 0.6931
        return float(np.mean(self.lls[-self.eval_window:]))

    def evaluate(self, x, y):
        ps = np.array([m.predict(x[None, :])[0] for m in self.experts])
        pe = float(self.w @ ps)
        self.hits.append(1.0 if (pe >= 0.5) == (y > 0.5) else 0.0)
        self.lls.append(log_loss(np.array([pe]), np.array([y])))
        self.labels.append(y)
        if self.eta > 0:
            ll = np.array([log_loss(np.array([p]), np.array([y])) for p in ps])
            w = self.w * np.exp(-self.eta * ll)
            w = np.maximum(w / w.sum(), self.floor)
            self.w = w / w.sum()
        self.seen += 1

    def train(self, x, y, weight):
        if weight <= 0:
            return
        xb, yb, wb = x[None, :], np.array([y]), np.array([weight])
        for m, lr in zip(self.experts, self.lrs):
            m.fit_epoch(xb, yb, wb, lr, self.l2, 1)


# ======================================================================
# Triple-barrier labeller - CLabeler
# ======================================================================
class Labeler:
    def __init__(self, horizon, barrier_atr, time_weight):
        self.h, self.k, self.tw = horizon, barrier_atr, time_weight
        self.pend: list[dict] = []

    def add(self, x, entry, atr_v):
        if atr_v <= 0:
            return
        self.pend.append({"x": x, "entry": entry,
                          "up": entry + self.k * atr_v,
                          "dn": entry - self.k * atr_v, "age": 0})

    def update(self, high, low, close):
        out, keep = [], []
        for p in self.pend:
            p["age"] += 1
            hit_up, hit_dn = high >= p["up"], low <= p["dn"]
            if hit_up and hit_dn:
                out.append((p["x"], 1.0 if close > p["entry"] else 0.0, 0.5 * self.tw))
            elif hit_up:
                out.append((p["x"], 1.0, 1.0))
            elif hit_dn:
                out.append((p["x"], 0.0, 1.0))
            elif p["age"] >= self.h:
                out.append((p["x"], 1.0 if close > p["entry"] else 0.0, self.tw))
            else:
                keep.append(p)
        self.pend = keep
        return out


# ======================================================================
# Class-balanced replay - CReplayBuffer
# ======================================================================
class Replay:
    def __init__(self, cap, rng):
        self.cap, self.rng = cap, rng
        self.X: list[np.ndarray] = []
        self.Y: list[float] = []
        self.W: list[float] = []

    def add(self, x, y, w):
        self.X.append(x); self.Y.append(y); self.W.append(w)
        if len(self.X) > self.cap:
            self.X.pop(0); self.Y.pop(0); self.W.pop(0)

    def sample(self):
        if not self.X:
            return None
        i = self.rng.integers(len(self.X))
        if self.rng.random() < 0.5:
            share = float(np.mean(self.Y))
            want = 1.0 if share < 0.5 else 0.0
            for _ in range(24):
                j = self.rng.integers(len(self.X))
                if (1.0 if self.Y[j] > 0.5 else 0.0) == want:
                    i = j
                    break
        return self.X[i], self.Y[i], self.W[i]


# ======================================================================
# Backtest engine
# ======================================================================
class Backtest:
    def __init__(self, df, F, cfg, spec):
        self.df, self.F, self.cfg, self.spec = df, F, cfg, spec
        self.rng = np.random.default_rng(cfg.seed)
        np.random.seed(cfg.seed)

        self.scaler = Scaler(N_FEATURES)
        self.ens = Ensemble(N_FEATURES, cfg, self.rng)
        self.lab = Labeler(cfg.label_horizon, cfg.barrier_atr, cfg.time_barrier_weight)
        self.replay = Replay(cfg.replay_capacity, self.rng)

        self.balance = cfg.deposit
        self.equity_peak = cfg.deposit
        self.equity_curve: list[tuple] = []
        self.trades: list[dict] = []
        self.pos: dict | None = None

        self.day = None
        self.day_start_equity = cfg.deposit
        self.trades_today = 0
        self.halted_day = False
        self.halted_forever = False
        self.loss_streak = 0
        self.cooldown = 0
        self.blocks: dict[str, int] = {}
        self._sample_i = 0

    # ---------------------------------------------------------------
    def _block(self, reason):
        self.blocks[reason] = self.blocks.get(reason, 0) + 1

    def _lots(self, stop_dist, scale):
        vpu = self.spec["tick_value"] / self.spec["tick_size"]
        risk_money = self.balance * (self.cfg.risk_percent / 100.0) * np.clip(scale, 0.25, 2.0)
        raw = risk_money / (stop_dist * vpu)
        step = self.spec["volume_step"]
        lots = np.floor(raw / step) * step
        lots = min(lots, self.spec["volume_max"], self.cfg.max_lots)
        return 0.0 if lots < self.spec["volume_min"] else round(lots, 8)

    # ---------------------------------------------------------------
    def _close(self, exit_price, when, reason):
        p = self.pos
        vpu = self.spec["tick_value"] / self.spec["tick_size"]
        move = (exit_price - p["entry"]) if p["long"] else (p["entry"] - exit_price)
        gross = move * p["lots"] * vpu
        cost = (p["spread_price"] * p["lots"] * vpu) + self.cfg.commission * p["lots"]
        net = gross - cost

        self.balance += net
        self.loss_streak = self.loss_streak + 1 if net < 0 else 0
        if self.loss_streak >= self.cfg.loss_streak_trigger:
            self.cooldown = self.cfg.cooldown_bars
            self.loss_streak = 0

        self.trades.append({
            "open_time": p["time"], "close_time": when, "long": p["long"],
            "lots": p["lots"], "entry": p["entry"], "exit": exit_price,
            "profit": net, "reason": reason, "prob": p["prob"],
            "r": net / max(p["risk_money"], 1e-9),
        })
        self.pos = None

    # ---------------------------------------------------------------
    def _manage(self, i):
        """Walk one bar of an open position: stops, targets, trail, time stop."""
        p, bar = self.pos, self.df.iloc[i]
        hi, lo = bar["high"], bar["low"]
        long = p["long"]

        hit_sl = (lo <= p["sl"]) if long else (hi >= p["sl"])
        hit_tp = (hi >= p["tp"]) if long else (lo <= p["tp"])

        if hit_sl and hit_tp:
            # Ambiguous inside one bar. Assume the stop went first unless the
            # caller explicitly chose the optimistic reading.
            if self.cfg.ambiguous_bar == "stop":
                self._close(p["sl"], bar.name, "sl")
            else:
                self._close(p["tp"], bar.name, "tp")
            return
        if hit_sl:
            self._close(p["sl"], bar.name, "sl")
            return
        if hit_tp:
            self._close(p["tp"], bar.name, "tp")
            return

        #--- break even, then ATR trail, both measured in R
        close, atr_v = bar["close"], p["atr"]
        moved = (close - p["entry"]) if long else (p["entry"] - close)
        r = moved / p["risk_price"]
        new_sl = p["sl"]
        if r >= self.cfg.breakeven_r:
            be = p["entry"] + (1 if long else -1) * self.cfg.breakeven_offset_r * p["risk_price"]
            new_sl = max(new_sl, be) if long else min(new_sl, be)
        if r >= self.cfg.trail_start_r:
            tr = close - self.cfg.trail_atr * atr_v if long else close + self.cfg.trail_atr * atr_v
            new_sl = max(new_sl, tr) if long else min(new_sl, tr)
        p["sl"] = new_sl

        p["bars"] += 1
        if self.cfg.max_hold_bars > 0 and p["bars"] >= self.cfg.max_hold_bars:
            self._close(close, bar.name, "time")

    # ---------------------------------------------------------------
    def run(self):
        cfg, df, F = self.cfg, self.df, self.F
        n = len(df)
        start = max(260, cfg.label_horizon + 5)

        for i in range(start, n - 1):
            bar = df.iloc[i]
            t = bar.name
            atr_v = bar["atr"]
            if not np.isfinite(atr_v) or atr_v <= 0:
                continue

            #--- daily roll and equity accounting ----------------------
            equity = self.balance
            if self.pos is not None:
                vpu = self.spec["tick_value"] / self.spec["tick_size"]
                mv = (bar["close"] - self.pos["entry"]) if self.pos["long"] \
                    else (self.pos["entry"] - bar["close"])
                equity += mv * self.pos["lots"] * vpu
                equity -= (self.pos["spread_price"] * self.pos["lots"] * vpu
                           + self.cfg.commission * self.pos["lots"])
            self.equity_peak = max(self.equity_peak, equity)
            self.equity_curve.append((t, equity))

            d = t.date()
            if d != self.day:
                self.day = d
                self.day_start_equity = equity
                self.trades_today = 0
                self.halted_day = False
            if self.cooldown > 0:
                self.cooldown -= 1
            if (equity - self.day_start_equity) / max(self.day_start_equity, 1e-9) * 100.0 \
                    <= -cfg.max_daily_loss_pct:
                self.halted_day = True
            if (self.equity_peak - equity) / max(self.equity_peak, 1e-9) * 100.0 \
                    >= cfg.max_drawdown_pct:
                self.halted_forever = True

            #--- learn from whatever matured on this bar ---------------
            for x_raw, y, w in self.lab.update(bar["high"], bar["low"], bar["close"]):
                #--- A sample opens every bar but takes label_horizon bars to
                #--- resolve, so neighbours share almost all of their outcome.
                #--- Scoring every one inflates accuracy; stride >1 keeps only
                #--- roughly independent samples for the walk-forward metric.
                self._sample_i += 1
                score_this = (self._sample_i % max(1, cfg.label_stride) == 0)
                self.scaler.observe(x_raw)
                xs = self.scaler.transform(x_raw)
                if score_this:
                    self.ens.evaluate(xs, y)
                self.ens.train(xs, y, w * cfg.overlap_weight)
                self.replay.add(xs, y, w)
                for _ in range(cfg.replay_steps):
                    s = self.replay.sample()
                    if s is None:
                        break
                    self.ens.train(s[0], s[1], s[2])

            raw = F[i]
            self.lab.add(raw, bar["close"], atr_v)

            xs = self.scaler.transform(raw)
            prob = self.ens.predict(xs)

            #--- manage the position on this bar, entry bar included -----
            if self.pos is not None:
                self._manage(i)
                continue

            #--- entry gates -------------------------------------------
            if self.ens.seen < cfg.min_train_samples:
                self._block("warm-up"); continue
            if self.ens.rolling_accuracy() < cfg.min_roll_accuracy:
                self._block("accuracy floor"); continue
            if self.ens.disagreement() > cfg.max_disagreement:
                self._block("expert disagreement"); continue

            #--- the spread has to be paid out of the edge, so express the
            #--- entry test in R: expected R at a 1:1 barrier is (2p-1), and
            #--- it must clear cost plus a margin before the trade is worth it
            spread_now = bar["spread"] * self.spec["point"]
            cost_r = spread_now / (cfg.stop_atr * atr_v)
            need = 0.5 + (cost_r + cfg.min_expected_r) / 2.0
            thresh = max(cfg.prob_threshold, need) if cfg.cost_aware else cfg.prob_threshold

            want_long = prob >= thresh and cfg.allow_long
            want_short = prob <= 1.0 - thresh and cfg.allow_short
            if not (want_long or want_short):
                self._block("no edge"); continue

            if self.halted_forever:
                self._block("max drawdown"); continue
            if self.halted_day:
                self._block("daily loss stop"); continue
            if self.cooldown > 0:
                self._block("loss-streak cooldown"); continue
            if self.trades_today >= cfg.max_trades_per_day:
                self._block("daily trade cap"); continue

            hour = t.hour
            in_sess = (cfg.trade_start <= hour < cfg.trade_end) if cfg.trade_start <= cfg.trade_end \
                else (hour >= cfg.trade_start or hour < cfg.trade_end)
            if cfg.use_session and not in_sess:
                self._block("outside session"); continue

            spread_price = bar["spread"] * self.spec["point"]
            if bar["spread"] > cfg.max_spread_points or spread_price > cfg.max_spread_atr_frac * atr_v:
                self._block("spread too wide"); continue

            #--- size and send ------------------------------------------
            stop_dist = cfg.stop_atr * atr_v
            edge = abs(prob - 0.5) * 2.0
            lots = self._lots(stop_dist, float(np.clip(0.6 + edge, 0.6, 1.4)))
            if lots <= 0:
                self._block("below minimum lot"); continue

            nxt = df.iloc[i + 1]
            entry = nxt["open"]
            long = want_long
            vpu = self.spec["tick_value"] / self.spec["tick_size"]
            self.pos = {
                "time": nxt.name, "long": long, "lots": lots, "entry": entry,
                "sl": entry - stop_dist if long else entry + stop_dist,
                "tp": entry + cfg.take_profit_r * stop_dist if long
                      else entry - cfg.take_profit_r * stop_dist,
                "risk_price": stop_dist, "risk_money": stop_dist * lots * vpu,
                "atr": atr_v, "bars": 0, "prob": prob,
                "spread_price": spread_price,
            }
            self.trades_today += 1

        if self.pos is not None:
            self._close(df.iloc[-1]["close"], df.index[-1], "end of data")
        return self


# ======================================================================
# Reporting
# ======================================================================
def report(bt: Backtest, cfg, df):
    tr = pd.DataFrame(bt.trades)
    eq = pd.DataFrame(bt.equity_curve, columns=["time", "equity"]).set_index("time")

    #--- --report-from replays the whole file but scores only a tail window,
    #--- so the model is already warmed up and adapted at the window's start.
    #--- That is the "I have been running this for a while" question, which is
    #--- a different one from starting the bot cold on a short history.
    base = cfg.deposit
    window_note = ""
    if getattr(cfg, "report_from", None):
        rf = pd.Timestamp(cfg.report_from)
        if not tr.empty:
            base = cfg.deposit + tr[tr.close_time < rf].profit.sum()
            tr = tr[tr.close_time >= rf].reset_index(drop=True)
        eq = eq[eq.index >= rf]
        df = df[df.index >= rf]
        window_note = f" (reporting from {rf:%Y-%m-%d}, warmed up on prior data)"

    print("=" * 72)
    print("NAS100 ML Bot - bar-replay backtest (Python replica, NOT MT5)")
    print("=" * 72)
    print(f"symbol / timeframe : {bt.spec.get('symbol','?')} {bt.spec.get('timeframe','?')}")
    print(f"period             : {df.index[0]}  ->  {df.index[-1]}{window_note}")
    print(f"bars               : {len(df):,}")
    print(f"starting equity    : {base:,.2f}")
    print(f"commission         : {cfg.commission:.2f} per lot per round turn")
    print(f"ambiguous bars     : resolved as {cfg.ambiguous_bar.upper()} first")
    print()

    if tr.empty:
        print("NO TRADES were taken in the reported window."
              if window_note else "NO TRADES were taken.")
        print()
        print("Why the EA stood aside (gate hit counts per bar):")
        for k, v in sorted(bt.blocks.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<24} {v:>8,}")
        print()
        acc, base = bt.ens.rolling_accuracy(), bt.ens.majority_baseline()
        print(f"walk-forward accuracy: {acc:.4f}  vs majority baseline {base:.4f} "
              f"(edge {acc - base:+.4f}) over {bt.ens.seen:,} scored samples")
        return

    final = base + tr.profit.sum()
    ret_pct = (final / base - 1.0) * 100.0
    wins = tr[tr.profit > 0]
    losses = tr[tr.profit <= 0]
    gross_win = wins.profit.sum()
    gross_loss = -losses.profit.sum()

    run_max = eq.equity.cummax()
    dd = (run_max - eq.equity) / run_max * 100.0
    max_dd = dd.max()

    days = max((df.index[-1] - df.index[0]).days, 1)
    cagr = ((final / base) ** (365.0 / days) - 1.0) * 100.0

    daily = eq.equity.resample("1D").last().dropna()
    dret = daily.pct_change().dropna()
    sharpe = (dret.mean() / dret.std() * np.sqrt(252)) if len(dret) > 5 and dret.std() > 0 else float("nan")

    print(f"RETURN             : {ret_pct:+.2f}%   ({base:,.2f} -> {final:,.2f})")
    print(f"annualised         : {cagr:+.2f}%")
    print(f"max drawdown       : {max_dd:.2f}%")
    print(f"Sharpe (daily)     : {sharpe:.2f}")
    print()
    print(f"trades             : {len(tr)}")
    print(f"win rate           : {len(wins) / len(tr) * 100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"profit factor      : {gross_win / gross_loss:.2f}" if gross_loss > 0 else "profit factor      : inf")
    print(f"avg trade          : {tr.profit.mean():+,.2f}   ({tr.r.mean():+.3f} R)")
    print(f"best / worst       : {tr.profit.max():+,.2f} / {tr.profit.min():+,.2f}")
    n_long = int(tr.long.astype(bool).sum())
    print(f"long / short       : {n_long} / {len(tr) - n_long}")
    print()
    print("exit reason breakdown:")
    for reason, grp in tr.groupby("reason"):
        print(f"  {reason:<12} {len(grp):>5}  net {grp.profit.sum():+12,.2f}  "
              f"win {len(grp[grp.profit > 0]) / len(grp) * 100:5.1f}%")
    print()
    acc, base = bt.ens.rolling_accuracy(), bt.ens.majority_baseline()
    print(f"walk-forward accuracy : {acc:.4f}  (last {bt.ens.eval_window}) "
          f"over {bt.ens.seen:,} scored samples")
    print(f"majority baseline     : {base:.4f}   ->  EDGE {acc - base:+.4f}"
          + ("   <-- no edge; the model has only learned the base rate"
             if acc - base <= 0.0 else ""))
    print(f"rolling log loss      : {bt.ens.rolling_logloss():.4f}")
    e = bt.ens
    nr, nm = len(e.rffs), len(e.mlps)
    print(f"experts               : 1 logistic + {nr} rff + {nm} mlp = {len(e.experts)}")
    print(f"hedge mass            : lin {e.w[0]:.3f} | rff {e.w[1:1+nr].sum():.3f} "
          f"| mlp {e.w[1+nr:].sum():.3f}")
    print()

    bh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1.0) * 100.0
    print(f"buy and hold over the same period: {bh:+.2f}%")
    print()

    monthly = tr.set_index("close_time").profit.resample("1ME").sum()
    if len(monthly) > 1:
        print("monthly net:")
        for m, v in monthly.items():
            bar = "#" * min(40, int(abs(v) / max(monthly.abs().max(), 1e-9) * 40))
            print(f"  {m:%Y-%m}  {v:+12,.2f}  {bar}")
        print()

    if bt.blocks:
        print("bars skipped by gate:")
        for k, v in sorted(bt.blocks.items(), key=lambda kv: -kv[1])[:6]:
            print(f"  {k:<24} {v:>8,}")


# ======================================================================
def load_bars(path, spec_overrides):
    """Read either ExportBars.mq5 output or MetaTrader's own bar export.

    MT5's chart "Save as" writes a tab-separated file whose header is
    <DATE>\t<TIME>\t<OPEN>\t...\t<SPREAD> and which carries no contract
    spec, so tick_value / tick_size have to be supplied or defaulted.
    """
    spec = {"point": 0.01, "tick_value": 0.01, "tick_size": 0.01,
            "volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0,
            "symbol": "?", "timeframe": "?"}

    with open(path, "r", encoding="ascii", errors="replace") as fh:
        first = fh.readline()

    if first.lstrip().startswith("<DATE>"):
        #--- MetaTrader native export
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip().strip("<>").lower() for c in df.columns]
        df["time"] = pd.to_datetime(df["date"] + " " + df["time"],
                                    format="%Y.%m.%d %H:%M:%S")
        df = df.rename(columns={"tickvol": "tick_volume"})
        df = df.set_index("time").sort_index()
        stem = os.path.basename(path)
        if "_" in stem:
            bits = stem.split("_")
            spec["symbol"] = bits[0].split("-")[-1]
            if len(bits) > 1:
                spec["timeframe"] = bits[1]
    else:
        skip = 0
        if first.startswith("#"):
            skip = 1
            for tok in first[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    if k in ("symbol", "timeframe"):
                        spec[k] = v
                    elif k in spec:
                        try:
                            spec[k] = float(v)
                        except ValueError:
                            pass
        df = pd.read_csv(path, skiprows=skip)
        df.columns = [c.strip().lower() for c in df.columns]
        df["time"] = pd.to_datetime(df["time"], format="mixed", dayfirst=False)
        df = df.set_index("time").sort_index()

    spec.update({k: v for k, v in spec_overrides.items() if v is not None})

    keep = ["open", "high", "low", "close"]
    df = df[keep + [c for c in ("tick_volume", "spread") if c in df.columns]].copy()
    if "spread" not in df:
        df["spread"] = 0.0
    if "tick_volume" not in df:
        df["tick_volume"] = 1.0
    df = df[np.isfinite(df[keep]).all(axis=1)]
    return df, spec


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bars", required=True, help="CSV from MQL5/Scripts/ExportBars.mq5")
    p.add_argument("--from", dest="dt_from", default=None)
    p.add_argument("--to", dest="dt_to", default=None)
    p.add_argument("--report-from", dest="report_from", default=None,
                   help="replay everything but report only from this date, so the "
                        "model enters the window already trained")
    p.add_argument("--deposit", type=float, default=10000.0)
    p.add_argument("--commission", type=float, default=0.0,
                   help="account currency per lot per round turn")
    p.add_argument("--ambiguous-bar", choices=["stop", "target"], default="stop",
                   help="which side to assume when one bar spans both SL and TP")

    # symbol spec overrides when the CSV has no '#' header
    p.add_argument("--point", type=float, default=None)
    p.add_argument("--tick-value", type=float, default=None)
    p.add_argument("--tick-size", type=float, default=None)
    p.add_argument("--volume-min", type=float, default=None)
    p.add_argument("--volume-step", type=float, default=None)
    p.add_argument("--volume-max", type=float, default=None)

    # EA inputs, same names and defaults as NAS100_ML_Bot.mq5
    p.add_argument("--lr", type=float, default=0.010)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--rff-dim", type=int, default=64)
    p.add_argument("--rff-sigma", type=float, default=8.0)
    p.add_argument("--mlp-hidden", type=int, default=24)
    p.add_argument("--hedge-eta", type=float, default=0.35)
    p.add_argument("--replay-steps", type=int, default=4)
    p.add_argument("--replay-capacity", type=int, default=4000)
    p.add_argument("--min-train-samples", type=int, default=400)
    p.add_argument("--eval-window", type=int, default=300)
    p.add_argument("--seed", type=int, default=20240517)
    p.add_argument("--n-rff", type=int, default=1, help="parallel RFF experts (different draws)")
    p.add_argument("--n-mlp", type=int, default=1, help="parallel MLP experts (different inits)")
    p.add_argument("--label-stride", type=int, default=1,
                   help="score only every Nth matured sample, to de-overlap the metric")
    p.add_argument("--overlap-weight", type=float, default=1.0,
                   help="scale training weight to offset overlapping samples")
    p.add_argument("--cost-aware", type=int, default=0,
                   help="raise the entry threshold until expected R clears the spread")
    p.add_argument("--min-expected-r", type=float, default=0.02,
                   help="required expected R above cost when --cost-aware=1")
    p.add_argument("--label-horizon", type=int, default=12)
    p.add_argument("--barrier-atr", type=float, default=1.20)
    p.add_argument("--time-barrier-weight", type=float, default=0.50)
    p.add_argument("--prob-threshold", type=float, default=0.58)
    p.add_argument("--min-roll-accuracy", type=float, default=0.52)
    p.add_argument("--max-disagreement", type=float, default=0.45)
    p.add_argument("--allow-long", type=int, default=1)
    p.add_argument("--allow-short", type=int, default=1)
    p.add_argument("--risk-percent", type=float, default=0.50)
    p.add_argument("--max-daily-loss-pct", type=float, default=3.0)
    p.add_argument("--max-drawdown-pct", type=float, default=15.0)
    p.add_argument("--max-spread-points", type=float, default=400.0)
    p.add_argument("--max-spread-atr-frac", type=float, default=0.15)
    p.add_argument("--max-trades-per-day", type=int, default=8)
    p.add_argument("--loss-streak-trigger", type=int, default=3)
    p.add_argument("--cooldown-bars", type=int, default=8)
    p.add_argument("--max-lots", type=float, default=5.0)
    p.add_argument("--stop-atr", type=float, default=1.50)
    p.add_argument("--take-profit-r", type=float, default=1.60)
    p.add_argument("--breakeven-r", type=float, default=0.90)
    p.add_argument("--breakeven-offset-r", type=float, default=0.10)
    p.add_argument("--trail-atr", type=float, default=2.00)
    p.add_argument("--trail-start-r", type=float, default=1.30)
    p.add_argument("--max-hold-bars", type=int, default=36)
    p.add_argument("--use-session", type=int, default=1)
    p.add_argument("--trade-start", type=int, default=9)
    p.add_argument("--trade-end", type=int, default=22)
    p.add_argument("--london-start", type=int, default=8)
    p.add_argument("--london-end", type=int, default=17)
    p.add_argument("--ny-start", type=int, default=14)
    p.add_argument("--ny-end", type=int, default=23)

    cfg = p.parse_args()

    df, spec = load_bars(cfg.bars, {
        "point": cfg.point, "tick_value": cfg.tick_value, "tick_size": cfg.tick_size,
        "volume_min": cfg.volume_min, "volume_step": cfg.volume_step,
        "volume_max": cfg.volume_max,
    })

    if cfg.dt_from:
        df = df[df.index >= pd.Timestamp(cfg.dt_from)]
    if cfg.dt_to:
        df = df[df.index < pd.Timestamp(cfg.dt_to)]
    if len(df) < 600:
        sys.exit(f"only {len(df)} bars in range - need at least ~600 to clear "
                 f"indicator warm-up plus the {cfg.min_train_samples}-sample model warm-up")

    sessions = {"london_start": cfg.london_start, "london_end": cfg.london_end,
                "ny_start": cfg.ny_start, "ny_end": cfg.ny_end}

    print(f"building features for {len(df):,} bars ...", file=sys.stderr)
    F = build_features(df, sessions)
    df = df.assign(atr=atr(df, 14))

    print("replaying ...", file=sys.stderr)
    bt = Backtest(df, F, cfg, spec).run()
    report(bt, cfg, df)


if __name__ == "__main__":
    main()

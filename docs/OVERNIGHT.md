# NAS100 Overnight

Long the index at a fixed evening time, flat before the cash open. Nothing is
predicted, nothing is learned, and there are three parameters that matter.

This exists because the ML bot in this repo does not work — `tools/edge_test.py`
showed its 40 features carry no signal for the triple-barrier target. Rather
than keep tuning a model with no edge, I went looking for structure in the data
directly. This is what was there.

---

## The evidence

Feed: `NAS100.s` M15, 23,684 bars, 2025-08-21 → 2026-08-21 (GMT+3 server).

Decompose the year by clock:

```
index moved            +6,120 pts over the year (+26.3%)
overnight (23:00→04:00) +4,504 pts across 200 nights
intraday  (04:00→23:00)   -363 pts across 249 days
```

**74% of the year's gain arrived while the US cash market was shut, and the
intraday session was net negative.** That is the well-documented overnight
drift in equity indices, not something mined out of this particular file — it
has a large published literature, which is why I trusted it enough to build on
one year of data.

Raw edge before costs: **+20.2 points per night**, t = +2.55 over 200 nights,
positive in both halves of the year (H1 +13.6, H2 +26.8). Against a 2.3-point
spread that is a workable cost-to-edge ratio — the opposite of the ML
strategy, where the edge was smaller than the spread.

Robustness checks it survived:

| Check | Result |
|---|---|
| Both halves of the year positive | yes (+13.6 / +26.8 pts) |
| Months positive | 10 of 13 |
| Remove the best month | still +1,684 pts |
| Remove the best two months | still +1,085 pts |

---

## What it returns, and what decides that

Simulated with `tools/backtest_overnight.py` — real per-bar spread from the
feed, intrabar stop checking, risk-based sizing, 3×ATR protective stop:

| Risk per night | Return | Max DD | Win rate | PF |
|---|---|---|---|---|
| 0.5% | +4.11% | 2.69% | 57.5% | 1.21 |
| 1.0% | +8.34% | 5.38% | 57.5% | 1.20 |
| 2.0% | +16.46% | 10.59% | 57.5% | 1.19 |

200 nights traded. Buy and hold over the same window was +26.3%, but with 24h
exposure and a far deeper drawdown; this holds a position ~5 hours a day.

### Swap is the whole game

The position is deliberately held across the 00:00 rollover, so you pay
financing **every single night**. At 0.5% risk:

```
swap  0 pts/night  ->  +8.04% per year
swap  2 pts/night  ->  +5.64%
swap  4 pts/night  ->  +3.45%
swap  8 pts/night  ->  -1.01%     <-- dead
```

The gross edge is ~13.8 points a night. Financing of 8 points eats 58% of it
and the strategy stops working.

**Before running this, read your symbol's long swap** — right-click the symbol
in Market Watch → Specification → *Swap long* — and convert it to index points.
If it costs more than roughly 6 points a night, close this document. The EA
prints the broker's swap on its chart panel for exactly this reason.

A raw/ECN account, or a broker that charges index financing as a small daily
percentage rather than a fat markup, is the difference between this working
and not.

---

## Install

1. Copy `MQL5/Experts/NAS100_Overnight.mq5` and the `MQL5/Include/NAS100ML/`
   folder into your terminal's data folder, and compile.
2. Attach to a NAS100 M15 chart.
3. **Set the session hours for your broker's server time.** The defaults
   (23:00 → 04:00) are for a GMT+3 server. Derive yours: bar range and tick
   volume roughly double at the New York cash open (09:30 ET); that hour is
   your `InpExitHour` minus a few hours. On the measured feed the NY open
   landed at server hour 16, so the overnight window runs 23:00 → 04:00.

Then verify on your own exported bars before risking anything:

```bash
python3 tools/backtest_overnight.py --bars NAS100.s_M15.csv \
    --swap-points <YOUR SWAP IN INDEX POINTS> --risk-percent 0.5
```

---

## Parameters

| Input | Default | Notes |
|---|---|---|
| `InpEntryHour` / `InpEntryMinute` | 23:00 | Server time. Broker-specific |
| `InpExitHour` / `InpExitMinute` | 04:00 | Next day, before the cash open |
| `InpMaxHoldHours` | 12 | Force flat if the exit bar never arrives |
| `InpSkipFriday` | true | Friday night has no exit before the weekend close |
| `InpSkipMonday` | false | Monday was the one negative weekday (−8.64/night over 48 nights). **Not** enabled by default: 48 observations across 4 weekday buckets is not enough to act on, and switching it on is curve-fitting unless your own data agrees |
| `InpRiskPercent` | 0.50 | Per night, as % of equity |
| `InpStopATR` | 3.00 | Protective stop. `0` disables it — which backtested *better* (+9.4% vs +4.1%) but leaves you exposed to an uncapped gap. The default keeps the tail bounded |
| `InpMaxDrawdownPct` | 25.0 | Kill switch. Does not re-arm |
| `InpMaxSpreadPts` | 400 | Skip the night if the spread is wider than this at entry |

---

## What this does not establish

* **One year, one broker feed, one instrument.** The overnight effect has broad
  published support, but this specific measurement does not.
* **It was a bull year** (+26.3%). The overnight leg was positive while the
  intraday leg was negative, which is the anomaly's signature rather than plain
  long bias — but a sustained bear market would still hurt, and this data
  cannot tell you how much. It is a long-only, directional strategy.
* **Bar replay, not ticks.** Entries fill at the bar open exactly; stops fill
  at their level. Real fills will be worse, especially at 23:00 when liquidity
  is thinning.
* **Gap risk is real.** The stop cannot protect against a weekend or headline
  gap through it. Worst night in the sample was −339 points.
* **The weekday breakdown is noise-prone.** Tue +18.18, Mon −8.64 per night
  looks like a pattern; with ~50 nights per bucket it mostly isn't. Do not
  build filters on it without much more data.

Trade it on demo for a few months and compare the realised per-night points
against the +13.8 gross the backtest expects. If your fills and financing land
far off that, the edge is gone before it reaches you.

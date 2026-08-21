# Backtesting

There are two routes. Use both, in this order.

| | `tools/backtest.py` | MT5 Strategy Tester |
|---|---|---|
| Speed | ~40 s for a year of M15 | minutes to hours |
| Data | OHLC bars | real ticks |
| Execution | modelled | broker-accurate |
| Purpose | is there an edge at all? | what would it actually have made? |

The Python replica is for answering *"is there any signal here"* in under a
minute. The Strategy Tester is the number you trust.

---

## Route 1: the Python replica (fast)

`tools/backtest.py` reproduces the EA's pipeline exactly — the same 40
features, the same triple-barrier labels, the same three online experts with
the same AdamW settings, the same Hedge blending, the same entry gates, the
same sizing and exits. It replays bars instead of ticks.

### Export the bars

1. Copy `MQL5/Scripts/ExportBars.mq5` into `<data folder>/MQL5/Scripts/` and
   compile it.
2. Open a NAS100 chart on the timeframe you want (M15), scroll back far enough
   that the terminal has downloaded the history.
3. Drag the script onto the chart. Set `InpBars` — a year of M15 on a 24/5 CFD
   is roughly 25,000 bars; 30,000 gives you margin.
4. The journal prints the output path, normally
   `<data folder>/MQL5/Files/NAS100ML/bars.csv`.

The CSV's first line records the symbol's `tick_value`, `tick_size`,
`point` and volume limits, so position sizing in the replay uses your broker's
real contract spec rather than a guess. The per-bar `spread` column is the
spread the terminal actually recorded, so trades are charged the real
historical spread.

### Run it

```bash
pip install numpy pandas

python3 tools/backtest.py --bars bars.csv --deposit 10000 --commission 4.0

# a specific window
python3 tools/backtest.py --bars bars.csv --from 2024-08-01 --to 2025-08-01

# every EA input is exposed with the same default
python3 tools/backtest.py --bars bars.csv --risk-percent 0.25 --prob-threshold 0.62
```

Set `--commission` to what your broker actually charges per lot per round
turn. Leaving it at zero will flatter the result.

### Read the output in this order

```
walk-forward accuracy : 0.6033  (last 300) over 24,791 scored samples
majority baseline     : 0.6367   ->  EDGE -0.0333   <-- no edge; ...
```

**The EDGE line first, before anything else.** Accuracy on its own is
meaningless — a model that has learned nothing but "the up barrier gets hit
more often" will happily report 0.60. If EDGE is not clearly positive, the
return figure below it is noise and no amount of parameter tuning will fix it.

Then, in order: number of trades (under ~100 tells you nothing), max drawdown,
profit factor, and only then the return.

The report also prints why the EA stood aside, bar by bar:

```
bars skipped by gate:
  max drawdown               16,269
  no edge                     4,722
  expert disagreement           814
```

A huge `max drawdown` count means the kill switch fired early and the rest of
the run was flat — the headline return is then a partial-year number, not a
full one. A huge `warm-up` count means you did not give it enough history.

### What the replica cannot model

* **Intrabar sequencing.** When one bar spans both the stop and the target,
  the replay has to guess. It assumes the stop went first (`--ambiguous-bar
  stop`); `--ambiguous-bar target` shows you the optimistic bound. The truth
  is in between, and only tick data resolves it.
* **Slippage and requotes.** Entries fill at the next bar's open exactly.
* **Variable spread within a bar.** One spread value per bar.
* **Swap** on positions held overnight.
* **Gaps against a stop.** The replay fills stops at their exact level; in
  reality a weekend gap fills wherever the market reopens.

All of these push the replica's result *optimistic* relative to live, except
the ambiguous-bar rule, which pushes it pessimistic. Treat the replica as a
screening tool, not a P&L forecast.

---

## Route 2: the MT5 Strategy Tester (authoritative)

Settings that matter:

* **Model:** *Every tick based on real ticks*. Nothing less models the
  intrabar stop/target sequencing the triple-barrier label depends on.
* **Period:** at least 2 years. The EA spends its first ~400 resolved samples
  learning without trading, so a 3-month test measures almost nothing.
* **Commission:** set it in the tester's symbol settings.
* **Deposit and leverage:** whatever you will actually trade.

At shutdown the EA prints its walk-forward accuracy to the journal. Read that
before the equity curve, for the same reason as above.

`OnTester()` returns a custom optimisation score — net profit per unit of
drawdown, times capped profit factor, discounted when the sample is thin.
Select *Custom max* to use it. It scores runs with fewer than 40 trades as
zero.

### Do not mass-optimise

The EA already adapts online. Grid-searching fifteen inputs over one history
produces a beautiful curve that means nothing. If you optimise anything,
optimise the parameters that change the *problem* rather than the fit:

* `InpLabelHorizon` and `InpBarrierATR` — these define what is being
  predicted. The barrier should be reachable within the horizon on a typical
  bar.
* `InpProbThreshold` — the trade / no-trade cut.

Then verify on a period you did not touch.

---

## Keeping the two in agreement

The replica and the EA are separate implementations of the same design, so
they can drift apart. If you change a feature, a label rule or an exit rule in
one, change it in the other:

| MQL5 | Python |
|---|---|
| `FeatureEngine.mqh` | `build_features()` in `tools/backtest.py` |
| `Labeler.mqh` | `Labeler` |
| `Models.mqh` | `Logistic` / `Rff` / `Mlp` in `tools/train_offline.py` |
| `Ensemble.mqh` | `Ensemble` |
| `RiskManager.mqh` + `TradeExecutor.mqh` | `Backtest._lots` / `_manage` / `run` |

Expect small residual differences even when they agree: MT5's indicator
smoothing conventions (Wilder for ATR/RSI/ADX, SMA for the MACD signal line)
are matched in the replica, but seeding at the start of the series and
floating-point ordering are not bit-identical. Directionally the two should
tell the same story; if they do not, one of them has a bug.

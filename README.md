# Trading Bots

Expert Advisors, and the tooling that measured them.

| Bot | Verdict |
|---|---|
| **[NAS100_Overnight](docs/OVERNIGHT.md)** — long overnight, flat by the cash open | **works, swap permitting**: +4.1% at 0.5% risk / night, 2.7% max drawdown, PF 1.21 on a year of real NAS100.s M15 |
| **NAS100_ML_Bot** — online-learning ensemble (below) | **does not work**: −32% over the year; its features carry no measurable signal |
| **[XAUUSD_Scalper](docs/XAUUSD_SCALPER.md)** — gold, spot on weekdays and the 24/7 book at weekends | **unmeasured**: no XAUUSD feed has been replayed through it. Built to be tested; `tools/backtest_scalper.py` is how you test it |

Read `docs/OVERNIGHT.md` first. The ML bot is documented below and kept because
the research tooling around it — `tools/edge_test.py` in particular — is what
established that it does not work, and what found the strategy that does.

---

# XAUUSD Scalper

Gold scalper that follows the market across the weekend: **XAUUSD.s** Monday to
Friday, **XAUUSD24/7.s** while spot is shut, flat across every handover. The
weekend is detected from the broker's own session table, so DST and holidays
need no re-tuning — and the nightly maintenance break is deliberately not
mistaken for one.

Two mechanisms — a Donchian breakout continuation and a Bollinger fade — both
ATR-normalised, both gated by a spread cap expressed as *a fraction of the stop
distance*, because that is what decides whether gold scalping clears its costs:

| Timeframe | typical ATR | spread 20 pts | spread 30 pts | spread 60 pts |
|---|---|---|---|---|
| M5  | ~$2.20 | 0.091 R | 0.136 R | 0.272 R |
| M15 | ~$3.80 | 0.052 R | 0.079 R | 0.157 R |

On M5 at 30 points you start every trade 0.14 R behind. That is the number to
beat before anything else matters.

**It has not been shown to be profitable.** No gold feed has been run through
it — that is your data to supply:

```bash
python3 tools/backtest_scalper.py --bars XAUUSD.s_M5.csv \
                                  --weekend-bars XAUUSD24-7.s_M5.csv --commission 7.0
python3 tools/backtest_scalper.py --demo     # no data? see what no edge looks like
```

The report's headline is the edge over a **random-direction control** — the same
entries, sizes, exits and spread, with only the direction call replaced by a coin
flip, so everything the strategy does not claim cancels out:

```
EDGE over the control : +0.0412 R   (SE 0.0290)   t = 1.42
```

The EA holds one position at a time, so these trades do not overlap and that t
statistic is honest — unlike the walk-forward accuracy that flattered the ML bot
below. Require t ≥ 2, both halves positive and every fold positive before
trading it. Full detail, including two measurements that set the defaults
(break-even and trailing stops cost ~0.05 R per trade on a driftless series, so
both ship disabled), is in **[docs/XAUUSD_SCALPER.md](docs/XAUUSD_SCALPER.md)**.

---

# NAS100 ML Bot

A machine-learning Expert Advisor for the NASDAQ 100 CFD (`NAS100`, `US100`,
`USTEC`, `NDX100` — whatever your broker calls it), written in pure MQL5.

The EA learns **online, on the instrument it trades**. There is no Python
runtime, no DLL, no ONNX file and no external service in the trading loop —
everything from feature extraction to gradient descent runs inside MetaTrader,
which means it works identically in the Strategy Tester, on demo and on live.

---

## What it actually does

Once per closed bar:

```
bar closes
  ├─ resolve the triple-barrier labels that matured on this bar
  ├─ score each matured sample out-of-sample  ← walk-forward accuracy
  ├─ re-weight the three experts (Hedge / exponential weights)
  ├─ take a gradient step on the new sample + a few replayed ones
  ├─ extract this bar's 40 features, open a new pending sample
  └─ predict P(up barrier hit first); if every gate agrees, send one order
```

### The prediction target

Not "will the next close be higher" — that target is mostly noise and it does
not match how a trade actually resolves. Instead the EA uses **triple-barrier
labelling**: from each bar's close, does price touch `+1.2 × ATR` before it
touches `−1.2 × ATR`, within 12 bars?

* upper barrier first → label 1, weight 1.0
* lower barrier first → label 0, weight 1.0
* neither, time runs out → label from the terminal close, weight 0.5

That is the same question the entry logic has to answer, so the model is
trained on the decision it is asked to make.

### The three experts

| Expert | What it captures | Trained parameters |
|---|---|---|
| **Logistic regression** | linear, monotone effects — the robust backbone | `n + 1` |
| **Random Fourier features** | smooth non-linearity, an O(D) streaming approximation of an RBF-kernel classifier | `D + 1` (projection frozen) |
| **MLP** (1 hidden tanh layer) | feature interactions the other two cannot express | `H·n + 2H + 1` |

All three are trained with **AdamW** (decoupled weight decay) plus gradient
clipping, on standardised inputs.

### Blending them

Each expert carries a multiplicative weight updated by the exponentially
weighted average forecaster rule:

```
w_i  ←  w_i · exp( −η · logloss_i )
```

Whichever model currently describes the market best dominates the blend, and a
floor of 0.05 on every weight lets a model that fell out of favour climb back
when the regime turns. The live weights are visible on the chart panel.

### Why it does not overfit itself into a corner

* **Causal by construction.** Features come from closed bars only; the
  standardiser only ever folds in samples whose label has already matured.
* **Honest scoring.** Every new sample is graded *before* it is learned from,
  so the rolling accuracy on the panel is genuinely out-of-sample in time —
  and the EA refuses to trade when it drops below `InpMinRollAccuracy`.
  **But read the caveat below**: because a sample is opened on every bar and
  each label takes `InpLabelHorizon` bars to resolve, consecutive samples
  overlap heavily and that accuracy overstates the edge a sequence of
  non-overlapping trades actually realises. On a year of real NAS100 M15 the
  gap was 0.63 label accuracy versus a ~48% realised win rate — see the
  measured results below, which found no tradeable edge at all.
* **Class-balanced replay.** A bounded replay buffer gives several effective
  epochs over a rolling window while half the draws target the minority class,
  so a long trend cannot bake in a permanent directional prior.
* **Disagreement gate.** When the three experts spread further apart than
  `InpMaxDisagreement`, the signal is discarded rather than traded.

---

## Risk layer

The model only ever proposes a direction. Whether the trade happens, and how
big it is, is decided by `CRiskManager`:

* fixed-fractional sizing from the ATR stop distance and the broker's real
  tick value — never a hard-coded lot size
* size scaled by how far the blended probability sits from a coin flip,
  bounded to 0.6×–1.4× so one confident signal cannot double the risk
* daily loss stop, account-level drawdown kill switch, per-day trade cap
* loss-streak cooldown
* spread filter — both an absolute cap and a cap relative to ATR, which is
  what actually matters on an index CFD
* session window filter (broker server hours)
* margin pre-check before every order

Exits: ATR stop, R-multiple target, break-even lift, ATR trailing stop, and a
bar-count time stop.

---

## Repository layout

```
MQL5/
  Experts/
    NAS100_Overnight.mq5       the working bot - clock-driven, no ML
    NAS100_ML_Bot.mq5          the ML EA - inputs, per-bar pipeline, panel
    XAUUSD_Scalper.mq5         gold scalper, spot + the 24/7 weekend book
  Include/XAUScalp/
    SymbolRouter.mqh           which gold symbol is live, and for how long
    SignalEngine.mqh           breakout and fade, from closed bars only
    ScalpRisk.mqh              sizing and every pre-trade gate
  Include/NAS100ML/
    Utils.mqh                  math helpers, AdamW, ring statistics
    FeatureEngine.mqh          40 ATR-normalised features from closed bars
    Scaler.mqh                 streaming z-score (Welford)
    Models.mqh                 logistic / RFF / MLP experts
    Ensemble.mqh               Hedge weighting + walk-forward scoring
    Labeler.mqh                triple-barrier labelling queue
    ReplayBuffer.mqh           bounded class-balanced experience replay
    RiskManager.mqh            sizing and every pre-trade gate
    TradeExecutor.mqh          orders, break-even, ATR trailing, deal polling
    Persistence.mqh            checkpoint I/O + CSV dataset export
  Scripts/
    ExportBars.mq5             dump chart bars to CSV for the replica
tools/
  backtest_scalper.py          backtest for the gold scalper (no dependencies)
  backtest_overnight.py        backtest for the overnight bot
  backtest.py                  bar-replay backtest - faithful ML replica
  train_offline.py             optional numpy pre-trainer (same layouts)
  verify_checkpoint.py         validate a checkpoint before loading it
  edge_test.py                 is there any signal? non-overlapping OOS test
docs/
  XAUUSD_SCALPER.md            the gold scalper: weekend switch, costs, measuring
  OVERNIGHT.md                 the working bot: evidence, swap, limits
  INSTALL.md                   install, sessions, parameters, troubleshooting
  BACKTESTING.md               both backtest routes and how to read them
```

---

## Quick start

1. Copy `MQL5/Experts/NAS100_ML_Bot.mq5` and the whole `MQL5/Include/NAS100ML/`
   folder into your terminal's data folder (File → Open Data Folder).
2. Compile `NAS100_ML_Bot.mq5` in MetaEditor (F7).
3. Backtest on your broker's NAS100 symbol, M15, **"Every tick based on real
   ticks"**, over at least 2 years. For a first answer in under a minute,
   export bars with `MQL5/Scripts/ExportBars.mq5` and run
   `python3 tools/backtest.py --bars bars.csv` instead — see
   `docs/BACKTESTING.md`.
4. Read `docs/INSTALL.md` before doing anything with real money — in
   particular the section on setting the session hours for *your* broker's
   server time, which the defaults will almost certainly get wrong.

The first `InpMinTrainSamples` (400) resolved samples are a warm-up: the EA
watches and learns but does not trade. On M15 that is roughly three weeks of
bars.

---

## Getting a number

```bash
# in MT5: run Scripts/ExportBars.mq5 on a NAS100 M15 chart
python3 tools/backtest.py --bars bars.csv --deposit 10000 --commission 4.0
```

`tools/backtest.py` replays the exact same pipeline on OHLC bars and reports
return, drawdown, trade statistics and — the line that matters most — the
ensemble's walk-forward accuracy **against the majority-class baseline**:

```
walk-forward accuracy : 0.6033  (last 300) over 24,791 scored samples
majority baseline     : 0.6367   ->  EDGE -0.0333   <-- no edge; ...
```

Accuracy on its own is a trap: a model that has learned nothing but the base
rate will report 0.60 and lose money. If EDGE is not clearly positive, the
return underneath it is noise. `docs/BACKTESTING.md` covers both routes and
what the bar replay cannot model.

## The offline trainer is optional

Run the EA with `InpExportDataset = true` to write one CSV row per resolved
sample, then:

```bash
python3 tools/train_offline.py --data dataset.csv --out model.txt
python3 tools/verify_checkpoint.py model.txt
```

The script mirrors the EA's parameter layouts exactly and reports
walk-forward accuracy against the majority-class baseline. **If it does not
beat that baseline out of sample, the feature set has no edge on your data —
do not trade the checkpoint.** That check is the most valuable thing in this
repository.

---

## Measured result on a year of real NAS100 M15

Broker feed `NAS100.s`, M15, 23,684 bars, 2025-08-21 → 2026-08-21, replayed
through `tools/backtest.py` with the terminal's own recorded spread
(median 2.3 index points) and no commission.

**On this evidence the strategy has no tradeable edge.** Not "it would work
with a tighter spread" — the expectancy is negative at *zero* spread too.

| Configuration | Return | Trades | Win rate | Avg trade |
|---|---|---|---|---|
| Shipped defaults | −8.3% | 189 | 42.9% | −0.084 R |
| Drawdown kill switch disabled | −32.1% | 1,498 | 44.4% | −0.056 R |
| Geometry matched to label, spread = real | −29.6% | 1,883 | 48.4% | −0.058 R |
| **Geometry matched, spread = ZERO** | **−29.3%** | 1,883 | 48.4% | **−0.032 R** |
| H1, 1.5 ATR barriers, real spread | −10.7% | 527 | 48.6% | −0.040 R |
| H1, 1.5 ATR barriers, spread ≈ 0 | −8.3% | 527 | 48.6% | −0.031 R |

Buy and hold over the same window: **+26.3%**.

Split the year in half and both halves lose at zero spread (−0.036 R then
−0.022 R per trade), so this is not one bad regime.

### What the numbers mean

1. **Cost is real but not decisive.** At a 1.2 ATR stop on M15 the 2.3-point
   spread costs ~5% of every R. Removing it entirely moves expectancy from
   −0.058 R to −0.032 R — a large improvement that still leaves it negative.
   Moving to H1, where the same spread is a much smaller fraction of the stop,
   lands in the same place. Cost is not what is standing between this and
   profitability.
2. **The label accuracy is an artifact.** The ensemble reports 0.63 walk-forward
   accuracy and 64.5% directional accuracy in the trade zone, and it is
   genuinely well calibrated (p > 0.62 → 69% realised, monotone across every
   bucket). But a sample opens on every bar and takes `InpLabelHorizon` bars
   to resolve, so consecutive labels share nearly all of their outcome. The
   effective independent sample count is roughly 1/12 of nominal, and the
   accuracy that survives into a sequence of *non-overlapping* trades is
   ~48–51% — a coin flip. This is the classic overlapping-labels trap, and
   this EA walks straight into it.
3. **Expectancy at zero cost hovers around −0.03 R and flips sign with
   configuration.** One earlier parameter combination (gates off, 12-bar time
   stop) produced +0.024 R at zero spread; turning the normal gates back on
   returns it to −0.032 R. A sign that moves with unrelated settings is noise,
   not signal.

### Three months is too short to measure anything here

Asked for the most recent quarter (2026-05-21 → 2026-08-21), the answer
depends entirely on which question is meant:

| Question | Result |
|---|---|
| "I've had it running since last August" | **0.00%** — zero trades; the kill switch halted it in October 2025 |
| Same, kill switch disabled | **−12.07%** over 382 trades |
| "I installed it three months ago" (cold start) | **+4.72%** over 338 trades |

The warmed-up and cold-start runs cover the *identical* three months with
identical parameters. They differ by 17 percentage points, and their long/short
splits nearly invert (155/227 versus 214/124), purely because the online model
entered the window in a different state.

Worse, that +4.72% is not reproducible. Changing only the RNG seed — which
sets the random Fourier projection and the MLP's initial weights, and has
nothing to do with the market — gives:

```
seed 20240517  +4.72%     seed 1  −1.77%     seed 2  +10.12%
seed 3         +8.74%     seed 4  −2.11%     seed 5   +0.20%
```

Mean +3.3%, spread 12.2 points, straddling zero. A quarter contains far too
few independent trades for the result to mean anything.

The full year does survive this check — every seed is negative on both return
and per-trade expectancy (−12.8% to −32.9%, −0.021 R to −0.056 R) — which is
why the negative conclusion above stands while any short-window number, good
or bad, should be ignored.

Use `--report-from` to score a tail window with the model already warmed up,
rather than `--from`, which restarts it cold:

```bash
python3 tools/backtest.py --bars NAS100.s_M15.csv --report-from 2026-05-21
```

### The features carry no measurable signal

`tools/edge_test.py` settles the question the backtest cannot. It keeps only
**non-overlapping** samples (one per label horizon), splits them
chronologically 70/30, trains on the first part and scores the second:

```
horizon/barrier     n_indep  baseline  logit OOS   mlp OOS  best edge
12b / 1.2ATR           1951    0.5171     0.5188    0.5051    +0.0017   (SE 0.0207)
24b / 1.5ATR            975    0.5051     0.4437    0.5358    +0.0307   (SE 0.0292)
24b / 2.5ATR            975    0.5461     0.4881    0.5119    -0.0341   (SE 0.0292)
48b / 2.5ATR            487    0.5238     0.4558    0.5646    +0.0408   (SE 0.0412)
48b / 3.5ATR            487    0.5170     0.4898    0.5034    -0.0136   (SE 0.0412)
96b / 4.0ATR            243    0.5205     0.5205    0.5342    +0.0137   (SE 0.0585)
```

Every edge is inside one standard error of zero, and a third are negative.
That is what no signal looks like. It also explains every result above: the
0.63 walk-forward accuracy, the −32% year, the seed lottery over three months.
There is nothing for the risk layer, the ensemble or the exits to convert.

**Run this before tuning anything.** No parameter search on the backtest can
manufacture an edge that is not in this table.

### What this does not establish

One year, one broker feed, one instrument, replayed on bars rather than ticks.
It does not prove the feature set is worthless everywhere — it does establish
that these features, this label and this timeframe do not clear costs on this
data, and that the EA's own accuracy readout cannot be trusted to tell you
otherwise.

Reproduce it:

```bash
python3 tools/backtest.py --bars NAS100.s_M15.csv --deposit 10000 \
  --ny-start 16 --ny-end 23 --london-start 10 --london-end 18 \
  --trade-start 10 --trade-end 23
```

Session hours above are for a GMT+3 broker, derived from that feed: bar range
and tick volume both roughly double at server hour 16, which is the New York
cash open. Derive yours the same way before trusting any result.

---

## Honest limitations

Read these before you decide what this is worth.

* **No edge is promised.** This is a well-built learning and risk framework.
  Whether these 40 features carry predictive information on your broker's
  NAS100 feed, after your spread and commission, is an empirical question that
  only your own walk-forward testing can answer.
* **Spread and commission dominate.** At `1.2 × ATR` barriers on M15 the
  average trade is small enough that a wide index spread can eat the entire
  edge. Backtest with real tick data and your real commission.
* **Online learning drifts.** The model adapts to the current regime, which
  cuts both ways: it recovers from regime change, and it can also chase noise.
  The accuracy floor and the drawdown kill switch exist for exactly that.
* **Overnight gaps.** Index CFDs gap. Stops are not guaranteed; the position
  sizing assumes the stop fills near its level, and sometimes it will not.
* **Backtest ≠ live.** Slippage, requotes, swap and variable spread are
  modelled optimistically by the tester.

Trade it on demo until you have seen it handle a full drawdown cycle.

---

## Licence

Provided as-is, for research and educational use. Trading leveraged CFDs
carries substantial risk of loss. You are responsible for anything this
software does with your money.

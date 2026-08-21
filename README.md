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
  so the rolling accuracy on the panel is a genuine walk-forward number — and
  the EA refuses to trade when it drops below `InpMinRollAccuracy`.
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
    NAS100_ML_Bot.mq5          the EA - inputs, per-bar pipeline, panel
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
tools/
  train_offline.py             optional numpy pre-trainer (same layouts)
  verify_checkpoint.py         validate a checkpoint before loading it
docs/
  INSTALL.md                   install, backtest, tune, troubleshoot
```

---

## Quick start

1. Copy `MQL5/Experts/NAS100_ML_Bot.mq5` and the whole `MQL5/Include/NAS100ML/`
   folder into your terminal's data folder (File → Open Data Folder).
2. Compile `NAS100_ML_Bot.mq5` in MetaEditor (F7).
3. Backtest on your broker's NAS100 symbol, M15, **"Every tick based on real
   ticks"**, over at least 2 years.
4. Read `docs/INSTALL.md` before doing anything with real money — in
   particular the section on setting the session hours for *your* broker's
   server time, which the defaults will almost certainly get wrong.

The first `InpMinTrainSamples` (400) resolved samples are a warm-up: the EA
watches and learns but does not trade. On M15 that is roughly three weeks of
bars.

---

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

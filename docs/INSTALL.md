# Install, backtest and tune

## 1. Install

1. In MetaTrader 5: **File → Open Data Folder**. You land in something like
   `C:\Users\<you>\AppData\Roaming\MetaQuotes\Terminal\<hash>\`.
2. Copy files so the tree looks like this:

```
<data folder>/MQL5/Experts/NAS100_ML_Bot.mq5
<data folder>/MQL5/Include/NAS100ML/Utils.mqh
<data folder>/MQL5/Include/NAS100ML/FeatureEngine.mqh
<data folder>/MQL5/Include/NAS100ML/Scaler.mqh
<data folder>/MQL5/Include/NAS100ML/Models.mqh
<data folder>/MQL5/Include/NAS100ML/Ensemble.mqh
<data folder>/MQL5/Include/NAS100ML/Labeler.mqh
<data folder>/MQL5/Include/NAS100ML/ReplayBuffer.mqh
<data folder>/MQL5/Include/NAS100ML/RiskManager.mqh
<data folder>/MQL5/Include/NAS100ML/TradeExecutor.mqh
<data folder>/MQL5/Include/NAS100ML/Persistence.mqh
```

   The include path matters — the EA uses `#include <NAS100ML/...>`, which
   resolves against `MQL5/Include/`.

3. Open `NAS100_ML_Bot.mq5` in MetaEditor and compile (**F7**). You should get
   0 errors. Warnings about unused parameters in the model base class are
   harmless.
4. Drag the EA onto a NAS100 M15 chart, tick **Allow Algo Trading**.

The EA writes checkpoints and dataset exports under
`<data folder>/MQL5/Files/NAS100ML/`. It creates the subfolder on first write;
if your terminal blocks that, change `InpCheckpointFile` to a bare filename
like `nas100ml_model.txt`.

**In the Strategy Tester those files land somewhere else** — each tester agent
has its own sandbox, typically
`<data folder>/Tester/<agent-name>/MQL5/Files/NAS100ML/`. That is where the
exported `dataset.csv` will be after a backtest, and it is the most common
reason people think the export "did not work".

---

## 2. Set the session hours — do this first

Every session input is in **broker server time**, not your local time and not
your exchange's time. Server time varies by broker and shifts with DST.

Find your offset: open the Market Watch, look at the server clock, and compare
it to the actual New York cash open (09:30 America/New_York).

| Broker server time | `InpNYStart` | `InpTradeStartHour` | `InpTradeEndHour` |
|---|---|---|---|
| GMT+2 (winter) / GMT+3 (summer) — most common | `15` | `9` | `22` |
| GMT+0 | `13` | `7` | `20` |
| GMT+3 fixed | `16` | `10` | `23` |

`InpNYStart` / `InpNYEnd` / `InpLondonStart` / `InpLondonEnd` feed the session
*features* — they tell the model which part of the day it is looking at.
`InpTradeStartHour` / `InpTradeEndHour` gate *trading*. Getting the first pair
wrong quietly degrades the model; getting the second pair wrong means trading
the thin overnight session, where the NAS100 spread is at its worst.

---

## 3. Backtest properly

Settings that matter:

* **Model:** *Every tick based on real ticks*. Anything less will not model
  the intrabar stop/target sequencing that the triple-barrier label depends on.
* **Period:** at least 2 years. The EA spends the first ~400 resolved samples
  learning without trading, so a 3-month test measures almost nothing.
* **Deposit / leverage:** whatever you will actually trade.
* **Commission:** set it in the tester's symbol settings. Index CFD commission
  is frequently the difference between a profitable and an unprofitable curve
  at this holding period.

What to look at, in order:

1. **Walk-forward accuracy** in the journal at shutdown. Below ~0.52 there is
   no signal and the rest of the report is noise.
2. **Number of trades.** Under 100 the equity curve tells you nothing.
3. **Drawdown**, then profit factor, then net profit. In that order.

`OnTester()` returns a custom optimisation score — net profit per unit of
drawdown, multiplied by capped profit factor, discounted when the sample is
thin. Select *Custom max* in the optimisation settings to use it. It refuses
to score runs with fewer than 40 trades.

### Do not mass-optimise this

The EA already adapts online. Grid-searching 15 inputs over one history will
produce a beautiful curve that means nothing. If you optimise anything,
optimise the three that change the *problem* rather than the fit:

* `InpLabelHorizon` and `InpBarrierATR` — these define what the model is
  predicting. Ratio matters more than absolute values; the barrier should be
  reachable within the horizon on a typical bar.
* `InpProbThreshold` — the trade/no-trade cut. Higher means fewer, better
  trades; too high and you never trade.

Then verify on a period you did not touch.

---

## 4. Optional: pre-train a checkpoint

Only worth doing if you want the bot to start with an opinion instead of
spending its warm-up flat, or — more usefully — if you want to measure whether
the feature set has any edge at all before committing.

```bash
# 1. Backtest with InpExportDataset = true
#    -> <data folder>/MQL5/Files/NAS100ML/dataset.csv

# 2. Train
python3 tools/train_offline.py --data dataset.csv --out model.txt

# 3. Sanity-check the file the EA will read
python3 tools/verify_checkpoint.py model.txt

# 4. Copy model.txt to <data folder>/MQL5/Files/NAS100ML/model.txt
#    and run with InpLoadCheckpoint = true
```

Requires only `numpy`.

The trainer prints ensemble validation accuracy against the majority-class
baseline on a **chronological** tail split. If it does not beat the baseline,
stop — a checkpoint that is not predictive out of sample will not become
predictive by being loaded into MetaTrader.

Note that the EA's warm-up counter starts at zero on every attach, so a loaded
checkpoint still waits `InpMinTrainSamples` bars before trading. Lower that
input if you want the checkpoint to trade immediately — at the cost of losing
the live sanity check on the model.

---

## 5. Parameter reference

### Learning

| Input | Default | Notes |
|---|---|---|
| `InpOnlineLearning` | `true` | `false` freezes the weights but keeps scoring, so the panel still shows live accuracy |
| `InpLearningRate` | `0.010` | AdamW base rate. The MLP runs at 0.6× this |
| `InpWeightDecay` | `0.0001` | Decoupled L2 |
| `InpRffDim` | `64` | More = more capacity and more CPU per bar (O(D·n)) |
| `InpRffSigma` | `8.0` | RBF bandwidth. Roughly √n for n features; too small and the kernel model degenerates to noise |
| `InpMlpHidden` | `24` | Hidden units |
| `InpHedgeEta` | `0.35` | How fast expert weights react. 0 freezes the blend at equal weights |
| `InpReplaySteps` | `4` | Extra gradient steps per new sample |
| `InpReplayCapacity` | `4000` | Rolling window the replay draws from |
| `InpMinTrainSamples` | `400` | Warm-up before the first trade |
| `InpEvalWindow` | `300` | Window for the rolling accuracy gate |

### Label

| Input | Default | Notes |
|---|---|---|
| `InpLabelHorizon` | `12` | Bars to the vertical barrier |
| `InpBarrierATR` | `1.20` | Horizontal barriers, in ATR |
| `InpTimeBarrierWeight` | `0.50` | Weight of samples that expired without touching a barrier |

### Signal gates

| Input | Default | Notes |
|---|---|---|
| `InpProbThreshold` | `0.58` | Buy at `p ≥ x`, sell at `p ≤ 1−x` |
| `InpMinRollAccuracy` | `0.52` | Stop trading when the walk-forward hit rate falls below this |
| `InpMaxDisagreement` | `0.45` | Max spread between the three experts' probabilities |

### Risk

| Input | Default | Notes |
|---|---|---|
| `InpRiskPercent` | `0.50` | Per trade, as % of equity. Start lower |
| `InpMaxDailyLossPct` | `3.0` | Stops trading for the rest of the server day |
| `InpMaxDrawdownPct` | `15.0` | Kill switch from the equity high-water mark. Does not re-arm |
| `InpMaxSpreadPoints` | `80` | Absolute cap. Check your broker's typical NAS100 spread |
| `InpMaxSpreadAtrFrac` | `0.15` | Spread cap relative to ATR |
| `InpMaxTradesPerDay` | `8` | |
| `InpLossStreakTrigger` / `InpCooldownBars` | `3` / `8` | Pause after consecutive losses |
| `InpMaxLots` | `5.0` | Hard cap regardless of what sizing computes |

### Exits

| Input | Default | Notes |
|---|---|---|
| `InpStopATR` | `1.50` | Initial stop distance |
| `InpTakeProfitR` | `1.60` | Target as a multiple of initial risk |
| `InpBreakEvenR` / `InpBreakEvenOffsetR` | `0.90` / `0.10` | Break-even lift |
| `InpTrailATR` / `InpTrailStartR` | `2.00` / `1.30` | ATR trail, once genuinely in profit |
| `InpMaxHoldBars` | `36` | Time stop. `0` disables |

Keep `InpStopATR` and `InpBarrierATR` in the same neighbourhood. The label
says "which barrier at ±1.2 ATR gets hit first" — putting the actual stop at
0.3 ATR means the model is answering a question the trade never asks.

---

## 6. Troubleshooting

**"failed to create one or more indicator handles"**
The symbol has no history for the working timeframe. Open the chart, scroll
back to force a download, and restart the EA.

**Panel stuck on "warming up (indicator history)"**
`CFeatureEngine::Compute()` needs 260 bars plus D1 data for the previous-day
levels. Let the terminal download history.

**Panel stuck on "warm-up N/400"**
Normal. Samples only mature after `InpLabelHorizon` bars, so on M15 you need
roughly 400 + 12 bars before the first trade is even considered.

**"blocked: spread too wide"**
Your broker's NAS100 spread exceeds `InpMaxSpreadPoints` or 15% of ATR. Check
the real spread at the hours you trade — if it is genuinely that wide, this
strategy's edge probably does not survive it, and raising the cap does not fix
that.

**"risk budget below minimum lot"**
`equity × risk% ÷ stop distance` came out under the broker's minimum volume.
Either the account is small, the stop is wide, or the symbol's contract size is
large. Raising `InpRiskPercent` "fixes" it by risking more — understand that
before you do it.

**"model below accuracy floor"**
Working as intended: the ensemble is not beating a coin flip on recent data,
so it is standing aside. If it never clears the floor, the features have no
edge on your data.

**Checkpoint not loading**
Run `tools/verify_checkpoint.py` on it. The most common cause is a feature
count mismatch — a checkpoint is only valid for the exact feature set that
produced it.

# XAUUSD Scalper

A gold scalper that follows the market across the weekend: **XAUUSD.s** Monday to
Friday, **XAUUSD24/7.s** while spot is shut, handing back when spot re-opens.

Files:

```
MQL5/Experts/XAUUSD_Scalper.mq5     the EA
MQL5/Include/XAUScalp/
  SymbolRouter.mqh                  which symbol is live, and for how long
  SignalEngine.mqh                  the two mechanisms
  ScalpRisk.mqh                     sizing and every pre-trade gate
tools/backtest_scalper.py           the measurement - no dependencies
```

---

## Read this first

**Nothing in here is known to be profitable.** This repository already contains
one bot that measured beautifully and lost 32% of a year
(`NAS100_ML_Bot`), and the tooling that caught it. The same standard applies
here: a gold scalper is a *hypothesis*, and `tools/backtest_scalper.py` is how
you test it on your broker's feed before it touches money.

The bot is built so that test is possible and honest — non-overlapping trades,
a real cost model, and a control that isolates the only thing the strategy
actually claims. What it is not is a promise. If the edge test below comes back
flat on your data, the correct response is not to tune it until the backtest
smiles.

---

## Why gold scalping is hard, in one table

This is arithmetic, not opinion. Gold at $2,400 moving 1.2% a day, stop at
1× ATR, spread quoted in points (1 point = $0.01):

| Timeframe | typical ATR | spread 20 pts | spread 30 pts | spread 60 pts |
|---|---|---|---|---|
| M5  | ~$2.20 | 0.091 R | 0.136 R | 0.272 R |
| M15 | ~$3.80 | 0.052 R | 0.079 R | 0.157 R |
| H1  | ~$7.60 | 0.026 R | 0.039 R | 0.079 R |

Every cell is what you pay, per trade, as a fraction of the money you are
risking. On M5 with a 30-point spread you start every trade **0.14 R behind**.
A strategy winning 40% of its trades at 1.6 R makes 0.04 R gross — the spread
alone turns that into a losing bot. This is the single fact that decides whether
gold scalping works at your broker, and it is why the EA's spread gate is
expressed as *a fraction of the stop distance* rather than as a fixed number of
points.

If your feed quotes 30+ points on XAUUSD, run the measurement on M15 before M5.
Not because M15 is a better idea, but because at M5 the arithmetic above has to
be overcome before anything else can matter.

---

## The weekend switch

### How it decides

Default `InpRouteMode = XS_ROUTE_AUTO` reads the broker's own trading session
table for the spot symbol. That table already knows about DST shifts and
holiday closes, so nothing needs re-tuning twice a year.

The rule is not "spot is closed → trade the weekend book". Gold has a **daily
maintenance break** (typically 23:59–01:00 server time) and trading a thin 24/7
book during a one-hour nightly gap is not what anyone means by a weekend bot.
So a closed stretch only counts as a weekend once it exceeds
`InpWeekendGapHours` (default 6). During the nightly break the EA stands down.

Verified against both common broker table styles:

| Now | daily-break table (Mon–Fri 01:00–23:59) | continuous 24/5 table |
|---|---|---|
| Wed 12:00 | spot | spot |
| Wed 00:30 | stand down (1.0h gap) | spot |
| Fri 23:50 | spot | spot |
| Sat 12:00 | **weekend book** (49h gap) | **weekend book** (48h gap) |
| Sun 22:00 | **weekend book** | **weekend book** |
| Mon 00:30 | **weekend book** (spot opens 01:00) | spot |

`XS_ROUTE_CLOCK` replaces the table with a fixed Friday-close / Monday-open
window (`InpWeekStart*` / `InpWeekEnd*`, server time) for brokers whose session
table is wrong or empty. `XS_ROUTE_WEEKDAY_ONLY` disables the weekend leg
entirely.

### Symbol names

`InpWeekdaySymbol` / `InpWeekendSymbol` are matched exactly first, then
case-insensitively with `. / _ -` stripped, then by containment. So
`XAUUSD24/7.s`, `XAUUSD247.s` and `xauusd24-7.S` all resolve to whatever your
broker actually calls it. If nothing matches, the EA says so in the journal and
runs weekday-only rather than silently doing nothing.

### The handover

Positions are **never** carried across it. `InpFlattenMinutes` (default 20)
before the switch the EA stops opening and closes what it holds. Two different
instruments, two different books, and a weekend gap between them — carrying a
scalp across that is not a scalp.

### Live: the timer matters

`OnTick()` only fires on ticks of the **chart** symbol. Attach the EA to a
XAUUSD.s chart and that chart is silent all weekend, so a tick-driven EA would
never wake up to trade the weekend book. This EA also runs on a one-second
timer, so it works from either chart. Attaching it to the **24/7 chart** is
still the better choice — the terminal keeps that symbol's data fresh.

### Strategy Tester: the weekend leg cannot be tested from a spot chart

The tester's clock is driven by the chart symbol. Test on XAUUSD.s and there are
no weekend ticks at all, so the weekend branch never executes — you will get a
weekday-only result and no warning beyond the one the EA prints on init. To test
the weekend leg in MetaTrader, run the tester **on the 24/7 symbol**.

Or use `tools/backtest_scalper.py`, which replays both books on one merged
timeline and reports them separately.

---

## The two mechanisms

Both read **closed bars only** and normalise every threshold by ATR, so the same
parameters mean the same thing in a quiet Asian session, during a CPI print, and
on the thin weekend book.

### Breakout (default)

Continuation out of an `InpDonchian`-bar range:

* bar 1 closes beyond the extreme of the preceding `InpDonchian` bars
* that bar's range ≥ `InpExpansionAtr` × ATR and its body ≥ `InpBodyFrac` of its
  range — a real move, not a drift over the line
* EMA stack agrees (`fast > slow` for longs) and price is on the right side of
  the fast EMA
* the higher timeframe agrees (`InpTrendTF` close vs its EMA)
* `ADX ≥ InpMinAdx` — there is a trend to continue
* price is not already more than `InpMaxStretchAtr` × ATR from the fast EMA —
  buying the top of a spike leaves no room to the target

Stop: behind the breakout bar's extreme, or `InpBoStopAtr` × ATR, whichever is
further, capped at `InpMaxStopAtr` × ATR. Target `InpBoTargetR` × R.

### Fade

Mean reversion, for ranging conditions:

* close outside the Bollinger band with RSI at `InpFadeRsi` (mirrored for shorts)
* stretched at least `InpFadeStretchAtr` × ATR from the fast EMA
* `ADX ≤ InpFadeMaxAdx` — do not fade a market that is trending
* optionally only in the higher timeframe's direction (`InpFadeWithHTF`)

`XS_MODE_BOTH` tries breakout first, then fade. Measure the three settings
separately before choosing; the backtest reports results per mechanism.

---

## Two things I measured, and what they mean for the defaults

Both come from replaying the strategy on a **synthetic driftless random walk**
at zero spread with the direction replaced by a coin flip — a series with no
edge in it by construction, which is exactly what isolates the mechanics.

**1. Break-even and trailing stops are not free.** n ≈ 14,700 trades per
configuration:

| Exits | expectancy | vs plain stop/target |
|---|---|---|
| stop and target only | −0.0725 R (SE 0.0103) | — |
| + 24-bar time stop | −0.0733 R (SE 0.0100) | free |
| + break-even at 0.8 R | −0.1164 R (SE 0.0092) | **−0.044 R** |
| + break-even and ATR trail | −0.1292 R (SE 0.0091) | **−0.057 R** |

Moving the stop to break even converts trades that would have recovered into
scratches, and the trail cuts winners short. On a series with no autocorrelation
that costs about 0.05 R per trade — comparable to the entire spread bill. They
may well pay for themselves on a market that actually trends, but they have to
earn it. **So `InpBeTriggerR` and `InpTrailStartR` both default to 0 (off).**
Turn them on only if the measurement on *your* data says they help.

**2. The bar replay has a pessimism floor of about −0.07 R.** When a single bar
contains both the stop and the target, the replay cannot know which came first
and always resolves it as a loss. That is why the random-direction control —
not zero — is the bar the strategy has to clear.

---

## Getting a number

Export both symbols on the trading timeframe with
`MQL5/Scripts/ExportBars.mq5` (run it on each chart), then:

```bash
python3 tools/backtest_scalper.py --bars XAUUSD.s_M5.csv \
                                  --weekend-bars XAUUSD24-7.s_M5.csv \
                                  --commission 7.0
```

No numpy, no pandas — standard library only.

With no data at all, `--demo` generates a random walk and runs the whole
pipeline on it. Everything should come back flat-to-negative; that is the
harness proving it does not manufacture edges.

### The line that matters

```
EDGE over the control : +0.0412 R   (SE 0.0290)   t = 1.42
```

The control replays the **same entry times, the same position sizes, the same
exits and the same spread**, replacing only the direction call with a coin flip.
Everything the strategy does not claim — session filter, sizing, cost, the
replay's own pessimism — is present on both sides and cancels. What is left is
the only thing the bot asserts: that it knows which way to go.

Because the EA holds one position at a time, these trades do not overlap, so
unlike the ML bot's walk-forward accuracy that t statistic is honest.

**Before it trades real money, require all of:**

1. `EDGE over the control` positive with **t ≥ 2**
2. both halves of the sample positive
3. every walk-forward fold positive, or close to it
4. it survives at your real commission, not at zero
5. at least a few hundred trades — a quarter of gold scalping is noise, and this
   repository has the receipts on what short windows do to a conclusion
   (see the seed lottery in the main README)

Miss any of them and you have a backtest, not a strategy.

### Everything the report breaks out

Return, drawdown, win rate, profit factor; expectancy in R with its standard
error; the zero-spread run (is cost the problem, or the strategy?); the control
and the edge over it; first half vs second half; walk-forward folds; and per
book, mechanism, weekday, entry hour, exit reason and month.

The **by book** rows are how you find out whether the weekend leg is worth
running at all. Expect it to be the weaker one.

---

## The weekend book deserves its own warning

* **The spread is the whole story.** Weekend gold typically quotes several times
  the weekday spread. `InpWeMaxSpreadFrac` (default 0.25, versus 0.12 on
  weekdays) is what decides whether the weekend leg trades at all. If your
  broker quotes 100+ points, most weekend setups will be refused — that is the
  gate working, not failing.
* **It is a different instrument.** It is the broker's own book, not the spot
  market. Its price can and does diverge from where spot re-opens on Monday.
* **Risk is halved by default** (`InpWeekendRiskMult = 0.5`), because thin
  liquidity means the stop is more likely to fill somewhere other than its level.
* **Measure it separately.** If the `by book` rows say the weekend leg loses,
  set `InpRouteMode = XS_ROUTE_WEEKDAY_ONLY` and keep the weekday bot.

---

## Install

1. Copy `MQL5/Experts/XAUUSD_Scalper.mq5`, `MQL5/Include/XAUScalp/` and
   `MQL5/Include/NAS100ML/` (the EA reuses `Utils.mqh` and `TradeExecutor.mqh`)
   into your terminal's data folder — File → Open Data Folder.
2. Compile in MetaEditor (F7).
3. Set `InpWeekdaySymbol` and `InpWeekendSymbol` to your broker's exact names.
4. **Set the session hours for your broker's server time.** `InpSessionStart` /
   `InpSessionEnd` default to 9–23, which suits a GMT+3 server (London open to
   the New York afternoon). Derive yours the way the NAS100 work did: look for
   the server hour where bar range and tick volume roughly double, which is the
   London open.
5. Attach to a chart — the 24/7 one for preference — with algo trading enabled.
6. Run it on demo until you have watched it hand over across at least one
   weekend in both directions.

### Journal lines worth reading on the first run

```
XAUScalp: weekday=XAUUSD.s  weekend=XAUUSD24/7.s  tf=PERIOD_M5  mode=XS_MODE_BREAKOUT
XAUScalp: XAUUSD.s open, closes in 07h 12m
XAUScalp: XAUUSD.s -> XAUUSD24/7.s   (XAUUSD.s shut for 00h 21m, opens in 48h 39m)
```

If the second line says something implausible, your broker's session table is
not usable — switch to `XS_ROUTE_CLOCK` and set the window by hand.

---

## Risk layer

* fixed-fractional sizing from the ATR stop distance and the broker's real tick
  value — never a fixed lot
* absolute **and** relative spread gates, per book
* daily loss stop, account drawdown kill switch, trades-per-day cap
* cooldown after a loss streak
* margin pre-check before every order
* session-hours filter, separately for each book
* one position at a time, across both symbols

Exits: ATR stop, R-multiple target, bar-count time stop, and optional break-even
and ATR trailing (off by default — see the measurement above).

---

## Honest limitations

* **No edge is promised.** Only your own walk-forward test on your own feed can
  answer that, and the answer may be no.
* **Bar replay is not tick replay.** Intrabar order is unknown, so ambiguous
  bars resolve as losses (a ~0.07 R pessimism), and the trailing stop is
  confirmed on closes rather than tick by tick.
* **Slippage is not modelled.** On the weekend book especially, assume it is
  worse than you think.
* **Gold gaps.** Sunday's re-open is not obliged to respect Friday's close, and
  the weekend book's own price is not obliged to respect either.
* **Two symbols means two sets of broker conditions** — different spreads,
  different stop levels, sometimes different contract sizes. The EA reads each
  symbol's own specification, but check both before you trade.

Trade it on demo until you have seen it handle a full drawdown cycle.

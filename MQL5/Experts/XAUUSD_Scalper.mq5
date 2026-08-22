//+------------------------------------------------------------------+
//|                                               XAUUSD_Scalper.mq5 |
//|                                                                  |
//|  A gold scalper that follows the market across the weekend.      |
//|                                                                  |
//|  Monday to Friday it trades the spot symbol (XAUUSD.s). When     |
//|  that market shuts for the weekend it moves to the broker's      |
//|  round-the-clock book (XAUUSD24/7.s) and trades that instead,    |
//|  then hands back when spot re-opens. Positions are never carried |
//|  across the handover: the two symbols are different instruments  |
//|  with different books, and a position on the wrong side of a     |
//|  weekend gap is not a scalp.                                     |
//|                                                                  |
//|  Two mechanisms, chosen with InpMode:                            |
//|    BREAKOUT  continuation out of an N-bar range on an expansion  |
//|              bar, with trend and higher-timeframe agreement      |
//|    FADE      mean reversion from a Bollinger extreme when ADX    |
//|              says the market is ranging                          |
//|                                                                  |
//|  READ THIS BEFORE TRADING IT                                     |
//|  ---------------------------                                     |
//|  Nothing here is known to be profitable on your feed. The        |
//|  strategy is built to be measurable, and the measurement is      |
//|  tools/backtest_scalper.py - run it on your own exported bars    |
//|  before risking money. The weekend book in particular usually    |
//|  quotes a spread several times the weekday one; the relative     |
//|  spread gate will refuse most weekend trades on purpose, and     |
//|  that refusal is the correct behaviour, not a bug.               |
//|                                                                  |
//|  See docs/XAUUSD_SCALPER.md.                                     |
//+------------------------------------------------------------------+
#property copyright "NAS100ML"
#property link      "https://github.com/ParadiseDevelopment/Nasdaq"
#property version   "1.00"
#property description "XAUUSD scalper - spot on weekdays, the 24/7 book at weekends"

#include <NAS100ML/Utils.mqh>
#include <NAS100ML/TradeExecutor.mqh>
#include <XAUScalp/SymbolRouter.mqh>
#include <XAUScalp/SignalEngine.mqh>
#include <XAUScalp/ScalpRisk.mqh>

//+------------------------------------------------------------------+
input group "=== Symbols and the weekend switch ==="
input string          InpWeekdaySymbol   = "XAUUSD.s";      // Weekday symbol
input string          InpWeekendSymbol   = "XAUUSD24/7.s";  // Weekend symbol
input ENUM_XS_ROUTE   InpRouteMode       = XS_ROUTE_AUTO;   // How the weekend is detected
input double          InpWeekendGapHours = 6.0;             // Closed stretch that counts as a weekend
input int             InpFlattenMinutes  = 20;              // Go flat this long before a handover
input int             InpWeekStartDow    = 1;               // CLOCK mode: week opens (0=Sun..6=Sat)
input int             InpWeekStartHour   = 1;               // CLOCK mode: week opens (server hour)
input int             InpWeekStartMinute = 5;               // CLOCK mode: week opens (minute)
input int             InpWeekEndDow      = 5;               // CLOCK mode: week closes (0=Sun..6=Sat)
input int             InpWeekEndHour     = 23;              // CLOCK mode: week closes (server hour)
input int             InpWeekEndMinute   = 45;              // CLOCK mode: week closes (minute)

input group "=== Strategy ==="
input ENUM_XS_MODE    InpMode            = XS_MODE_BREAKOUT; // Mechanism
input ENUM_TIMEFRAMES InpTimeframe       = PERIOD_M5;        // Trading timeframe
input ENUM_TIMEFRAMES InpTrendTF         = PERIOD_H1;        // Higher timeframe filter
input bool            InpUseHTF          = true;             // Use the higher timeframe filter
input int             InpHtfEmaPeriod    = 50;               // Higher timeframe EMA
input int             InpDonchian        = 20;               // Breakout channel length (bars)
input int             InpEmaFast         = 20;               // Fast EMA
input int             InpEmaSlow         = 50;               // Slow EMA
input int             InpAtrPeriod       = 14;               // ATR period
input int             InpRsiPeriod       = 14;               // RSI period
input int             InpAdxPeriod       = 14;               // ADX period
input int             InpBbPeriod        = 20;               // Bollinger period
input double          InpBbDev           = 2.0;              // Bollinger deviations

input group "=== Breakout mode ==="
input double          InpExpansionAtr    = 0.80;   // Breakout bar range, minimum (x ATR)
input double          InpBodyFrac        = 0.50;   // Breakout bar body, minimum (x its range)
input double          InpMinAdx          = 18.0;   // Minimum ADX
input double          InpMaxStretchAtr   = 2.50;   // Maximum distance from the fast EMA (x ATR)
input double          InpBoStopAtr       = 1.00;   // Stop (x ATR, or behind the bar)
input double          InpBoTargetR       = 1.60;   // Target (R)

input group "=== Fade mode ==="
input double          InpFadeRsi         = 25.0;   // RSI at or below this to buy (mirrored to sell)
input double          InpFadeStretchAtr  = 1.00;   // Minimum stretch from the fast EMA (x ATR)
input double          InpFadeMaxAdx      = 25.0;   // Maximum ADX - above this the market trends
input bool            InpFadeWithHTF     = true;   // Only fade with the higher timeframe
input double          InpFadeStopAtr     = 1.20;   // Stop (x ATR, or behind the bar)
input double          InpFadeTargetR     = 1.00;   // Target (R)

input group "=== Sessions (BROKER SERVER HOURS) ==="
input int             InpSessionStart    = 9;      // Weekday trading from this hour
input int             InpSessionEnd      = 23;     // Weekday trading until this hour
input int             InpWeSessionStart  = 0;      // Weekend from this hour (0/0 = all hours)
input int             InpWeSessionEnd    = 0;      // Weekend until this hour

input group "=== Risk ==="
input double          InpRiskPercent     = 0.25;   // Risk per trade (% of equity)
input double          InpWeekendRiskMult = 0.50;   // Weekend risk multiplier
input double          InpMaxDailyLossPct = 2.0;    // Stop for the day after losing this much (%)
input double          InpMaxDrawdownPct  = 12.0;   // Kill switch drawdown (%)
input double          InpMaxSpreadPoints = 40.0;   // Weekday spread cap (points, 0 = off)
input double          InpMaxSpreadFrac   = 0.12;   // Weekday spread cap (fraction of the stop)
input double          InpWeMaxSpreadPts  = 150.0;  // Weekend spread cap (points, 0 = off)
input double          InpWeMaxSpreadFrac = 0.25;   // Weekend spread cap (fraction of the stop)
input int             InpMaxTradesPerDay = 12;     // Trades per day (0 = unlimited)
input int             InpLossStreak      = 3;      // Losses in a row before a cooldown
input int             InpCooldownBars    = 12;     // Cooldown length (bars)
input double          InpMaxLots         = 5.0;    // Hard volume cap
input double          InpMarginFraction  = 0.25;   // Max share of free margin per trade

input group "=== Exits ==="
input double          InpMaxStopAtr      = 2.50;   // Hard cap on the stop distance (x ATR)
input int             InpMaxBarsInTrade  = 24;     // Time stop (bars, 0 = off)
input double          InpBeTriggerR      = 0.00;   // Break even at this many R (0 = off, see docs)
input double          InpBeOffsetR       = 0.05;   // Where break even sits (R)
input double          InpTrailStartR     = 0.00;   // Trail from this many R (0 = off, see docs)
input double          InpTrailAtr        = 1.20;   // Trailing distance (x ATR)

input group "=== General ==="
input long            InpMagic           = 771003; // Magic number
input ulong           InpSlippagePts     = 25;     // Max slippage (points)
input bool            InpShowPanel       = true;   // Status panel on the chart

//+------------------------------------------------------------------+
CSymbolRouter  g_router;
CSignalEngine  g_engWeekday;
CSignalEngine  g_engWeekend;
CTradeExecutor g_execWeekday;
CTradeExecutor g_execWeekend;
CScalpRisk     g_risk;

ENUM_XS_ACTIVE g_active      = XS_ACTIVE_NONE;
datetime       g_lastBar     = 0;
double         g_atr         = 0.0;
double         g_stopDist    = 0.0;
int            g_barsInTrade = 0;
int            g_trades      = 0;
int            g_wins        = 0;
double         g_realised    = 0.0;
string         g_state       = "starting";
bool           g_busy        = false;

//+------------------------------------------------------------------+
CSignalEngine *EngineFor(const ENUM_XS_ACTIVE a)
  {
   if(a == XS_ACTIVE_WEEKEND)
      return(GetPointer(g_engWeekend));
   return(GetPointer(g_engWeekday));
  }

CTradeExecutor *ExecFor(const ENUM_XS_ACTIVE a)
  {
   if(a == XS_ACTIVE_WEEKEND)
      return(GetPointer(g_execWeekend));
   return(GetPointer(g_execWeekday));
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   if(InpEmaFast >= InpEmaSlow)
     {
      Print("XAUScalp: the fast EMA must be shorter than the slow one");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(InpDonchian < 5 || InpAtrPeriod < 2)
     {
      Print("XAUScalp: InpDonchian >= 5 and InpAtrPeriod >= 2 please");
      return(INIT_PARAMETERS_INCORRECT);
     }

   if(!g_router.Init(InpWeekdaySymbol, InpWeekendSymbol, InpRouteMode, InpWeekendGapHours,
                     InpWeekStartDow, InpWeekStartHour * 3600 + InpWeekStartMinute * 60,
                     InpWeekEndDow,   InpWeekEndHour   * 3600 + InpWeekEndMinute   * 60))
     {
      Print("XAUScalp: symbol routing could not be set up - ", g_router.Note());
      return(INIT_FAILED);
     }

   SXSParams p;
   p.mode           = (int)InpMode;
   p.atrPeriod      = InpAtrPeriod;
   p.rsiPeriod      = InpRsiPeriod;
   p.adxPeriod      = InpAdxPeriod;
   p.emaFast        = InpEmaFast;
   p.emaSlow        = InpEmaSlow;
   p.bbPeriod       = InpBbPeriod;
   p.bbDev          = InpBbDev;
   p.donchian       = InpDonchian;
   p.htfEmaPeriod   = InpHtfEmaPeriod;
   p.useHtf         = InpUseHTF;
   p.expansionAtr   = InpExpansionAtr;
   p.bodyFrac       = InpBodyFrac;
   p.minAdx         = InpMinAdx;
   p.maxStretchAtr  = InpMaxStretchAtr;
   p.boStopAtr      = InpBoStopAtr;
   p.boTargetR      = InpBoTargetR;
   p.fadeRsi        = InpFadeRsi;
   p.fadeStretchAtr = InpFadeStretchAtr;
   p.fadeMaxAdx     = InpFadeMaxAdx;
   p.fadeWithHtf    = InpFadeWithHTF;
   p.fadeStopAtr    = InpFadeStopAtr;
   p.fadeTargetR    = InpFadeTargetR;
   p.maxStopAtr     = InpMaxStopAtr;

   if(!g_engWeekday.Init(g_router.WeekdaySymbol(), InpTimeframe, InpTrendTF, p))
     {
      Print("XAUScalp: cannot create indicators for ", g_router.WeekdaySymbol());
      return(INIT_FAILED);
     }
   g_execWeekday.Init(g_router.WeekdaySymbol(), InpMagic, InpSlippagePts);
   g_execWeekday.ConfigureExits(InpBeTriggerR, InpBeOffsetR, InpTrailAtr, InpTrailStartR);

   if(g_router.WeekendUsable())
     {
      if(!g_engWeekend.Init(g_router.WeekendSymbol(), InpTimeframe, InpTrendTF, p))
        {
         Print("XAUScalp: cannot create indicators for ", g_router.WeekendSymbol());
         return(INIT_FAILED);
        }
      g_execWeekend.Init(g_router.WeekendSymbol(), InpMagic, InpSlippagePts);
      g_execWeekend.ConfigureExits(InpBeTriggerR, InpBeOffsetR, InpTrailAtr, InpTrailStartR);
     }

   g_risk.Configure(InpRiskPercent, InpMaxDailyLossPct, InpMaxDrawdownPct,
                    InpMaxTradesPerDay, InpLossStreak, InpCooldownBars,
                    InpMaxLots, InpMarginFraction);

   //--- a one second timer keeps the weekend leg alive: OnTick only fires on
   //--- ticks of the CHART symbol, and the chart symbol is usually the spot
   //--- one, which is silent all weekend
   EventSetTimer(1);

   PrintFormat("XAUScalp: weekday=%s  weekend=%s%s  tf=%s  mode=%s",
               g_router.WeekdaySymbol(),
               g_router.WeekendSymbol() == "" ? "(none)" : g_router.WeekendSymbol(),
               g_router.WeekendUsable() ? "" : " [NOT TRADABLE - weekday only]",
               EnumToString(InpTimeframe), EnumToString(InpMode));
   if(g_router.Note() != "")
      Print("XAUScalp: ", g_router.Note());
   Print("XAUScalp: ", g_router.Describe());

   if(MQLInfoInteger(MQL_TESTER) && _Symbol != g_router.WeekendSymbol())
      Print("XAUScalp: TESTER NOTE - the tester's clock follows the chart symbol, which "
            "has no weekend ticks. Test on ", g_router.WeekendSymbol(),
            " if you want the weekend leg exercised. See docs/XAUUSD_SCALPER.md.");

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   g_engWeekday.Release();
   g_engWeekend.Release();
   Comment("");
  }

//+------------------------------------------------------------------+
void OnTick()  { Process(); }
void OnTimer() { Process(); }

//+------------------------------------------------------------------+
//| One pass of the whole decision, safe to call from either the     |
//| tick or the timer.                                               |
//+------------------------------------------------------------------+
void Process()
  {
   if(g_busy)
      return;
   g_busy = true;
   Work();
   g_busy = false;
  }

//+------------------------------------------------------------------+
void Work()
  {
   g_risk.OnHeartbeat();
   g_router.Refresh();

   ENUM_XS_ACTIVE act = g_router.Resolve();

   //--- handover: flatten whatever the outgoing symbol still holds
   if(act != g_active)
     {
      if(g_active != XS_ACTIVE_NONE)
        {
         CTradeExecutor *old = ExecFor(g_active);
         if(old.HasPosition())
            old.CloseAll("symbol handover");
        }
      PrintFormat("XAUScalp: %s -> %s   (%s)",
                  g_active == XS_ACTIVE_NONE ? "stand down" : g_router.SymbolFor(g_active),
                  act == XS_ACTIVE_NONE ? "stand down" : g_router.SymbolFor(act),
                  g_router.Describe());
      g_active      = act;
      g_lastBar     = 0;
      g_barsInTrade = 0;
     }

   if(act == XS_ACTIVE_NONE)
     {
      g_state = "market shut - standing down";
      Panel();
      return;
     }

   string          sym = g_router.SymbolFor(act);
   CTradeExecutor *ex  = ExecFor(act);
   CSignalEngine  *eng = EngineFor(act);

   //--- go flat before the handover rather than at it
   int  toHandover   = g_router.SecondsToHandover(act);
   bool handoverSoon = (toHandover <= InpFlattenMinutes * 60);
   if(handoverSoon && ex.HasPosition())
     {
      ex.CloseAll("flat before handover");
      g_barsInTrade = 0;
     }

   //--- exits are managed on every pass, not once per bar
   if(ex.HasPosition())
      ex.ManageOpen(g_atr, g_stopDist);

   //--- everything below happens once per closed bar
   datetime t = iTime(sym, InpTimeframe, 0);
   if(t == 0)
     {
      //--- history for a non-chart symbol is still downloading
      g_state = "waiting for " + sym + " history";
      Panel();
      return;
     }
   if(t == g_lastBar)
      return;
   g_lastBar = t;

   g_risk.OnBar();

   //--- poll both books: a position closed at the handover belongs to the
   //--- symbol we have just stopped trading
   double profit = 0.0;
   int    closed = g_execWeekday.PollClosedTrades(profit);
   if(g_router.WeekendUsable())
     {
      double wkndProfit = 0.0;
      closed += g_execWeekend.PollClosedTrades(wkndProfit);
      profit += wkndProfit;
     }
   if(closed > 0)
     {
      g_risk.OnTradeClosed(profit);
      g_realised += profit;
      g_trades++;
      if(profit > 0.0)
         g_wins++;
      g_barsInTrade = 0;
     }

   double atrNow = eng.Atr();
   if(atrNow > 0.0)
      g_atr = atrNow;

   //--- an open position only needs its time stop checked
   if(ex.HasPosition())
     {
      g_barsInTrade++;
      if(InpMaxBarsInTrade > 0 && g_barsInTrade >= InpMaxBarsInTrade)
        {
         ex.CloseAll("time stop");
         g_barsInTrade = 0;
         g_state = "closed on the time stop";
        }
      else
         g_state = StringFormat("in trade, %d bars", g_barsInTrade);
      Panel();
      return;
     }

   if(handoverSoon)
     {
      g_state = "flat into the handover, " + CSymbolRouter::FormatSpan(toHandover) + " left";
      Panel();
      return;
     }

   //--- signal
   SXSSignal sig;
   if(!eng.Evaluate(sig))
     {
      g_state = eng.LastReject();
      Panel();
      return;
     }

   bool   weekend   = (act == XS_ACTIVE_WEEKEND);
   int    sessStart = weekend ? InpWeSessionStart : InpSessionStart;
   int    sessEnd   = weekend ? InpWeSessionEnd   : InpSessionEnd;
   double capPts    = weekend ? InpWeMaxSpreadPts  : InpMaxSpreadPoints;
   double capFrac   = weekend ? InpWeMaxSpreadFrac : InpMaxSpreadFrac;

   ENUM_XS_BLOCK block = g_risk.Check(sym, sig.stopDist, sessStart, sessEnd, capPts, capFrac);
   if(block != XS_OK)
     {
      g_state = "blocked: " + CScalpRisk::BlockText(block);
      Panel();
      return;
     }

   //--- the weekend book is thinner; take less risk on it
   double lots = g_risk.Lots(sym, sig.stopDist, sig.dir > 0);
   if(weekend && InpWeekendRiskMult > 0.0 && InpWeekendRiskMult < 1.0)
     {
      double step = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
      if(step <= 0.0)
         step = 0.01;
      double scaled = MathFloor((lots * InpWeekendRiskMult) / step) * step;
      if(scaled >= SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN))
         lots = scaled;
     }
   if(lots <= 0.0)
     {
      g_state = "blocked: " + CScalpRisk::BlockText(g_risk.LastBlock());
      Panel();
      return;
     }

   double price = (sig.dir > 0) ? SymbolInfoDouble(sym, SYMBOL_ASK)
                                : SymbolInfoDouble(sym, SYMBOL_BID);
   if(price <= 0.0)
     {
      g_state = "no price";
      Panel();
      return;
     }

   double sl = (sig.dir > 0) ? price - sig.stopDist : price + sig.stopDist;
   double tp = (sig.dir > 0) ? price + sig.targetR * sig.stopDist
                             : price - sig.targetR * sig.stopDist;

   if(ex.Open(sig.dir > 0, lots, sl, tp, "xauscalp"))
     {
      g_stopDist    = sig.stopDist;
      g_barsInTrade = 0;
      g_risk.OnTradeOpened();
      g_state = StringFormat("opened %s %.2f lots - %s",
                             sig.dir > 0 ? "long" : "short", lots, sig.why);
      PrintFormat("XAUScalp ENTRY %s %s %.2f lots @ %.2f  sl %.2f  tp %.2f  (%s)",
                  sig.dir > 0 ? "BUY" : "SELL", sym, lots, price, sl, tp, sig.why);
     }
   else
      g_state = "order rejected";

   Panel();
  }

//+------------------------------------------------------------------+
void Panel()
  {
   if(!InpShowPanel)
      return;

   string sym = g_router.SymbolFor(g_active);
   string s = "XAUUSD Scalper\n";
   s += "------------------------------------------\n";
   s += StringFormat("routing      : %s\n", g_router.Describe());
   s += StringFormat("trading      : %s%s\n",
                     sym == "" ? "(nothing)" : sym,
                     g_active == XS_ACTIVE_WEEKEND ? "   [WEEKEND BOOK]" : "");
   if(sym != "")
     {
      double point  = SymbolInfoDouble(sym, SYMBOL_POINT);
      double spread = (double)SymbolInfoInteger(sym, SYMBOL_SPREAD);
      s += StringFormat("spread       : %.0f pts (%.2f price)\n", spread, spread * point);
      s += StringFormat("atr          : %.2f\n", g_atr);
      s += StringFormat("handover in  : %s\n",
                        CSymbolRouter::FormatSpan(g_router.SecondsToHandover(g_active)));
     }
   s += StringFormat("mode         : %s on %s\n",
                     EnumToString(InpMode), EnumToString(InpTimeframe));
   s += StringFormat("trades       : %d  (%.0f%% won)  realised %.2f\n",
                     g_trades, g_trades > 0 ? 100.0 * g_wins / g_trades : 0.0, g_realised);
   s += StringFormat("today        : %d trades%s\n", g_risk.TradesToday(),
                     g_risk.HaltedToday() ? "  [STOPPED FOR THE DAY]" : "");
   if(g_risk.Cooldown() > 0)
      s += StringFormat("cooldown     : %d bars\n", g_risk.Cooldown());
   if(g_risk.Halted())
      s += "HALTED       : drawdown kill switch\n";
   s += "state        : " + g_state + "\n";
   Comment(s);
  }
//+------------------------------------------------------------------+

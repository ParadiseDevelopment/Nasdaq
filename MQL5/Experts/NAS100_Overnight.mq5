//+------------------------------------------------------------------+
//|                                            NAS100_Overnight.mq5  |
//|                                                                  |
//|  Overnight drift harvester for the NASDAQ 100 CFD.               |
//|                                                                  |
//|  Long at a fixed evening time, flat before the cash open. That   |
//|  is the whole strategy. Nothing is predicted and nothing is      |
//|  learned - it exploits the fact that, in equity indices, the     |
//|  drift accrues while the cash market is shut.                    |
//|                                                                  |
//|  On the feed this was built against (NAS100.s M15, Aug 2025 to   |
//|  Aug 2026) the decomposition was stark:                          |
//|                                                                  |
//|      index moved       +6,120 pts over the year                  |
//|      overnight 23->04  +4,504 pts across 200 nights              |
//|      intraday  04->23    -363 pts across 249 days                |
//|                                                                  |
//|  READ THIS BEFORE TRADING IT                                     |
//|  ---------------------------                                     |
//|  The position is deliberately held across the 00:00 rollover, so |
//|  you pay swap EVERY night, and swap decides whether this works.  |
//|  Measured, at 0.5% risk per night:                               |
//|                                                                  |
//|      swap 0 pts/night  ->  +8.0% per year                        |
//|      swap 4 pts/night  ->  +3.5%                                 |
//|      swap 8 pts/night  ->  -1.0%     <-- dead                    |
//|                                                                  |
//|  Check your symbol's long swap (right-click the symbol ->        |
//|  Specification -> Swap long) BEFORE you run this. If it costs    |
//|  more than about 6 index points a night, do not bother.          |
//|                                                                  |
//|  See docs/OVERNIGHT.md for the full measurement and its limits.  |
//+------------------------------------------------------------------+
#property copyright "NAS100ML"
#property link      "https://github.com/ParadiseDevelopment/Nasdaq"
#property version   "1.00"
#property description "Long the index overnight, flat before the cash open. Swap-sensitive."

#include <NAS100ML/Utils.mqh>
#include <NAS100ML/TradeExecutor.mqh>

input group "=== Session (BROKER SERVER TIME) ==="
input int    InpEntryHour     = 23;      // Entry hour
input int    InpEntryMinute   = 0;       // Entry minute
input int    InpExitHour      = 4;       // Exit hour (next day)
input int    InpExitMinute    = 0;       // Exit minute
input int    InpMaxHoldHours  = 12;      // Force flat after this many hours

input group "=== Which nights ==="
input bool   InpSkipMonday    = false;   // Skip Monday nights (UNVALIDATED)
input bool   InpSkipTuesday   = false;   // Skip Tuesday nights
input bool   InpSkipWednesday = false;   // Skip Wednesday nights
input bool   InpSkipThursday  = false;   // Skip Thursday nights
input bool   InpSkipFriday    = true;    // Skip Friday (no exit before the close)

input group "=== Risk ==="
input double InpRiskPercent   = 0.50;    // Risk per night (% of equity)
input double InpStopATR       = 3.00;    // Protective stop (x ATR, 0 = none)
input double InpMaxLots       = 5.00;    // Hard volume cap
input double InpMaxDrawdownPct= 25.0;    // Kill switch drawdown (%)
input double InpMaxSpreadPts  = 400.0;   // Max spread at entry (points)

input group "=== General ==="
input long   InpMagic         = 771002;  // Magic number
input ulong  InpSlippagePts   = 30;      // Max slippage (points)
input bool   InpShowPanel     = true;    // Status panel on chart

//+------------------------------------------------------------------+
CTradeExecutor g_exec;
int            g_hATR         = INVALID_HANDLE;
datetime       g_lastBar      = 0;
datetime       g_entryTime    = 0;
double         g_equityPeak   = 0.0;
bool           g_halted       = false;
int            g_nights       = 0;
double         g_realised     = 0.0;
string         g_state        = "waiting";

//+------------------------------------------------------------------+
int OnInit()
  {
   if(InpEntryHour < 0 || InpEntryHour > 23 || InpExitHour < 0 || InpExitHour > 23)
     {
      Print("Overnight: entry/exit hour must be 0..23");
      return(INIT_PARAMETERS_INCORRECT);
     }

   g_hATR = iATR(_Symbol, _Period, 14);
   if(g_hATR == INVALID_HANDLE)
     {
      Print("Overnight: cannot create ATR handle");
      return(INIT_FAILED);
     }

   g_exec.Init(_Symbol, InpMagic, InpSlippagePts);
   g_equityPeak = AccountInfoDouble(ACCOUNT_EQUITY);
   g_lastBar    = iTime(_Symbol, _Period, 0);

   PrintFormat("Overnight: long %02d:%02d -> flat %02d:%02d server time on %s. "
               "CHECK YOUR SWAP - this strategy dies above ~6 pts/night.",
               InpEntryHour, InpEntryMinute, InpExitHour, InpExitMinute, _Symbol);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(g_hATR != INVALID_HANDLE)
      IndicatorRelease(g_hATR);
   Comment("");
  }

//+------------------------------------------------------------------+
//| True when this weekday is one we are allowed to enter on.        |
//+------------------------------------------------------------------+
bool NightAllowed(const int dayOfWeek)
  {
   switch(dayOfWeek)
     {
      case 1: return(!InpSkipMonday);
      case 2: return(!InpSkipTuesday);
      case 3: return(!InpSkipWednesday);
      case 4: return(!InpSkipThursday);
      case 5: return(!InpSkipFriday);
     }
   return(false);            // weekend
  }

//+------------------------------------------------------------------+
double CurrentATR()
  {
   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(g_hATR, 0, 1, 1, buf) != 1)
      return(0.0);
   return(buf[0]);
  }

//+------------------------------------------------------------------+
//| Volume from the money at risk. With no stop we still size off a  |
//| nominal distance so risk per night stays comparable.             |
//+------------------------------------------------------------------+
double CalcLots(const double riskDistance)
  {
   if(riskDistance <= 0.0)
      return(0.0);

   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickSize <= 0.0)
      tickSize = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(tickValue <= 0.0 || tickSize <= 0.0)
      return(0.0);

   double riskMoney = AccountInfoDouble(ACCOUNT_EQUITY) * InpRiskPercent / 100.0;
   double lots = riskMoney / (riskDistance * (tickValue / tickSize));

   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(stepLot <= 0.0)
      stepLot = 0.01;

   lots = MathFloor(lots / stepLot) * stepLot;
   lots = MathMin(lots, MathMin(maxLot, InpMaxLots));
   if(lots < minLot)
      return(0.0);

   int digits = (int)MathMax(0, MathCeil(-MathLog(stepLot) / MathLog(10.0)));
   return(NormalizeDouble(lots, digits));
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   //--- equity high-water mark and kill switch, checked continuously
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq > g_equityPeak)
      g_equityPeak = eq;
   double ddPct = SafeDiv(g_equityPeak - eq, g_equityPeak) * 100.0;
   if(!g_halted && ddPct >= InpMaxDrawdownPct)
     {
      g_halted = true;
      PrintFormat("Overnight: max drawdown %.2f%% reached - trading disabled", ddPct);
      g_exec.CloseAll("kill switch");
     }

   bool hasPos = g_exec.HasPosition();

   //--- hard time stop, enforced on every tick rather than on the bar,
   //--- so a missing exit bar cannot strand the position overnight
   if(hasPos && g_entryTime > 0 &&
      (TimeCurrent() - g_entryTime) > (long)InpMaxHoldHours * 3600)
     {
      g_exec.CloseAll("max hold time");
      g_entryTime = 0;
      g_state = "closed on max hold";
      return;
     }

   //--- everything else happens once per bar
   datetime t = iTime(_Symbol, _Period, 0);
   if(t == g_lastBar || t == 0)
      return;
   g_lastBar = t;

   double profit = 0.0;
   if(g_exec.PollClosedTrades(profit) > 0)
     {
      g_realised += profit;
      g_nights++;
      g_entryTime = 0;
     }

   MqlDateTime dt;
   TimeToStruct(t, dt);

   //--- exit leg
   if(hasPos)
     {
      if(dt.hour == InpExitHour && dt.min == InpExitMinute)
        {
         g_exec.CloseAll("session exit");
         g_entryTime = 0;
         g_state = "flat, exited on schedule";
        }
      else
         g_state = "long, holding overnight";
      if(InpShowPanel) Panel();
      return;
     }

   //--- entry leg
   if(dt.hour == InpEntryHour && dt.min == InpEntryMinute)
     {
      if(g_halted)                       { g_state = "DISABLED (drawdown)"; }
      else if(!NightAllowed(dt.day_of_week)) { g_state = "night skipped by filter"; }
      else
        {
         double spread = (double)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
         if(spread > InpMaxSpreadPts)
            g_state = StringFormat("skipped, spread %.0f pts", spread);
         else
           {
            double atr = CurrentATR();
            if(atr <= 0.0)
               g_state = "skipped, no ATR yet";
            else
              {
               double ask      = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
               double stopDist = InpStopATR > 0.0 ? InpStopATR * atr : 0.0;
               //--- when the stop is disabled, size as if it were 3 ATR so a
               //--- gap cannot deliver many times the intended loss
               double riskDist = stopDist > 0.0 ? stopDist : 3.0 * atr;
               double lots     = CalcLots(riskDist);

               if(lots <= 0.0)
                  g_state = "risk budget below minimum lot";
               else
                 {
                  double sl = stopDist > 0.0 ? ask - stopDist : 0.0;
                  if(g_exec.Open(true, lots, sl, 0.0, "overnight"))
                    {
                     g_entryTime = TimeCurrent();
                     g_state = StringFormat("opened long %.2f lots", lots);
                     PrintFormat("Overnight ENTRY %.2f lots @ %.2f  atr=%.1f  sl=%.2f",
                                 lots, ask, atr, sl);
                    }
                  else
                     g_state = "order rejected";
                 }
              }
           }
        }
     }

   if(InpShowPanel)
      Panel();
  }

//+------------------------------------------------------------------+
void Panel()
  {
   string s = "NAS100 Overnight   " + _Symbol + "\n";
   s += "------------------------------------------\n";
   s += StringFormat("session      : %02d:%02d -> %02d:%02d server\n",
                     InpEntryHour, InpEntryMinute, InpExitHour, InpExitMinute);
   s += StringFormat("nights done  : %d\n", g_nights);
   s += StringFormat("realised     : %.2f\n", g_realised);
   s += StringFormat("equity peak  : %.2f\n", g_equityPeak);
   s += StringFormat("spread now   : %d pts\n", (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD));
   s += StringFormat("swap long    : %.2f  <- the number that decides this\n",
                     SymbolInfoDouble(_Symbol, SYMBOL_SWAP_LONG));
   s += "state        : " + g_state + "\n";
   Comment(s);
  }
//+------------------------------------------------------------------+

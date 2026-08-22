//+------------------------------------------------------------------+
//|                                                    ScalpRisk.mqh |
//|  Everything that can stop a trade happening, and how big it is.   |
//|                                                                  |
//|  The signal engine only ever proposes a direction. This class     |
//|  owns the account: it sizes from the stop distance and the        |
//|  broker's real tick value, and it refuses the trade outright when |
//|  any limit is breached.                                          |
//|                                                                  |
//|  The gate that matters most on gold is the RELATIVE spread one.   |
//|  A scalp risking 1 ATR on M5 is a small target, and a 30-point    |
//|  spread eats a meaningful share of it. Capping the spread as a    |
//|  fraction of the stop distance keeps the cost of every trade      |
//|  proportional to what the trade is trying to win - which is also  |
//|  what keeps the EA out of the weekend book when it is wide.       |
//+------------------------------------------------------------------+
#ifndef __XAUSCALP_SCALPRISK_MQH__
#define __XAUSCALP_SCALPRISK_MQH__

enum ENUM_XS_BLOCK
  {
   XS_OK = 0,
   XS_BLOCK_HALTED,          // drawdown kill switch
   XS_BLOCK_DAILY_LOSS,
   XS_BLOCK_TRADES_PER_DAY,
   XS_BLOCK_COOLDOWN,
   XS_BLOCK_SESSION,
   XS_BLOCK_SPREAD_ABS,
   XS_BLOCK_SPREAD_REL,
   XS_BLOCK_LOT_TOO_SMALL,
   XS_BLOCK_MARGIN,
   XS_BLOCK_MARKET_SHUT
  };

//+------------------------------------------------------------------+
class CScalpRisk
  {
private:
   //--- configuration
   double            m_riskPct;
   double            m_maxDailyLossPct;
   double            m_maxDrawdownPct;
   int               m_maxTradesPerDay;
   int               m_lossStreakTrigger;
   int               m_cooldownBars;
   double            m_maxLots;
   double            m_marginFrac;

   //--- state
   double            m_equityPeak;
   double            m_dayStartEquity;
   int               m_dayKey;
   int               m_tradesToday;
   int               m_lossStreak;
   int               m_cooldownLeft;
   bool              m_haltedForGood;
   bool              m_haltedForDay;
   ENUM_XS_BLOCK     m_lastBlock;

   static int        DayKey(const datetime t)
     {
      MqlDateTime d;
      TimeToStruct(t, d);
      return(d.year * 10000 + d.mon * 100 + d.day);
     }

public:
                     CScalpRisk(void) : m_riskPct(0.25), m_maxDailyLossPct(2.0),
                                        m_maxDrawdownPct(12.0), m_maxTradesPerDay(12),
                                        m_lossStreakTrigger(3), m_cooldownBars(12),
                                        m_maxLots(5.0), m_marginFrac(0.25),
                                        m_equityPeak(0.0), m_dayStartEquity(0.0),
                                        m_dayKey(0), m_tradesToday(0), m_lossStreak(0),
                                        m_cooldownLeft(0), m_haltedForGood(false),
                                        m_haltedForDay(false), m_lastBlock(XS_OK) {}

   void              Configure(const double riskPct, const double maxDailyLossPct,
                               const double maxDrawdownPct, const int maxTradesPerDay,
                               const int lossStreakTrigger, const int cooldownBars,
                               const double maxLots, const double marginFrac)
     {
      m_riskPct           = riskPct;
      m_maxDailyLossPct   = maxDailyLossPct;
      m_maxDrawdownPct    = maxDrawdownPct;
      m_maxTradesPerDay   = maxTradesPerDay;
      m_lossStreakTrigger = lossStreakTrigger;
      m_cooldownBars      = cooldownBars;
      m_maxLots           = maxLots;
      m_marginFrac        = marginFrac;

      double eq = AccountInfoDouble(ACCOUNT_EQUITY);
      m_equityPeak     = eq;
      m_dayStartEquity = eq;
      m_dayKey         = DayKey(TimeCurrent());
     }

   //--- readouts for the panel
   bool              Halted(void)        const { return(m_haltedForGood); }
   bool              HaltedToday(void)   const { return(m_haltedForDay);  }
   int               TradesToday(void)   const { return(m_tradesToday);   }
   int               LossStreak(void)    const { return(m_lossStreak);    }
   int               Cooldown(void)      const { return(m_cooldownLeft);  }
   double            EquityPeak(void)    const { return(m_equityPeak);    }
   ENUM_XS_BLOCK     LastBlock(void)     const { return(m_lastBlock);     }

   static string     BlockText(const ENUM_XS_BLOCK b)
     {
      switch(b)
        {
         case XS_OK:                  return("ok");
         case XS_BLOCK_HALTED:        return("HALTED - max drawdown");
         case XS_BLOCK_DAILY_LOSS:    return("daily loss limit");
         case XS_BLOCK_TRADES_PER_DAY:return("trades per day");
         case XS_BLOCK_COOLDOWN:      return("cooldown after losses");
         case XS_BLOCK_SESSION:       return("outside session hours");
         case XS_BLOCK_SPREAD_ABS:    return("spread too wide");
         case XS_BLOCK_SPREAD_REL:    return("spread too wide vs the stop");
         case XS_BLOCK_LOT_TOO_SMALL: return("risk below the minimum lot");
         case XS_BLOCK_MARGIN:        return("not enough free margin");
         case XS_BLOCK_MARKET_SHUT:   return("market shut");
        }
      return("?");
     }

   //+---------------------------------------------------------------+
   //| Called on every tick: equity high-water mark, day rollover and |
   //| the two halt conditions.                                      |
   //+---------------------------------------------------------------+
   void              OnHeartbeat(void)
     {
      double eq = AccountInfoDouble(ACCOUNT_EQUITY);

      int key = DayKey(TimeCurrent());
      if(key != m_dayKey)
        {
         m_dayKey         = key;
         m_dayStartEquity = eq;
         m_tradesToday    = 0;
         m_haltedForDay   = false;
        }

      if(eq > m_equityPeak)
         m_equityPeak = eq;

      if(m_equityPeak > 0.0)
        {
         double dd = (m_equityPeak - eq) / m_equityPeak * 100.0;
         if(!m_haltedForGood && dd >= m_maxDrawdownPct)
           {
            m_haltedForGood = true;
            PrintFormat("XAUScalp: drawdown %.2f%% >= %.2f%% - trading disabled for good",
                        dd, m_maxDrawdownPct);
           }
        }

      if(m_dayStartEquity > 0.0 && !m_haltedForDay)
        {
         double dayLoss = (m_dayStartEquity - eq) / m_dayStartEquity * 100.0;
         if(dayLoss >= m_maxDailyLossPct)
           {
            m_haltedForDay = true;
            PrintFormat("XAUScalp: down %.2f%% today - no more trades until tomorrow", dayLoss);
           }
        }
     }

   //--- one call per closed bar of the active symbol
   void              OnBar(void)
     {
      if(m_cooldownLeft > 0)
         m_cooldownLeft--;
     }

   //--- one call per closed trade
   void              OnTradeClosed(const double profit)
     {
      if(profit < 0.0)
        {
         m_lossStreak++;
         if(m_lossStreak >= m_lossStreakTrigger)
           {
            m_cooldownLeft = m_cooldownBars;
            m_lossStreak   = 0;
           }
        }
      else
         m_lossStreak = 0;
     }

   void              OnTradeOpened(void) { m_tradesToday++; }

   //+---------------------------------------------------------------+
   //| Session window in server hours. end may be smaller than start, |
   //| which means the window runs over midnight. start == end means  |
   //| "all hours".                                                   |
   //+---------------------------------------------------------------+
   static bool       InSession(const datetime t, const int startHour, const int endHour)
     {
      if(startHour == endHour)
         return(true);
      MqlDateTime d;
      TimeToStruct(t, d);
      if(startHour < endHour)
         return(d.hour >= startHour && d.hour < endHour);
      return(d.hour >= startHour || d.hour < endHour);
     }

   //+---------------------------------------------------------------+
   //| All the pre-trade gates. stopDist is in price units.          |
   //+---------------------------------------------------------------+
   ENUM_XS_BLOCK     Check(const string symbol, const double stopDist,
                           const int sessionStart, const int sessionEnd,
                           const double maxSpreadPoints, const double maxSpreadFracStop)
     {
      m_lastBlock = XS_OK;

      if(m_haltedForGood)
         m_lastBlock = XS_BLOCK_HALTED;
      else if(m_haltedForDay)
         m_lastBlock = XS_BLOCK_DAILY_LOSS;
      else if(m_maxTradesPerDay > 0 && m_tradesToday >= m_maxTradesPerDay)
         m_lastBlock = XS_BLOCK_TRADES_PER_DAY;
      else if(m_cooldownLeft > 0)
         m_lastBlock = XS_BLOCK_COOLDOWN;
      else if(!InSession(TimeCurrent(), sessionStart, sessionEnd))
         m_lastBlock = XS_BLOCK_SESSION;
      else if((ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE)
              != SYMBOL_TRADE_MODE_FULL)
         m_lastBlock = XS_BLOCK_MARKET_SHUT;
      else
        {
         double point  = SymbolInfoDouble(symbol, SYMBOL_POINT);
         double spread = (double)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
         if(maxSpreadPoints > 0.0 && spread > maxSpreadPoints)
            m_lastBlock = XS_BLOCK_SPREAD_ABS;
         else if(maxSpreadFracStop > 0.0 && stopDist > 0.0 &&
                 (spread * point) > maxSpreadFracStop * stopDist)
            m_lastBlock = XS_BLOCK_SPREAD_REL;
        }

      return(m_lastBlock);
     }

   //+---------------------------------------------------------------+
   //| Volume from the money at risk. Returns 0 when the trade cannot |
   //| be sized inside the account's limits.                          |
   //+---------------------------------------------------------------+
   double            Lots(const string symbol, const double stopDist, const bool isLong)
     {
      if(stopDist <= 0.0)
        {
         m_lastBlock = XS_BLOCK_LOT_TOO_SMALL;
         return(0.0);
        }

      double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
      double tickSize  = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
      if(tickSize <= 0.0)
         tickSize = SymbolInfoDouble(symbol, SYMBOL_POINT);
      if(tickValue <= 0.0 || tickSize <= 0.0)
        {
         m_lastBlock = XS_BLOCK_LOT_TOO_SMALL;
         return(0.0);
        }

      double riskMoney = AccountInfoDouble(ACCOUNT_EQUITY) * m_riskPct / 100.0;
      double lots      = riskMoney / (stopDist * (tickValue / tickSize));

      double minLot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
      double maxLot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
      double stepLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
      if(stepLot <= 0.0)
         stepLot = 0.01;

      lots = MathFloor(lots / stepLot) * stepLot;
      lots = MathMin(lots, MathMin(maxLot, m_maxLots));
      if(lots < minLot)
        {
         m_lastBlock = XS_BLOCK_LOT_TOO_SMALL;
         return(0.0);
        }

      int digits = (int)MathMax(0.0, MathCeil(-MathLog(stepLot) / MathLog(10.0)));
      lots = NormalizeDouble(lots, digits);

      //--- margin pre-check
      double price  = isLong ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                             : SymbolInfoDouble(symbol, SYMBOL_BID);
      double margin = 0.0;
      if(OrderCalcMargin(isLong ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, symbol, lots, price, margin))
        {
         double free = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
         if(margin > free * m_marginFrac)
           {
            m_lastBlock = XS_BLOCK_MARGIN;
            return(0.0);
           }
        }

      return(lots);
     }
  };

#endif // __XAUSCALP_SCALPRISK_MQH__
//+------------------------------------------------------------------+

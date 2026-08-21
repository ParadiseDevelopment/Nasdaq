//+------------------------------------------------------------------+
//|                                                  RiskManager.mqh |
//|  Capital protection layer.                                       |
//|                                                                  |
//|  The model decides direction; this class decides whether the     |
//|  trade is allowed at all and how big it may be. Every limit is   |
//|  checked before an entry and re-checked on the equity curve, so  |
//|  a model that degrades cannot run the account down.              |
//+------------------------------------------------------------------+
#ifndef __NAS100ML_RISKMANAGER_MQH__
#define __NAS100ML_RISKMANAGER_MQH__

#include "Utils.mqh"

//+------------------------------------------------------------------+
enum ENUM_NASML_BLOCK
  {
   NASML_OK = 0,
   NASML_BLOCK_DAILY_LOSS,
   NASML_BLOCK_MAX_DD,
   NASML_BLOCK_SPREAD,
   NASML_BLOCK_SESSION,
   NASML_BLOCK_TRADES_PER_DAY,
   NASML_BLOCK_COOLDOWN,
   NASML_BLOCK_MARGIN
  };

//+------------------------------------------------------------------+
class CRiskManager
  {
private:
   string            m_symbol;

   //--- configuration
   double            m_riskPct;
   double            m_maxDailyLossPct;
   double            m_maxDrawdownPct;
   double            m_maxSpreadPoints;
   double            m_maxSpreadAtrFrac;
   int               m_maxTradesPerDay;
   int               m_cooldownBars;
   int               m_lossStreakTrigger;
   double            m_maxLots;
   double            m_minLots;

   int               m_sessStartHour;
   int               m_sessEndHour;
   bool              m_useSession;

   //--- state
   double            m_equityPeak;
   double            m_dayStartEquity;
   int               m_dayOfYear;
   int               m_tradesToday;
   int               m_lossStreak;
   int               m_cooldownLeft;
   bool              m_haltedForDay;
   bool              m_haltedForGood;
   ENUM_NASML_BLOCK  m_lastBlock;

public:
                     CRiskManager(void) :
                     m_symbol(""), m_riskPct(0.5), m_maxDailyLossPct(3.0),
                     m_maxDrawdownPct(15.0), m_maxSpreadPoints(60.0),
                     m_maxSpreadAtrFrac(0.15), m_maxTradesPerDay(8),
                     m_cooldownBars(6), m_lossStreakTrigger(3),
                     m_maxLots(5.0), m_minLots(0.01),
                     m_sessStartHour(9), m_sessEndHour(22), m_useSession(true),
                     m_equityPeak(0.0), m_dayStartEquity(0.0), m_dayOfYear(-1),
                     m_tradesToday(0), m_lossStreak(0), m_cooldownLeft(0),
                     m_haltedForDay(false), m_haltedForGood(false),
                     m_lastBlock(NASML_OK) {}

   void              Init(const string symbol)
     {
      m_symbol         = symbol;
      double eq        = AccountInfoDouble(ACCOUNT_EQUITY);
      m_equityPeak     = eq;
      m_dayStartEquity = eq;
      m_dayOfYear      = -1;
      m_minLots        = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
     }

   void              Configure(const double riskPct, const double maxDailyLossPct,
                               const double maxDrawdownPct, const double maxSpreadPoints,
                               const double maxSpreadAtrFrac, const int maxTradesPerDay,
                               const int cooldownBars, const int lossStreakTrigger,
                               const double maxLots, const bool useSession,
                               const int sessStart, const int sessEnd)
     {
      m_riskPct           = MathMax(0.01, riskPct);
      m_maxDailyLossPct   = MathMax(0.1,  maxDailyLossPct);
      m_maxDrawdownPct    = MathMax(1.0,  maxDrawdownPct);
      m_maxSpreadPoints   = MathMax(1.0,  maxSpreadPoints);
      m_maxSpreadAtrFrac  = MathMax(0.01, maxSpreadAtrFrac);
      m_maxTradesPerDay   = MathMax(1,    maxTradesPerDay);
      m_cooldownBars      = MathMax(0,    cooldownBars);
      m_lossStreakTrigger = MathMax(1,    lossStreakTrigger);
      m_maxLots           = MathMax(0.01, maxLots);
      m_useSession        = useSession;
      m_sessStartHour     = sessStart;
      m_sessEndHour       = sessEnd;
     }

   ENUM_NASML_BLOCK  LastBlock(void) const { return(m_lastBlock); }
   bool              HaltedForGood(void) const { return(m_haltedForGood); }
   bool              HaltedForDay(void) const { return(m_haltedForDay); }
   int               TradesToday(void) const { return(m_tradesToday); }
   int               LossStreak(void) const { return(m_lossStreak); }
   double            EquityPeak(void) const { return(m_equityPeak); }

   string            BlockReason(void) const
     {
      switch(m_lastBlock)
        {
         case NASML_OK:                  return("ok");
         case NASML_BLOCK_DAILY_LOSS:    return("daily loss limit");
         case NASML_BLOCK_MAX_DD:        return("max drawdown");
         case NASML_BLOCK_SPREAD:        return("spread too wide");
         case NASML_BLOCK_SESSION:       return("outside session");
         case NASML_BLOCK_TRADES_PER_DAY:return("daily trade cap");
         case NASML_BLOCK_COOLDOWN:      return("loss-streak cooldown");
         case NASML_BLOCK_MARGIN:        return("insufficient margin");
        }
      return("unknown");
     }

   //+---------------------------------------------------------------+
   //| Call once per bar. Rolls the daily counters and refreshes the |
   //| equity high-water mark.                                       |
   //+---------------------------------------------------------------+
   void              OnBar(const datetime now)
     {
      double eq = AccountInfoDouble(ACCOUNT_EQUITY);
      if(eq > m_equityPeak)
         m_equityPeak = eq;

      MqlDateTime dt;
      TimeToStruct(now, dt);
      if(dt.day_of_year != m_dayOfYear)
        {
         m_dayOfYear      = dt.day_of_year;
         m_dayStartEquity = eq;
         m_tradesToday    = 0;
         m_haltedForDay   = false;
        }

      if(m_cooldownLeft > 0)
         m_cooldownLeft--;

      //--- daily stop
      double dayPnlPct = SafeDiv(eq - m_dayStartEquity, m_dayStartEquity) * 100.0;
      if(dayPnlPct <= -m_maxDailyLossPct)
         m_haltedForDay = true;

      //--- account level circuit breaker
      double ddPct = SafeDiv(m_equityPeak - eq, m_equityPeak) * 100.0;
      if(ddPct >= m_maxDrawdownPct)
        {
         if(!m_haltedForGood)
            PrintFormat("NASML RISK: max drawdown %.2f%% reached - trading disabled", ddPct);
         m_haltedForGood = true;
        }
     }

   //+---------------------------------------------------------------+
   //| Full pre-trade gate.                                          |
   //+---------------------------------------------------------------+
   bool              CanTrade(const datetime now, const double atr)
     {
      m_lastBlock = NASML_OK;

      if(m_haltedForGood)   { m_lastBlock = NASML_BLOCK_MAX_DD;         return(false); }
      if(m_haltedForDay)    { m_lastBlock = NASML_BLOCK_DAILY_LOSS;     return(false); }
      if(m_cooldownLeft > 0){ m_lastBlock = NASML_BLOCK_COOLDOWN;       return(false); }

      if(m_tradesToday >= m_maxTradesPerDay)
        { m_lastBlock = NASML_BLOCK_TRADES_PER_DAY; return(false); }

      if(m_useSession)
        {
         MqlDateTime dt;
         TimeToStruct(now, dt);
         bool inSession = (m_sessStartHour <= m_sessEndHour)
                          ? (dt.hour >= m_sessStartHour && dt.hour < m_sessEndHour)
                          : (dt.hour >= m_sessStartHour || dt.hour < m_sessEndHour);
         if(!inSession)
           { m_lastBlock = NASML_BLOCK_SESSION; return(false); }
        }

      if(!SpreadOk(atr))
        { m_lastBlock = NASML_BLOCK_SPREAD; return(false); }

      return(true);
     }

   //+---------------------------------------------------------------+
   //| Spread filter: absolute cap plus a cap relative to ATR, which |
   //| is what actually matters on an index CFD.                     |
   //+---------------------------------------------------------------+
   bool              SpreadOk(const double atr) const
     {
      double point  = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
      double spread = (double)SymbolInfoInteger(m_symbol, SYMBOL_SPREAD);
      if(spread <= 0.0)
        {
         double ask = SymbolInfoDouble(m_symbol, SYMBOL_ASK);
         double bid = SymbolInfoDouble(m_symbol, SYMBOL_BID);
         spread = SafeDiv(ask - bid, point);
        }
      if(spread > m_maxSpreadPoints)
         return(false);
      if(atr > 0.0 && spread * point > m_maxSpreadAtrFrac * atr)
         return(false);
      return(true);
     }

   //+---------------------------------------------------------------+
   //| Volume from the money at risk and the distance to the stop.   |
   //|                                                               |
   //|   lots = (equity * risk%) / (stopDistance * valuePerPriceUnit)|
   //+---------------------------------------------------------------+
   double            CalcLots(const double stopDistancePrice, const double confidenceScale)
     {
      if(stopDistancePrice <= 0.0)
         return(0.0);

      double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
      double riskMoney = equity * (m_riskPct / 100.0) * Clamp(confidenceScale, 0.25, 2.0);

      double tickValue = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_VALUE);
      double tickSize  = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE);
      if(tickSize <= 0.0)
         tickSize = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
      if(tickValue <= 0.0 || tickSize <= 0.0)
        {
         Print("NASML RISK: broker did not report tick value/size - cannot size position");
         return(0.0);
        }

      double valuePerPriceUnit = tickValue / tickSize;      // account ccy per 1.0 move, 1 lot
      double lots = SafeDiv(riskMoney, stopDistancePrice * valuePerPriceUnit);

      return(NormalizeLots(lots));
     }

   double            NormalizeLots(const double raw) const
     {
      double minLot  = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
      double maxLot  = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MAX);
      double stepLot = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
      if(stepLot <= 0.0)
         stepLot = 0.01;

      double lots = MathFloor(raw / stepLot) * stepLot;
      lots = MathMin(lots, MathMin(maxLot, m_maxLots));

      if(lots < minLot)
         return(0.0);          // risk budget too small for this symbol - skip the trade

      //--- round away binary noise introduced by the division
      int digits = (int)MathMax(0, MathCeil(-MathLog(stepLot) / MathLog(10.0)));
      return(NormalizeDouble(lots, digits));
     }

   //+---------------------------------------------------------------+
   //| Margin sanity check before sending the order.                 |
   //+---------------------------------------------------------------+
   bool              MarginOk(const ENUM_ORDER_TYPE type, const double lots, const double price)
     {
      double margin = 0.0;
      if(!OrderCalcMargin(type, m_symbol, lots, price, margin))
         return(true);         // broker refused to quote - let the server decide
      double free = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
      if(margin > free * 0.5)
        {
         m_lastBlock = NASML_BLOCK_MARGIN;
         return(false);
        }
      return(true);
     }

   //--- trade bookkeeping -----------------------------------------
   void              RegisterEntry(void) { m_tradesToday++; }

   void              RegisterResult(const double profit)
     {
      if(profit < 0.0)
        {
         m_lossStreak++;
         if(m_lossStreak >= m_lossStreakTrigger)
           {
            m_cooldownLeft = m_cooldownBars;
            m_lossStreak   = 0;
            PrintFormat("NASML RISK: loss streak hit - pausing for %d bars", m_cooldownBars);
           }
        }
      else
         m_lossStreak = 0;
     }
  };

#endif // __NAS100ML_RISKMANAGER_MQH__
//+------------------------------------------------------------------+

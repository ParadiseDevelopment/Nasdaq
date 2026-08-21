//+------------------------------------------------------------------+
//|                                                 TradeExecutor.mqh|
//|  Order placement and open-position management.                   |
//+------------------------------------------------------------------+
#ifndef __NAS100ML_TRADEEXECUTOR_MQH__
#define __NAS100ML_TRADEEXECUTOR_MQH__

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include "Utils.mqh"

//+------------------------------------------------------------------+
class CTradeExecutor
  {
private:
   CTrade            m_trade;
   CPositionInfo     m_pos;
   string            m_symbol;
   long              m_magic;
   ulong             m_lastDeal;

   double            m_beTriggerR;    // move to break even after this many R
   double            m_beOffsetR;     // where break even sits, in R
   double            m_trailAtr;      // trailing distance in ATR units
   double            m_trailStartR;   // start trailing after this many R

public:
                     CTradeExecutor(void) : m_symbol(""), m_magic(0), m_lastDeal(0),
                                            m_beTriggerR(1.0), m_beOffsetR(0.1),
                                            m_trailAtr(2.0), m_trailStartR(1.5) {}

   bool              Init(const string symbol, const long magic, const ulong slippagePoints)
     {
      m_symbol = symbol;
      m_magic  = magic;
      m_trade.SetExpertMagicNumber((ulong)magic);
      m_trade.SetDeviationInPoints(slippagePoints);
      m_trade.SetTypeFillingBySymbol(symbol);
      m_trade.SetAsyncMode(false);
      m_trade.LogLevel(LOG_LEVEL_ERRORS);

      //--- start the deal cursor at "now" so a fresh attach does not
      //--- replay months of unrelated history
      if(HistorySelect(0, TimeCurrent()))
        {
         int total = HistoryDealsTotal();
         if(total > 0)
            m_lastDeal = HistoryDealGetTicket(total - 1);
        }
      return(true);
     }

   void              ConfigureExits(const double beTriggerR, const double beOffsetR,
                                    const double trailAtr, const double trailStartR)
     {
      m_beTriggerR  = beTriggerR;
      m_beOffsetR   = beOffsetR;
      m_trailAtr    = trailAtr;
      m_trailStartR = trailStartR;
     }

   //+---------------------------------------------------------------+
   //| Position inspection                                           |
   //+---------------------------------------------------------------+
   bool              HasPosition(void)
     {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         if(!m_pos.SelectByIndex(i))
            continue;
         if(m_pos.Symbol() == m_symbol && m_pos.Magic() == m_magic)
            return(true);
        }
      return(false);
     }

   int               PositionDirection(void)
     {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         if(!m_pos.SelectByIndex(i))
            continue;
         if(m_pos.Symbol() != m_symbol || m_pos.Magic() != m_magic)
            continue;
         return(m_pos.PositionType() == POSITION_TYPE_BUY ? 1 : -1);
        }
      return(0);
     }

   //+---------------------------------------------------------------+
   //| Market entry with attached stop and target.                   |
   //+---------------------------------------------------------------+
   bool              Open(const bool isLong, const double lots, const double slPrice,
                          const double tpPrice, const string comment)
     {
      int    digits = (int)SymbolInfoInteger(m_symbol, SYMBOL_DIGITS);
      double price  = isLong ? SymbolInfoDouble(m_symbol, SYMBOL_ASK)
                             : SymbolInfoDouble(m_symbol, SYMBOL_BID);
      if(price <= 0.0)
         return(false);

      double sl = NormalizeDouble(slPrice, digits);
      double tp = NormalizeDouble(tpPrice, digits);

      //--- respect the broker's minimum stop distance
      double point   = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
      double stopLvl = (double)SymbolInfoInteger(m_symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
      //--- push the levels out to the broker's minimum distance, but leave a
      //--- level of 0 alone: 0 means "no stop"/"no target", and clamping it
      //--- would silently attach one the caller never asked for
      if(stopLvl > 0.0)
        {
         if(isLong)
           {
            if(sl > 0.0) sl = MathMin(sl, NormalizeDouble(price - stopLvl, digits));
            if(tp > 0.0) tp = MathMax(tp, NormalizeDouble(price + stopLvl, digits));
           }
         else
           {
            if(sl > 0.0) sl = MathMax(sl, NormalizeDouble(price + stopLvl, digits));
            if(tp > 0.0) tp = MathMin(tp, NormalizeDouble(price - stopLvl, digits));
           }
        }

      bool ok = isLong ? m_trade.Buy(lots, m_symbol, 0.0, sl, tp, comment)
                       : m_trade.Sell(lots, m_symbol, 0.0, sl, tp, comment);

      if(!ok)
         PrintFormat("NASML: order failed retcode=%d (%s)",
                     m_trade.ResultRetcode(), m_trade.ResultRetcodeDescription());
      return(ok);
     }

   bool              CloseAll(const string reason = "")
     {
      bool any = false;
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         if(!m_pos.SelectByIndex(i))
            continue;
         if(m_pos.Symbol() != m_symbol || m_pos.Magic() != m_magic)
            continue;
         if(m_trade.PositionClose(m_pos.Ticket()))
            any = true;
        }
      if(any && reason != "")
         Print("NASML: positions closed - " + reason);
      return(any);
     }

   //+---------------------------------------------------------------+
   //| Break-even lift then ATR trailing stop. Called on every tick. |
   //| R is measured from the original stop distance recorded in the |
   //| position comment-free way: we reconstruct it from entry - SL  |
   //| while the stop is still the initial one, so once the stop has |
   //| moved we trail on ATR alone.                                  |
   //+---------------------------------------------------------------+
   void              ManageOpen(const double atr, const double initialRiskPrice)
     {
      if(atr <= 0.0 || initialRiskPrice <= 0.0)
         return;

      int    digits = (int)SymbolInfoInteger(m_symbol, SYMBOL_DIGITS);
      double point  = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
      double stopLvl = (double)SymbolInfoInteger(m_symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;

      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         if(!m_pos.SelectByIndex(i))
            continue;
         if(m_pos.Symbol() != m_symbol || m_pos.Magic() != m_magic)
            continue;

         bool   isLong = (m_pos.PositionType() == POSITION_TYPE_BUY);
         double entry  = m_pos.PriceOpen();
         double curSL  = m_pos.StopLoss();
         double tp     = m_pos.TakeProfit();
         double last   = isLong ? SymbolInfoDouble(m_symbol, SYMBOL_BID)
                                : SymbolInfoDouble(m_symbol, SYMBOL_ASK);
         if(last <= 0.0)
            continue;

         double moved = isLong ? (last - entry) : (entry - last);
         double rMult = SafeDiv(moved, initialRiskPrice);

         //--- seed the running stop so MathMax/MathMin behave when the
         //--- position happens to carry no stop yet
         bool   hasSL = (curSL > 0.0);
         double newSL = hasSL ? curSL : (isLong ? 0.0 : DBL_MAX);

         //--- 1) lift to break even
         if(rMult >= m_beTriggerR)
           {
            double be = isLong ? entry + m_beOffsetR * initialRiskPrice
                               : entry - m_beOffsetR * initialRiskPrice;
            newSL = isLong ? MathMax(newSL, be) : MathMin(newSL, be);
           }

         //--- 2) ATR trail once the trade is genuinely in profit
         if(rMult >= m_trailStartR)
           {
            double trail = isLong ? last - m_trailAtr * atr
                                  : last + m_trailAtr * atr;
            newSL = isLong ? MathMax(newSL, trail) : MathMin(newSL, trail);
           }

         if(newSL == DBL_MAX || newSL <= 0.0)
            continue;
         if(MathAbs(newSL - curSL) < point)
            continue;

         //--- never move the stop against the position
         if(isLong  && hasSL && newSL <= curSL) continue;
         if(!isLong && hasSL && newSL >= curSL) continue;

         //--- broker distance rules
         if(stopLvl > 0.0)
           {
            if(isLong  && newSL > last - stopLvl) continue;
            if(!isLong && newSL < last + stopLvl) continue;
           }

         newSL = NormalizeDouble(newSL, digits);
         if(!m_trade.PositionModify(m_pos.Ticket(), newSL, tp))
            PrintFormat("NASML: trail modify failed retcode=%d", m_trade.ResultRetcode());
        }
     }

   //+---------------------------------------------------------------+
   //| Realised P/L of deals closed since the previous call.         |
   //| Returns the number of closing deals found.                    |
   //+---------------------------------------------------------------+
   int               PollClosedTrades(double &totalProfit)
     {
      totalProfit = 0.0;
      int found   = 0;

      datetime from = TimeCurrent() - 30 * 24 * 60 * 60;
      if(!HistorySelect(from, TimeCurrent() + 60))
         return(0);

      int total = HistoryDealsTotal();
      ulong maxSeen = m_lastDeal;

      for(int i = 0; i < total; i++)
        {
         ulong ticket = HistoryDealGetTicket(i);
         if(ticket == 0 || ticket <= m_lastDeal)
            continue;
         if(ticket > maxSeen)
            maxSeen = ticket;

         if(HistoryDealGetString(ticket, DEAL_SYMBOL) != m_symbol)
            continue;
         if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != m_magic)
            continue;
         if((ENUM_DEAL_ENTRY)HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT)
            continue;

         totalProfit += HistoryDealGetDouble(ticket, DEAL_PROFIT)
                      + HistoryDealGetDouble(ticket, DEAL_SWAP)
                      + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
         found++;
        }

      m_lastDeal = maxSeen;
      return(found);
     }
  };

#endif // __NAS100ML_TRADEEXECUTOR_MQH__
//+------------------------------------------------------------------+

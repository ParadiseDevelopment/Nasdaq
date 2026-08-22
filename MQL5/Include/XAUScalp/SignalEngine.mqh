//+------------------------------------------------------------------+
//|                                                 SignalEngine.mqh |
//|  The two scalping mechanisms, computed from CLOSED bars only.     |
//|                                                                  |
//|  BREAKOUT - continuation. Price closes beyond an N-bar extreme on |
//|             an expansion bar, with the fast/slow EMA stack and    |
//|             the higher timeframe agreeing, ADX confirming that    |
//|             there is a trend to continue, and the move not yet    |
//|             stretched too far from the fast EMA to have room.     |
//|                                                                  |
//|  FADE     - mean reversion. Price closes outside a Bollinger band |
//|             with RSI at an extreme and a large ATR-normalised     |
//|             stretch from the fast EMA, while ADX says the market  |
//|             is ranging, and (optionally) only in the direction    |
//|             the higher timeframe is already pointing.             |
//|                                                                  |
//|  Every quantity is normalised by ATR, so the same parameters mean |
//|  the same thing on a quiet Asian session and a CPI print - and on |
//|  the weekend book, where ranges are a fraction of weekday ones.   |
//|                                                                  |
//|  The engine returns a direction and a stop DISTANCE. It never     |
//|  invents an entry price: the EA anchors stop and target on the    |
//|  real fill, so a slipped entry keeps its intended R.              |
//+------------------------------------------------------------------+
#ifndef __XAUSCALP_SIGNALENGINE_MQH__
#define __XAUSCALP_SIGNALENGINE_MQH__

enum ENUM_XS_MODE
  {
   XS_MODE_BREAKOUT = 0,  // continuation only
   XS_MODE_FADE     = 1,  // mean reversion only
   XS_MODE_BOTH     = 2   // whichever fires (breakout is tested first)
  };

//+------------------------------------------------------------------+
struct SXSParams
  {
   int               mode;
   //--- indicator periods
   int               atrPeriod;
   int               rsiPeriod;
   int               adxPeriod;
   int               emaFast;
   int               emaSlow;
   int               bbPeriod;
   double            bbDev;
   int               donchian;
   int               htfEmaPeriod;
   bool              useHtf;
   //--- breakout
   double            expansionAtr;
   double            bodyFrac;
   double            minAdx;
   double            maxStretchAtr;
   double            boStopAtr;
   double            boTargetR;
   //--- fade
   double            fadeRsi;
   double            fadeStretchAtr;
   double            fadeMaxAdx;
   bool              fadeWithHtf;
   double            fadeStopAtr;
   double            fadeTargetR;
   //--- shared
   double            maxStopAtr;
  };

//+------------------------------------------------------------------+
struct SXSSignal
  {
   int               dir;        // +1 long, -1 short, 0 nothing
   int               mode;       // the mechanism that fired
   double            atr;
   double            stopDist;   // price distance, always > 0 when dir != 0
   double            targetR;
   string            why;
  };

//+------------------------------------------------------------------+
class CSignalEngine
  {
private:
   string            m_symbol;
   ENUM_TIMEFRAMES   m_tf;
   ENUM_TIMEFRAMES   m_htf;
   SXSParams         m_p;

   int               m_hAtr, m_hRsi, m_hAdx, m_hFast, m_hSlow, m_hBands, m_hHtf;
   string            m_lastReject;

   bool              Val(const int handle, const int buffer, const int shift, double &v)
     {
      double b[];
      ArraySetAsSeries(b, true);
      if(handle == INVALID_HANDLE)
         return(false);
      if(CopyBuffer(handle, buffer, shift, 1, b) != 1)
         return(false);
      v = b[0];
      return(MathIsValidNumber(v));
     }

public:
                     CSignalEngine(void) : m_symbol(""), m_tf(PERIOD_M5), m_htf(PERIOD_H1),
                                           m_hAtr(INVALID_HANDLE), m_hRsi(INVALID_HANDLE),
                                           m_hAdx(INVALID_HANDLE), m_hFast(INVALID_HANDLE),
                                           m_hSlow(INVALID_HANDLE), m_hBands(INVALID_HANDLE),
                                           m_hHtf(INVALID_HANDLE), m_lastReject("") {}

   string            Symbol(void)      const { return(m_symbol); }
   string            LastReject(void)  const { return(m_lastReject); }

   //+---------------------------------------------------------------+
   bool              Init(const string symbol, const ENUM_TIMEFRAMES tf,
                          const ENUM_TIMEFRAMES htf, const SXSParams &p)
     {
      m_symbol = symbol;
      m_tf     = tf;
      m_htf    = htf;
      m_p      = p;

      m_hAtr   = iATR(symbol, tf, p.atrPeriod);
      m_hRsi   = iRSI(symbol, tf, p.rsiPeriod, PRICE_CLOSE);
      m_hAdx   = iADX(symbol, tf, p.adxPeriod);
      m_hFast  = iMA(symbol, tf, p.emaFast, 0, MODE_EMA, PRICE_CLOSE);
      m_hSlow  = iMA(symbol, tf, p.emaSlow, 0, MODE_EMA, PRICE_CLOSE);
      m_hBands = iBands(symbol, tf, p.bbPeriod, 0, p.bbDev, PRICE_CLOSE);
      m_hHtf   = p.useHtf ? iMA(symbol, htf, p.htfEmaPeriod, 0, MODE_EMA, PRICE_CLOSE)
                          : INVALID_HANDLE;

      if(m_hAtr == INVALID_HANDLE || m_hRsi == INVALID_HANDLE || m_hAdx == INVALID_HANDLE ||
         m_hFast == INVALID_HANDLE || m_hSlow == INVALID_HANDLE || m_hBands == INVALID_HANDLE)
         return(false);
      if(p.useHtf && m_hHtf == INVALID_HANDLE)
         return(false);
      return(true);
     }

   void              Release(void)
     {
      if(m_hAtr   != INVALID_HANDLE) IndicatorRelease(m_hAtr);
      if(m_hRsi   != INVALID_HANDLE) IndicatorRelease(m_hRsi);
      if(m_hAdx   != INVALID_HANDLE) IndicatorRelease(m_hAdx);
      if(m_hFast  != INVALID_HANDLE) IndicatorRelease(m_hFast);
      if(m_hSlow  != INVALID_HANDLE) IndicatorRelease(m_hSlow);
      if(m_hBands != INVALID_HANDLE) IndicatorRelease(m_hBands);
      if(m_hHtf   != INVALID_HANDLE) IndicatorRelease(m_hHtf);
      m_hAtr = m_hRsi = m_hAdx = m_hFast = m_hSlow = m_hBands = m_hHtf = INVALID_HANDLE;
     }

   //+---------------------------------------------------------------+
   //| ATR of the last closed bar, 0 when not available yet.          |
   //+---------------------------------------------------------------+
   double            Atr(void)
     {
      double v = 0.0;
      if(!Val(m_hAtr, 0, 1, v) || v <= 0.0)
         return(0.0);
      return(v);
     }

   //+---------------------------------------------------------------+
   //| Enough history for every indicator to be warm?                 |
   //+---------------------------------------------------------------+
   bool              Ready(void)
     {
      int need = m_p.donchian + 2;
      need = (int)MathMax(need, m_p.emaSlow * 3);
      need = (int)MathMax(need, m_p.bbPeriod * 3);
      return(Bars(m_symbol, m_tf) > need);
     }

   //+---------------------------------------------------------------+
   //| The whole decision. Reads bar 1 (last closed) and back.        |
   //+---------------------------------------------------------------+
   bool              Evaluate(SXSSignal &out)
     {
      out.dir      = 0;
      out.mode     = m_p.mode;
      out.atr      = 0.0;
      out.stopDist = 0.0;
      out.targetR  = 0.0;
      out.why      = "";
      m_lastReject = "";

      if(!Ready())
        {
         m_lastReject = "warming up";
         return(false);
        }

      MqlRates r[];
      ArraySetAsSeries(r, true);
      if(CopyRates(m_symbol, m_tf, 1, 1, r) != 1)
        {
         m_lastReject = "no bar data";
         return(false);
        }
      double o1 = r[0].open, h1 = r[0].high, l1 = r[0].low, c1 = r[0].close;

      double atr = 0.0, rsi = 0.0, adx = 0.0, fast = 0.0, slow = 0.0;
      double bbUp = 0.0, bbLo = 0.0;
      if(!Val(m_hAtr, 0, 1, atr) || atr <= 0.0)
        {
         m_lastReject = "no ATR";
         return(false);
        }
      if(!Val(m_hRsi, 0, 1, rsi) || !Val(m_hAdx, 0, 1, adx) ||
         !Val(m_hFast, 0, 1, fast) || !Val(m_hSlow, 0, 1, slow) ||
         !Val(m_hBands, 1, 1, bbUp) || !Val(m_hBands, 2, 1, bbLo))
        {
         m_lastReject = "indicators not ready";
         return(false);
        }
      out.atr = atr;

      //--- higher timeframe bias, from the last CLOSED higher-TF bar
      int htfBias = 0;
      if(m_p.useHtf)
        {
         double htfEma = 0.0;
         if(!Val(m_hHtf, 0, 1, htfEma))
           {
            m_lastReject = "higher timeframe not ready";
            return(false);
           }
         double htfClose = iClose(m_symbol, m_htf, 1);
         if(htfClose <= 0.0)
           {
            m_lastReject = "higher timeframe not ready";
            return(false);
           }
         htfBias = (htfClose > htfEma) ? 1 : -1;
        }

      //--- Donchian channel of the bars BEFORE the signal bar
      double hi[], lo[];
      ArraySetAsSeries(hi, true);
      ArraySetAsSeries(lo, true);
      if(CopyHigh(m_symbol, m_tf, 2, m_p.donchian, hi) != m_p.donchian ||
         CopyLow(m_symbol, m_tf, 2, m_p.donchian, lo)  != m_p.donchian)
        {
         m_lastReject = "no channel data";
         return(false);
        }
      double hh = hi[ArrayMaximum(hi)];
      double ll = lo[ArrayMinimum(lo)];

      double range = h1 - l1;
      double body  = MathAbs(c1 - o1);

      //================================================================
      //| BREAKOUT                                                     |
      //================================================================
      if(m_p.mode == XS_MODE_BREAKOUT || m_p.mode == XS_MODE_BOTH)
        {
         bool expansion = (range >= m_p.expansionAtr * atr) &&
                          (range > 0.0 && body >= m_p.bodyFrac * range);
         bool trendOk   = (adx >= m_p.minAdx);

         int dir = 0;
         if(c1 > hh && c1 > o1 && fast > slow && c1 > fast)
            dir = 1;
         else
            if(c1 < ll && c1 < o1 && fast < slow && c1 < fast)
               dir = -1;

         if(dir != 0)
           {
            if(!expansion)
               m_lastReject = "breakout bar too small";
            else
               if(!trendOk)
                  m_lastReject = StringFormat("ADX %.1f below %.1f", adx, m_p.minAdx);
               else
                  if(m_p.useHtf && htfBias != dir)
                     m_lastReject = "higher timeframe disagrees";
                  else
                     if(MathAbs(c1 - fast) > m_p.maxStretchAtr * atr)
                        m_lastReject = "already stretched from the EMA";
                     else
                       {
                        double swing = (dir > 0) ? (l1 - 0.10 * atr) : (h1 + 0.10 * atr);
                        double byAtr = (dir > 0) ? (c1 - m_p.boStopAtr * atr)
                                                 : (c1 + m_p.boStopAtr * atr);
                        double stop  = (dir > 0) ? MathMin(swing, byAtr) : MathMax(swing, byAtr);
                        double dist  = MathAbs(c1 - stop);
                        dist = MathMin(dist, m_p.maxStopAtr * atr);

                        out.dir      = dir;
                        out.mode     = XS_MODE_BREAKOUT;
                        out.stopDist = dist;
                        out.targetR  = m_p.boTargetR;
                        out.why      = StringFormat("breakout %s  adx %.0f  range %.2fATR",
                                                    dir > 0 ? "up" : "down", adx, range / atr);
                        return(true);
                       }
           }
        }

      //================================================================
      //| FADE                                                         |
      //================================================================
      if(m_p.mode == XS_MODE_FADE || m_p.mode == XS_MODE_BOTH)
        {
         int dir = 0;
         if(c1 < bbLo && rsi <= m_p.fadeRsi)
            dir = 1;
         else
            if(c1 > bbUp && rsi >= (100.0 - m_p.fadeRsi))
               dir = -1;

         if(dir != 0)
           {
            double stretch = (dir > 0) ? (fast - c1) : (c1 - fast);
            if(adx > m_p.fadeMaxAdx)
               m_lastReject = StringFormat("ADX %.1f - trending, not fading", adx);
            else
               if(stretch < m_p.fadeStretchAtr * atr)
                  m_lastReject = "not stretched enough to fade";
               else
                  if(m_p.useHtf && m_p.fadeWithHtf && htfBias != dir)
                     m_lastReject = "fade against the higher timeframe";
                  else
                    {
                     double swing = (dir > 0) ? (l1 - 0.10 * atr) : (h1 + 0.10 * atr);
                     double byAtr = (dir > 0) ? (c1 - m_p.fadeStopAtr * atr)
                                              : (c1 + m_p.fadeStopAtr * atr);
                     double stop  = (dir > 0) ? MathMin(swing, byAtr) : MathMax(swing, byAtr);
                     double dist  = MathAbs(c1 - stop);
                     dist = MathMin(dist, m_p.maxStopAtr * atr);

                     out.dir      = dir;
                     out.mode     = XS_MODE_FADE;
                     out.stopDist = dist;
                     out.targetR  = m_p.fadeTargetR;
                     out.why      = StringFormat("fade %s  rsi %.0f  stretch %.2fATR",
                                                 dir > 0 ? "up" : "down", rsi, stretch / atr);
                     return(true);
                    }
           }
        }

      if(m_lastReject == "")
         m_lastReject = "no setup";
      return(false);
     }
  };

#endif // __XAUSCALP_SIGNALENGINE_MQH__
//+------------------------------------------------------------------+

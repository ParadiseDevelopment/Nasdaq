//+------------------------------------------------------------------+
//|                                                FeatureEngine.mqh |
//|  Causal feature extraction for NAS100 intraday bars.             |
//|                                                                  |
//|  Everything is computed from CLOSED bars only (shift >= 1) and   |
//|  every price-derived quantity is normalised by ATR so the same   |
//|  model stays valid whether the index trades at 12,000 or 25,000. |
//+------------------------------------------------------------------+
#ifndef __NAS100ML_FEATUREENGINE_MQH__
#define __NAS100ML_FEATUREENGINE_MQH__

#include "Utils.mqh"

#define NASML_FEATURE_COUNT 40
#define NASML_RATES_DEPTH   48

//+------------------------------------------------------------------+
//| CFeatureEngine                                                   |
//+------------------------------------------------------------------+
class CFeatureEngine
  {
private:
   string            m_symbol;
   ENUM_TIMEFRAMES   m_tf;

   int               m_hATR14;
   int               m_hATR50;
   int               m_hRSI;
   int               m_hEMA20;
   int               m_hEMA50;
   int               m_hEMA200;
   int               m_hMACD;
   int               m_hBands;
   int               m_hStoch;
   int               m_hADX;

   //--- session boundaries in broker server hours
   int               m_londonStart, m_londonEnd;
   int               m_nyStart,     m_nyEnd;

   double            m_lastATR;     // ATR(14) of the last computed bar

   bool              CopyOne(const int handle, const int buffer, const int count, double &dst[]) const
     {
      ArraySetAsSeries(dst, true);
      if(handle == INVALID_HANDLE)
         return(false);
      int copied = CopyBuffer(handle, buffer, 1, count, dst);
      return(copied == count);
     }

public:
                     CFeatureEngine(void) :
                     m_symbol(""), m_tf(PERIOD_CURRENT),
                     m_hATR14(INVALID_HANDLE), m_hATR50(INVALID_HANDLE),
                     m_hRSI(INVALID_HANDLE), m_hEMA20(INVALID_HANDLE),
                     m_hEMA50(INVALID_HANDLE), m_hEMA200(INVALID_HANDLE),
                     m_hMACD(INVALID_HANDLE), m_hBands(INVALID_HANDLE),
                     m_hStoch(INVALID_HANDLE), m_hADX(INVALID_HANDLE),
                     m_londonStart(8), m_londonEnd(17),
                     m_nyStart(14), m_nyEnd(23),
                     m_lastATR(0.0) {}

                    ~CFeatureEngine(void) { Release(); }

   //+---------------------------------------------------------------+
   //| Create every indicator handle. Returns false if any fails.    |
   //+---------------------------------------------------------------+
   bool              Init(const string symbol, const ENUM_TIMEFRAMES tf)
     {
      m_symbol = symbol;
      m_tf     = tf;

      m_hATR14  = iATR(symbol, tf, 14);
      m_hATR50  = iATR(symbol, tf, 50);
      m_hRSI    = iRSI(symbol, tf, 14, PRICE_CLOSE);
      m_hEMA20  = iMA(symbol, tf, 20,  0, MODE_EMA, PRICE_CLOSE);
      m_hEMA50  = iMA(symbol, tf, 50,  0, MODE_EMA, PRICE_CLOSE);
      m_hEMA200 = iMA(symbol, tf, 200, 0, MODE_EMA, PRICE_CLOSE);
      m_hMACD   = iMACD(symbol, tf, 12, 26, 9, PRICE_CLOSE);
      m_hBands  = iBands(symbol, tf, 20, 0, 2.0, PRICE_CLOSE);
      m_hStoch  = iStochastic(symbol, tf, 14, 3, 3, MODE_SMA, STO_LOWHIGH);
      m_hADX    = iADX(symbol, tf, 14);

      if(m_hATR14 == INVALID_HANDLE || m_hATR50 == INVALID_HANDLE ||
         m_hRSI   == INVALID_HANDLE || m_hEMA20 == INVALID_HANDLE ||
         m_hEMA50 == INVALID_HANDLE || m_hEMA200 == INVALID_HANDLE ||
         m_hMACD  == INVALID_HANDLE || m_hBands == INVALID_HANDLE ||
         m_hStoch == INVALID_HANDLE || m_hADX   == INVALID_HANDLE)
        {
         Print("FeatureEngine: failed to create one or more indicator handles");
         return(false);
        }
      return(true);
     }

   void              Release(void)
     {
      if(m_hATR14  != INVALID_HANDLE) { IndicatorRelease(m_hATR14);  m_hATR14  = INVALID_HANDLE; }
      if(m_hATR50  != INVALID_HANDLE) { IndicatorRelease(m_hATR50);  m_hATR50  = INVALID_HANDLE; }
      if(m_hRSI    != INVALID_HANDLE) { IndicatorRelease(m_hRSI);    m_hRSI    = INVALID_HANDLE; }
      if(m_hEMA20  != INVALID_HANDLE) { IndicatorRelease(m_hEMA20);  m_hEMA20  = INVALID_HANDLE; }
      if(m_hEMA50  != INVALID_HANDLE) { IndicatorRelease(m_hEMA50);  m_hEMA50  = INVALID_HANDLE; }
      if(m_hEMA200 != INVALID_HANDLE) { IndicatorRelease(m_hEMA200); m_hEMA200 = INVALID_HANDLE; }
      if(m_hMACD   != INVALID_HANDLE) { IndicatorRelease(m_hMACD);   m_hMACD   = INVALID_HANDLE; }
      if(m_hBands  != INVALID_HANDLE) { IndicatorRelease(m_hBands);  m_hBands  = INVALID_HANDLE; }
      if(m_hStoch  != INVALID_HANDLE) { IndicatorRelease(m_hStoch);  m_hStoch  = INVALID_HANDLE; }
      if(m_hADX    != INVALID_HANDLE) { IndicatorRelease(m_hADX);    m_hADX    = INVALID_HANDLE; }
     }

   void              SetSessions(const int lonStart, const int lonEnd, const int nyS, const int nyE)
     {
      m_londonStart = lonStart;
      m_londonEnd   = lonEnd;
      m_nyStart     = nyS;
      m_nyEnd       = nyE;
     }

   int               FeatureCount(void) const { return(NASML_FEATURE_COUNT); }
   double            LastATR(void) const { return(m_lastATR); }

   //+---------------------------------------------------------------+
   //| Build the feature vector for the most recently CLOSED bar.    |
   //+---------------------------------------------------------------+
   bool              Compute(double &f[])
     {
      ArrayResize(f, NASML_FEATURE_COUNT);
      ArrayInitialize(f, 0.0);

      if(Bars(m_symbol, m_tf) < 260)
         return(false);

      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      if(CopyRates(m_symbol, m_tf, 1, NASML_RATES_DEPTH, rates) != NASML_RATES_DEPTH)
         return(false);

      double atr14[], atr50[], rsi[], ema20[], ema50[], ema200[];
      double macdMain[], macdSig[], bbUp[], bbLo[], stoK[], stoD[];
      double adx[], diPlus[], diMinus[];

      if(!CopyOne(m_hATR14, 0,  3, atr14))    return(false);
      if(!CopyOne(m_hATR50, 0,  2, atr50))    return(false);
      if(!CopyOne(m_hRSI,   0,  6, rsi))      return(false);
      if(!CopyOne(m_hEMA20, 0,  8, ema20))    return(false);
      if(!CopyOne(m_hEMA50, 0, 12, ema50))    return(false);
      if(!CopyOne(m_hEMA200,0,  2, ema200))   return(false);
      if(!CopyOne(m_hMACD,  0,  3, macdMain)) return(false);
      if(!CopyOne(m_hMACD,  1,  3, macdSig))  return(false);
      if(!CopyOne(m_hBands, 1,  2, bbUp))     return(false);
      if(!CopyOne(m_hBands, 2,  2, bbLo))     return(false);
      if(!CopyOne(m_hStoch, 0,  2, stoK))     return(false);
      if(!CopyOne(m_hStoch, 1,  2, stoD))     return(false);
      if(!CopyOne(m_hADX,   0,  2, adx))      return(false);
      if(!CopyOne(m_hADX,   1,  2, diPlus))   return(false);
      if(!CopyOne(m_hADX,   2,  2, diMinus))  return(false);

      double atr = atr14[0];
      if(!MathIsValidNumber(atr) || atr <= 0.0)
         return(false);
      m_lastATR = atr;

      double c0 = rates[0].close;   // last closed bar
      double o0 = rates[0].open;
      double h0 = rates[0].high;
      double l0 = rates[0].low;
      double rng = h0 - l0;

      //--- 0..5  multi-horizon momentum, ATR normalised -------------
      f[0] = SafeDiv(c0 - rates[1].close,  atr);
      f[1] = SafeDiv(c0 - rates[2].close,  atr);
      f[2] = SafeDiv(c0 - rates[3].close,  atr);
      f[3] = SafeDiv(c0 - rates[5].close,  atr);
      f[4] = SafeDiv(c0 - rates[8].close,  atr);
      f[5] = SafeDiv(c0 - rates[13].close, atr);

      //--- 6..9  oscillators ---------------------------------------
      f[6] = (rsi[0] - 50.0) / 50.0;
      f[7] = (rsi[0] - rsi[3]) / 50.0;
      f[8] = (stoK[0] - 50.0) / 50.0;
      f[9] = (stoK[0] - stoD[0]) / 50.0;

      //--- 10..13 trend / band structure ---------------------------
      f[10] = SafeDiv(macdMain[0] - macdSig[0], atr);
      f[11] = SafeDiv(macdMain[0], atr);
      double bw = bbUp[0] - bbLo[0];
      f[12] = Clamp(SafeDiv(c0 - bbLo[0], bw, 0.5) * 2.0 - 1.0, -3.0, 3.0);
      f[13] = SafeDiv(bw, atr) - 4.0;

      //--- 14..19 distance from and slope of moving averages -------
      f[14] = SafeDiv(c0 - ema20[0],  atr);
      f[15] = SafeDiv(c0 - ema50[0],  atr);
      f[16] = SafeDiv(c0 - ema200[0], atr);
      f[17] = SafeDiv(ema20[0] - ema20[5],  atr);
      f[18] = SafeDiv(ema50[0] - ema50[10], atr);
      f[19] = SafeDiv(ema20[0] - ema50[0],  atr);

      //--- 20..21 volatility regime --------------------------------
      f[20] = SafeDiv(atr, atr50[0], 1.0) - 1.0;
      f[21] = SafeDiv(atr, c0) * 1000.0;

      //--- 22..25 candle anatomy -----------------------------------
      f[22] = SafeDiv(c0 - o0, rng);
      f[23] = SafeDiv(h0 - MathMax(o0, c0), rng);
      f[24] = SafeDiv(MathMin(o0, c0) - l0, rng);
      f[25] = SafeDiv(rng, atr) - 1.0;

      //--- 26 tick volume relative to its own 20 bar average -------
      double volSum = 0.0;
      for(int i = 1; i <= 20; i++)
         volSum += (double)rates[i].tick_volume;
      f[26] = Clamp(SafeDiv((double)rates[0].tick_volume, volSum / 20.0, 1.0) - 1.0, -3.0, 5.0);

      //--- 27..28 directional strength -----------------------------
      f[27] = (adx[0] - 25.0) / 25.0;
      f[28] = (diPlus[0] - diMinus[0]) / 50.0;

      //--- 29 signed streak of consecutive same-direction closes ----
      int streak = 0;
      bool up = (rates[0].close > rates[1].close);
      for(int i = 0; i < 10; i++)
        {
         bool ui = (rates[i].close > rates[i + 1].close);
         if(ui != up)
            break;
         streak++;
        }
      f[29] = (up ? streak : -streak) / 5.0;

      //--- 30..32 previous-day reference levels --------------------
      double pdh = iHigh(m_symbol,  PERIOD_D1, 1);
      double pdl = iLow(m_symbol,   PERIOD_D1, 1);
      double pdc = iClose(m_symbol, PERIOD_D1, 1);
      if(pdh > 0.0 && pdl > 0.0 && pdc > 0.0)
        {
         f[30] = Clamp(SafeDiv(c0 - pdh, atr), -20.0, 20.0);
         f[31] = Clamp(SafeDiv(c0 - pdl, atr), -20.0, 20.0);
         f[32] = Clamp(SafeDiv(c0 - pdc, atr), -20.0, 20.0);
        }

      //--- 33..39 calendar and session context ---------------------
      MqlDateTime dt;
      TimeToStruct(rates[0].time, dt);
      double hourFrac = dt.hour + dt.min / 60.0;

      f[33] = MathSin(2.0 * M_PI * hourFrac / 24.0);
      f[34] = MathCos(2.0 * M_PI * hourFrac / 24.0);
      f[35] = MathSin(2.0 * M_PI * dt.day_of_week / 7.0);
      f[36] = MathCos(2.0 * M_PI * dt.day_of_week / 7.0);

      bool inLondon = (dt.hour >= m_londonStart && dt.hour < m_londonEnd);
      bool inNY     = (dt.hour >= m_nyStart     && dt.hour < m_nyEnd);
      bool nyOpen   = (hourFrac >= m_nyStart && hourFrac < m_nyStart + 1.5);

      f[37] = inLondon ? 1.0 : 0.0;
      f[38] = inNY     ? 1.0 : 0.0;
      f[39] = nyOpen   ? 1.0 : 0.0;

      for(int i = 0; i < NASML_FEATURE_COUNT; i++)
        {
         if(!MathIsValidNumber(f[i]))
            f[i] = 0.0;
         f[i] = Clamp(f[i], -50.0, 50.0);
        }
      return(true);
     }

   //+---------------------------------------------------------------+
   //| Column headers used by the CSV dataset export.                |
   //+---------------------------------------------------------------+
   static string     FeatureName(const int i)
     {
      switch(i)
        {
         case  0: return("ret1");        case  1: return("ret2");
         case  2: return("ret3");        case  3: return("ret5");
         case  4: return("ret8");        case  5: return("ret13");
         case  6: return("rsi");         case  7: return("rsi_slope");
         case  8: return("stoch_k");     case  9: return("stoch_kd");
         case 10: return("macd_hist");   case 11: return("macd_line");
         case 12: return("bb_pctb");     case 13: return("bb_width");
         case 14: return("d_ema20");     case 15: return("d_ema50");
         case 16: return("d_ema200");    case 17: return("ema20_slope");
         case 18: return("ema50_slope"); case 19: return("ema_spread");
         case 20: return("atr_ratio");   case 21: return("atr_norm");
         case 22: return("body");        case 23: return("upper_wick");
         case 24: return("lower_wick");  case 25: return("range_exp");
         case 26: return("vol_ratio");   case 27: return("adx");
         case 28: return("di_diff");     case 29: return("streak");
         case 30: return("dist_pdh");    case 31: return("dist_pdl");
         case 32: return("dist_pdc");    case 33: return("hour_sin");
         case 34: return("hour_cos");    case 35: return("dow_sin");
         case 36: return("dow_cos");     case 37: return("sess_london");
         case 38: return("sess_ny");     case 39: return("sess_nyopen");
        }
      return("f" + IntegerToString(i));
     }
  };

#endif // __NAS100ML_FEATUREENGINE_MQH__
//+------------------------------------------------------------------+

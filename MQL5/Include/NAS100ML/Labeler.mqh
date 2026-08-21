//+------------------------------------------------------------------+
//|                                                      Labeler.mqh |
//|  Triple-barrier labelling.                                       |
//|                                                                  |
//|  When a bar closes we snapshot its feature vector and open a     |
//|  pending sample with a profit barrier at +k*ATR, a loss barrier  |
//|  at -k*ATR and a vertical (time) barrier H bars out. The sample  |
//|  only becomes a training example once one of those barriers is   |
//|  touched, which is exactly the question the EA has to answer at  |
//|  entry time: "which side gets hit first?".                       |
//|                                                                  |
//|  Samples closed by the time barrier carry a reduced weight - the |
//|  market gave no clear answer, so they should not push the model  |
//|  as hard as a clean barrier touch.                               |
//+------------------------------------------------------------------+
#ifndef __NAS100ML_LABELER_MQH__
#define __NAS100ML_LABELER_MQH__

#include "Utils.mqh"

//+------------------------------------------------------------------+
struct SPendingSample
  {
   datetime          bar_time;
   double            entry;
   double            atr;
   double            upper;
   double            lower;
   int               age;
   bool              active;
   double            f[NASML_MAX_FEATURES];
  };

//+------------------------------------------------------------------+
class CLabeler
  {
private:
   SPendingSample    m_pend[];
   int               m_capacity;
   int               m_nfeat;
   int               m_horizon;
   double            m_barrier;      // barrier width in ATR units
   double            m_timeWeight;   // weight for time-barrier samples

   //--- resolved samples waiting to be consumed
   double            m_resX[];       // flat, stride = m_nfeat
   double            m_resY[];
   double            m_resW[];
   int               m_resCount;

   void              PushResolved(const double &f[], const double y, const double w)
     {
      int idx = m_resCount;
      ArrayResize(m_resX, (idx + 1) * m_nfeat);
      ArrayResize(m_resY, idx + 1);
      ArrayResize(m_resW, idx + 1);
      for(int i = 0; i < m_nfeat; i++)
         m_resX[idx * m_nfeat + i] = f[i];
      m_resY[idx] = y;
      m_resW[idx] = w;
      m_resCount++;
     }

public:
                     CLabeler(void) : m_capacity(0), m_nfeat(0), m_horizon(12),
                                      m_barrier(1.0), m_timeWeight(0.5), m_resCount(0) {}

   void              Init(const int nfeat, const int horizon, const double barrierAtr,
                          const double timeBarrierWeight = 0.5)
     {
      m_nfeat      = nfeat;
      m_horizon    = MathMax(1, horizon);
      m_barrier    = MathMax(0.1, barrierAtr);
      m_timeWeight = Clamp(timeBarrierWeight, 0.05, 1.0);
      m_capacity   = m_horizon + 4;

      ArrayResize(m_pend, m_capacity);
      for(int i = 0; i < m_capacity; i++)
         m_pend[i].active = false;

      m_resCount = 0;
      ArrayResize(m_resX, 0);
      ArrayResize(m_resY, 0);
      ArrayResize(m_resW, 0);
     }

   int               Horizon(void) const { return(m_horizon); }

   //+---------------------------------------------------------------+
   //| Register the bar that just closed as a new pending sample.    |
   //+---------------------------------------------------------------+
   bool              Add(const datetime barTime, const double entry, const double atr,
                         const double &f[])
     {
      if(atr <= 0.0 || m_nfeat <= 0 || m_nfeat > NASML_MAX_FEATURES)
         return(false);

      for(int s = 0; s < m_capacity; s++)
        {
         if(m_pend[s].active)
            continue;
         m_pend[s].bar_time = barTime;
         m_pend[s].entry    = entry;
         m_pend[s].atr      = atr;
         m_pend[s].upper    = entry + m_barrier * atr;
         m_pend[s].lower    = entry - m_barrier * atr;
         m_pend[s].age      = 0;
         m_pend[s].active   = true;
         for(int i = 0; i < m_nfeat; i++)
            m_pend[s].f[i] = f[i];
         return(true);
        }
      return(false);   // queue full - should not happen with capacity = H + 4
     }

   //+---------------------------------------------------------------+
   //| Feed the newly closed bar to every pending sample.            |
   //+---------------------------------------------------------------+
   void              Update(const double high, const double low, const double close)
     {
      for(int s = 0; s < m_capacity; s++)
        {
         if(!m_pend[s].active)
            continue;

         m_pend[s].age++;

         bool hitUp = (high >= m_pend[s].upper);
         bool hitDn = (low  <= m_pend[s].lower);

         if(hitUp && hitDn)
           {
            //--- both barriers inside one bar: direction is ambiguous, fall
            //--- back to the close and halve the confidence.
            double y = (close > m_pend[s].entry) ? 1.0 : 0.0;
            PushResolved(m_pend[s].f, y, 0.5 * m_timeWeight);
            m_pend[s].active = false;
            continue;
           }
         if(hitUp)
           {
            PushResolved(m_pend[s].f, 1.0, 1.0);
            m_pend[s].active = false;
            continue;
           }
         if(hitDn)
           {
            PushResolved(m_pend[s].f, 0.0, 1.0);
            m_pend[s].active = false;
            continue;
           }
         if(m_pend[s].age >= m_horizon)
           {
            double y = (close > m_pend[s].entry) ? 1.0 : 0.0;
            PushResolved(m_pend[s].f, y, m_timeWeight);
            m_pend[s].active = false;
           }
        }
     }

   int               ResolvedCount(void) const { return(m_resCount); }

   void              GetResolved(const int idx, double &x[], double &y, double &w)
     {
      ArrayResize(x, m_nfeat);
      for(int i = 0; i < m_nfeat; i++)
         x[i] = m_resX[idx * m_nfeat + i];
      y = m_resY[idx];
      w = m_resW[idx];
     }

   void              ClearResolved(void)
     {
      m_resCount = 0;
      ArrayResize(m_resX, 0);
      ArrayResize(m_resY, 0);
      ArrayResize(m_resW, 0);
     }

   void              Reset(void)
     {
      for(int i = 0; i < m_capacity; i++)
         m_pend[i].active = false;
      ClearResolved();
     }
  };

#endif // __NAS100ML_LABELER_MQH__
//+------------------------------------------------------------------+

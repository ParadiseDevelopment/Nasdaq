//+------------------------------------------------------------------+
//|                                                 ReplayBuffer.mqh |
//|  Bounded experience replay with class-balanced sampling.         |
//|                                                                  |
//|  Pure single-pass online learning wastes information and drifts  |
//|  with the most recent regime. Replaying a handful of older       |
//|  samples after every new observation gives the models several    |
//|  effective epochs over a rolling window while keeping the memory |
//|  footprint fixed.                                                |
//+------------------------------------------------------------------+
#ifndef __NAS100ML_REPLAYBUFFER_MQH__
#define __NAS100ML_REPLAYBUFFER_MQH__

#include "Utils.mqh"

//+------------------------------------------------------------------+
class CReplayBuffer
  {
private:
   double            m_X[];       // flat, stride = m_nfeat
   double            m_Y[];
   double            m_W[];
   int               m_nfeat;
   int               m_cap;
   int               m_head;
   int               m_count;
   long              m_pos;       // running count of y == 1
   long              m_neg;

public:
                     CReplayBuffer(void) : m_nfeat(0), m_cap(0), m_head(0),
                                           m_count(0), m_pos(0), m_neg(0) {}

   void              Init(const int nfeat, const int capacity)
     {
      m_nfeat = nfeat;
      m_cap   = MathMax(16, capacity);
      ArrayResize(m_X, m_cap * m_nfeat);
      ArrayResize(m_Y, m_cap);
      ArrayResize(m_W, m_cap);
      ArrayInitialize(m_X, 0.0);
      ArrayInitialize(m_Y, 0.0);
      ArrayInitialize(m_W, 0.0);
      m_head  = 0;
      m_count = 0;
      m_pos   = 0;
      m_neg   = 0;
     }

   int               Count(void) const { return(m_count); }
   int               Capacity(void) const { return(m_cap); }

   double            PositiveShare(void) const
     {
      long tot = m_pos + m_neg;
      if(tot <= 0)
         return(0.5);
      return((double)m_pos / (double)tot);
     }

   void              Add(const double &x[], const double y, const double w)
     {
      if(m_cap <= 0)
         return;
      int base = m_head * m_nfeat;
      for(int i = 0; i < m_nfeat; i++)
         m_X[base + i] = x[i];
      m_Y[m_head] = y;
      m_W[m_head] = w;

      if(y > 0.5)
         m_pos++;
      else
         m_neg++;

      m_head = (m_head + 1) % m_cap;
      if(m_count < m_cap)
         m_count++;
     }

   //+---------------------------------------------------------------+
   //| Draw one sample. Half the draws target the minority class so  |
   //| a trending stretch cannot bias the models into a permanent    |
   //| long (or short) prior.                                        |
   //+---------------------------------------------------------------+
   bool              Sample(double &x[], double &y, double &w)
     {
      if(m_count <= 0)
         return(false);

      int idx = RandInt(m_count);

      if(RandUniform() < 0.5)
        {
         double wantPos = (PositiveShare() < 0.5) ? 1.0 : 0.0;
         for(int tries = 0; tries < 24; tries++)
           {
            int cand = RandInt(m_count);
            double yc = m_Y[cand];
            if((yc > 0.5 ? 1.0 : 0.0) == wantPos)
              {
               idx = cand;
               break;
              }
           }
        }

      ArrayResize(x, m_nfeat);
      int base = idx * m_nfeat;
      for(int i = 0; i < m_nfeat; i++)
         x[i] = m_X[base + i];
      y = m_Y[idx];
      w = m_W[idx];
      return(true);
     }

   void              Clear(void)
     {
      m_head  = 0;
      m_count = 0;
      m_pos   = 0;
      m_neg   = 0;
     }
  };

#endif // __NAS100ML_REPLAYBUFFER_MQH__
//+------------------------------------------------------------------+

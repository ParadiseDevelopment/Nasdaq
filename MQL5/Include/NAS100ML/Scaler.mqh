//+------------------------------------------------------------------+
//|                                                       Scaler.mqh |
//|  Online feature standardisation (streaming z-score, Welford).    |
//+------------------------------------------------------------------+
#ifndef __NAS100ML_SCALER_MQH__
#define __NAS100ML_SCALER_MQH__

#include "Utils.mqh"

//+------------------------------------------------------------------+
//| CScaler                                                          |
//|                                                                  |
//| Keeps a running mean/variance per feature so the models always   |
//| see roughly unit-scale inputs. Statistics are updated only from  |
//| observations that have already been consumed for training, which |
//| keeps the transform causal (no look-ahead).                      |
//|                                                                  |
//| Parameter layout (GetParams/SetParams): [ mean(n) , std(n) ]     |
//+------------------------------------------------------------------+
class CScaler
  {
private:
   int               m_n;
   double            m_mean[];
   double            m_m2[];     // sum of squared deviations
   double            m_std[];
   long              m_count;
   bool              m_frozen;   // true when parameters were loaded from file

public:
                     CScaler(void) : m_n(0), m_count(0), m_frozen(false) {}

   void              Init(const int n)
     {
      m_n = n;
      ArrayResize(m_mean, n);
      ArrayResize(m_m2,   n);
      ArrayResize(m_std,  n);
      ArrayInitialize(m_mean, 0.0);
      ArrayInitialize(m_m2,   0.0);
      ArrayInitialize(m_std,  1.0);
      m_count  = 0;
      m_frozen = false;
     }

   long              Count(void) const { return(m_count); }
   bool              IsFrozen(void) const { return(m_frozen); }
   void              Freeze(const bool v) { m_frozen = v; }

   //--- Fold one observation into the running statistics.
   void              Observe(const double &x[])
     {
      if(m_frozen)
         return;
      m_count++;
      for(int i = 0; i < m_n; i++)
        {
         double xi = x[i];
         if(!MathIsValidNumber(xi))
            xi = 0.0;
         double delta = xi - m_mean[i];
         m_mean[i] += delta / (double)m_count;
         m_m2[i]   += delta * (xi - m_mean[i]);
         if(m_count > 1)
           {
            double var = m_m2[i] / (double)(m_count - 1);
            m_std[i] = MathSqrt(MathMax(var, 1.0e-10));
           }
        }
     }

   //--- Standardise and clip to +/- 5 sigma so a single freak bar cannot
   //--- dominate a gradient step.
   void              Transform(const double &x[], double &out[])
     {
      ArrayResize(out, m_n);
      for(int i = 0; i < m_n; i++)
        {
         double xi = x[i];
         if(!MathIsValidNumber(xi))
            xi = 0.0;
         double z = (xi - m_mean[i]) / MathMax(m_std[i], 1.0e-8);
         out[i] = Clamp(z, -5.0, 5.0);
        }
     }

   int               ParamCount(void) const { return(2 * m_n); }

   void              GetParams(double &p[])
     {
      ArrayResize(p, 2 * m_n);
      for(int i = 0; i < m_n; i++)
        {
         p[i]       = m_mean[i];
         p[m_n + i] = m_std[i];
        }
     }

   bool              SetParams(const double &p[])
     {
      if(ArraySize(p) < 2 * m_n)
         return(false);
      for(int i = 0; i < m_n; i++)
        {
         m_mean[i] = p[i];
         m_std[i]  = MathMax(p[m_n + i], 1.0e-8);
         m_m2[i]   = 0.0;
        }
      m_frozen = true;
      return(true);
     }
  };

#endif // __NAS100ML_SCALER_MQH__
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                        Utils.mqh |
//|                        NAS100 Machine Learning Expert Advisor    |
//|  Small numeric helpers shared by every module.                   |
//+------------------------------------------------------------------+
#property copyright "NAS100ML"
#property link      ""
#ifndef __NAS100ML_UTILS_MQH__
#define __NAS100ML_UTILS_MQH__

//--- Hard upper bound on the feature vector. Structs use fixed arrays so
//--- they stay POD and can live inside plain MQL5 arrays.
#define NASML_MAX_FEATURES   64
#define NASML_EPS            1.0e-12

//+------------------------------------------------------------------+
//| Division that never divides by (near) zero.                      |
//+------------------------------------------------------------------+
double SafeDiv(const double a, const double b, const double fallback = 0.0)
  {
   if(MathAbs(b) < NASML_EPS)
      return(fallback);
   double r = a / b;
   if(!MathIsValidNumber(r))
      return(fallback);
   return(r);
  }

//+------------------------------------------------------------------+
//| Clamp a value into [lo, hi].                                     |
//+------------------------------------------------------------------+
double Clamp(const double v, const double lo, const double hi)
  {
   if(!MathIsValidNumber(v))
      return(0.0);
   if(v < lo)
      return(lo);
   if(v > hi)
      return(hi);
   return(v);
  }

//+------------------------------------------------------------------+
//| Numerically stable logistic function.                            |
//+------------------------------------------------------------------+
double Sigmoid(const double z)
  {
   double zz = Clamp(z, -35.0, 35.0);
   if(zz >= 0.0)
      return(1.0 / (1.0 + MathExp(-zz)));
   double e = MathExp(zz);
   return(e / (1.0 + e));
  }

//+------------------------------------------------------------------+
//| Hyperbolic tangent (own implementation for portability).         |
//+------------------------------------------------------------------+
double TanhAct(const double z)
  {
   double zz = Clamp(z, -20.0, 20.0);
   double e2 = MathExp(2.0 * zz);
   return((e2 - 1.0) / (e2 + 1.0));
  }

//+------------------------------------------------------------------+
//| Binary cross entropy for a single sample, clipped.               |
//+------------------------------------------------------------------+
double LogLoss(const double p, const double y)
  {
   double pp = Clamp(p, 1.0e-7, 1.0 - 1.0e-7);
   return(-(y * MathLog(pp) + (1.0 - y) * MathLog(1.0 - pp)));
  }

//+------------------------------------------------------------------+
//| Uniform random number in (0,1).                                  |
//+------------------------------------------------------------------+
double RandUniform()
  {
   return((MathRand() + 0.5) / 32768.0);
  }

//+------------------------------------------------------------------+
//| Standard normal deviate via Box-Muller.                          |
//+------------------------------------------------------------------+
double RandNormal()
  {
   double u1 = RandUniform();
   double u2 = RandUniform();
   return(MathSqrt(-2.0 * MathLog(u1)) * MathCos(2.0 * M_PI * u2));
  }

//+------------------------------------------------------------------+
//| Random integer in [0, n-1]. MathRand() only spans 15 bits, so we |
//| compose two draws to stay uniform over larger ranges.            |
//+------------------------------------------------------------------+
int RandInt(const int n)
  {
   if(n <= 1)
      return(0);
   int r = (MathRand() << 15) | MathRand();
   if(r < 0)
      r = -r;
   return(r % n);
  }

//+------------------------------------------------------------------+
//| Clip every element of a gradient vector to +/- limit.            |
//+------------------------------------------------------------------+
void ClipGradient(double &g[], const double limit)
  {
   int n = ArraySize(g);
   for(int i = 0; i < n; i++)
     {
      if(!MathIsValidNumber(g[i]))
        {
         g[i] = 0.0;
         continue;
        }
      g[i] = Clamp(g[i], -limit, limit);
     }
  }

//+------------------------------------------------------------------+
//| Adam / AdamW optimiser state for one flat parameter vector.      |
//+------------------------------------------------------------------+
class CAdam
  {
private:
   double            m_m[];      // first moment
   double            m_v[];      // second moment
   long              m_t;        // step counter
   double            m_beta1;
   double            m_beta2;
   double            m_eps;

public:
                     CAdam(void) : m_t(0), m_beta1(0.9), m_beta2(0.999), m_eps(1.0e-8) {}

   void              Init(const int n, const double beta1 = 0.9, const double beta2 = 0.999)
     {
      ArrayResize(m_m, n);
      ArrayResize(m_v, n);
      ArrayInitialize(m_m, 0.0);
      ArrayInitialize(m_v, 0.0);
      m_t     = 0;
      m_beta1 = beta1;
      m_beta2 = beta2;
     }

   void              Reset(void)
     {
      ArrayInitialize(m_m, 0.0);
      ArrayInitialize(m_v, 0.0);
      m_t = 0;
     }

   //--- One AdamW step. Weight decay is decoupled from the gradient.
   void              Step(double &p[], const double &g[], const double lr, const double weight_decay)
     {
      int n = ArraySize(p);
      if(ArraySize(m_m) != n)
         Init(n, m_beta1, m_beta2);

      m_t++;
      double bc1 = 1.0 - MathPow(m_beta1, (double)m_t);
      double bc2 = 1.0 - MathPow(m_beta2, (double)m_t);
      if(bc1 < NASML_EPS)
         bc1 = NASML_EPS;
      if(bc2 < NASML_EPS)
         bc2 = NASML_EPS;

      for(int i = 0; i < n; i++)
        {
         double gi = g[i];
         if(!MathIsValidNumber(gi))
            gi = 0.0;

         m_m[i] = m_beta1 * m_m[i] + (1.0 - m_beta1) * gi;
         m_v[i] = m_beta2 * m_v[i] + (1.0 - m_beta2) * gi * gi;

         double mhat = m_m[i] / bc1;
         double vhat = m_v[i] / bc2;

         double upd = lr * (mhat / (MathSqrt(vhat) + m_eps) + weight_decay * p[i]);
         if(!MathIsValidNumber(upd))
            upd = 0.0;
         p[i] -= upd;

         if(!MathIsValidNumber(p[i]))
            p[i] = 0.0;
        }
     }
  };

//+------------------------------------------------------------------+
//| Fixed size ring buffer of doubles with running mean helpers.     |
//+------------------------------------------------------------------+
class CRingStats
  {
private:
   double            m_buf[];
   int               m_cap;
   int               m_head;
   int               m_count;

public:
                     CRingStats(void) : m_cap(0), m_head(0), m_count(0) {}

   void              Init(const int capacity)
     {
      m_cap = MathMax(1, capacity);
      ArrayResize(m_buf, m_cap);
      ArrayInitialize(m_buf, 0.0);
      m_head  = 0;
      m_count = 0;
     }

   void              Push(const double v)
     {
      if(m_cap <= 0)
         Init(64);
      m_buf[m_head] = v;
      m_head = (m_head + 1) % m_cap;
      if(m_count < m_cap)
         m_count++;
     }

   int               Count(void) const { return(m_count); }
   bool              IsFull(void) const { return(m_count >= m_cap); }

   double            Mean(void) const
     {
      if(m_count <= 0)
         return(0.0);
      double s = 0.0;
      for(int i = 0; i < m_count; i++)
         s += m_buf[i];
      return(s / m_count);
     }

   double            Sum(void) const
     {
      double s = 0.0;
      for(int i = 0; i < m_count; i++)
         s += m_buf[i];
      return(s);
     }

   void              Clear(void)
     {
      ArrayInitialize(m_buf, 0.0);
      m_head  = 0;
      m_count = 0;
     }
  };

#endif // __NAS100ML_UTILS_MQH__
//+------------------------------------------------------------------+

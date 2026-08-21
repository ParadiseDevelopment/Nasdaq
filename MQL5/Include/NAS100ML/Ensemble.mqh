//+------------------------------------------------------------------+
//|                                                     Ensemble.mqh |
//|  Hedge-weighted ensemble of the three online learners.           |
//|                                                                  |
//|  Each expert keeps its own multiplicative weight, updated with   |
//|  the exponentially weighted average forecaster rule              |
//|                                                                  |
//|      w_i  <-  w_i * exp( -eta * logloss_i )                      |
//|                                                                  |
//|  so whichever model is currently describing the market best      |
//|  dominates the blend, while a floor on every weight lets a model |
//|  that fell out of favour climb back when the regime turns.       |
//+------------------------------------------------------------------+
#ifndef __NAS100ML_ENSEMBLE_MQH__
#define __NAS100ML_ENSEMBLE_MQH__

#include "Utils.mqh"
#include "Models.mqh"
#include "Scaler.mqh"
#include "ReplayBuffer.mqh"

#define NASML_N_EXPERTS 3

//+------------------------------------------------------------------+
class CEnsemble
  {
private:
   CLogisticModel    m_lin;
   CRffModel         m_rff;
   CMlpModel         m_mlp;

   double            m_w[NASML_N_EXPERTS];
   double            m_eta;
   double            m_wFloor;

   int               m_nfeat;
   long              m_trained;

   CRingStats        m_hit;      // 1 when the ensemble called the side right
   CRingStats        m_ll;       // rolling ensemble log loss
   double            m_lastP[NASML_N_EXPERTS];

   void              Normalise(void)
     {
      double s = 0.0;
      for(int i = 0; i < NASML_N_EXPERTS; i++)
        {
         if(!MathIsValidNumber(m_w[i]) || m_w[i] < m_wFloor)
            m_w[i] = m_wFloor;
         s += m_w[i];
        }
      if(s < NASML_EPS)
        {
         for(int i = 0; i < NASML_N_EXPERTS; i++)
            m_w[i] = 1.0 / NASML_N_EXPERTS;
         return;
        }
      for(int i = 0; i < NASML_N_EXPERTS; i++)
         m_w[i] /= s;
     }

public:
                     CEnsemble(void) : m_eta(0.35), m_wFloor(0.05), m_nfeat(0), m_trained(0) {}

   void              Init(const int nfeat, const double lr, const double l2,
                          const int rffDim, const double rffSigma,
                          const int mlpHidden, const int seed,
                          const double hedgeEta, const int evalWindow)
     {
      m_nfeat = nfeat;
      m_eta   = MathMax(0.0, hedgeEta);

      m_lin.Init(nfeat, lr, l2);
      m_rff.Init(nfeat, rffDim, rffSigma, lr, l2, seed + 7919);
      m_mlp.Init(nfeat, mlpHidden, lr * 0.6, l2, seed + 104729);

      for(int i = 0; i < NASML_N_EXPERTS; i++)
        {
         m_w[i]     = 1.0 / NASML_N_EXPERTS;
         m_lastP[i] = 0.5;
        }

      m_hit.Init(evalWindow);
      m_ll.Init(evalWindow);
      m_trained = 0;
     }

   long              TrainedSamples(void) const { return(m_trained); }
   double            ExpertWeight(const int i) const { return(m_w[i]); }
   double            LastExpertProb(const int i) const { return(m_lastP[i]); }

   //--- Rolling out-of-sample hit rate (each sample is scored before it
   //--- is used for training, so this is an honest walk-forward number).
   double            RollingAccuracy(void) const
     {
      if(m_hit.Count() < 20)
         return(0.5);
      return(m_hit.Mean());
     }

   double            RollingLogLoss(void) const
     {
      if(m_ll.Count() < 20)
         return(0.6931);
      return(m_ll.Mean());
     }

   int               EvalCount(void) const { return(m_hit.Count()); }

   //+---------------------------------------------------------------+
   //| Blended probability that price reaches the upper barrier      |
   //| before the lower one.                                         |
   //+---------------------------------------------------------------+
   double            Predict(const double &x[])
     {
      m_lastP[0] = m_lin.Predict(x);
      m_lastP[1] = m_rff.Predict(x);
      m_lastP[2] = m_mlp.Predict(x);

      double p = 0.0;
      for(int i = 0; i < NASML_N_EXPERTS; i++)
         p += m_w[i] * m_lastP[i];
      return(Clamp(p, 0.0, 1.0));
     }

   //+---------------------------------------------------------------+
   //| Disagreement between experts. High spread means the models    |
   //| are not describing the same market and the signal should be   |
   //| treated with suspicion.                                       |
   //+---------------------------------------------------------------+
   double            Disagreement(void) const
     {
      double mn = m_lastP[0], mx = m_lastP[0];
      for(int i = 1; i < NASML_N_EXPERTS; i++)
        {
         mn = MathMin(mn, m_lastP[i]);
         mx = MathMax(mx, m_lastP[i]);
        }
      return(mx - mn);
     }

   //+---------------------------------------------------------------+
   //| Score a labelled sample WITHOUT touching the model weights.   |
   //| This is the walk-forward step: the sample is graded on what   |
   //| the ensemble knew before it arrived, and the Hedge weights    |
   //| move accordingly.                                             |
   //+---------------------------------------------------------------+
   void              Evaluate(const double &x[], const double y)
     {
      double pLin = m_lin.Predict(x);
      double pRff = m_rff.Predict(x);
      double pMlp = m_mlp.Predict(x);

      double pEns = m_w[0] * pLin + m_w[1] * pRff + m_w[2] * pMlp;
      m_hit.Push(((pEns >= 0.5 && y > 0.5) || (pEns < 0.5 && y <= 0.5)) ? 1.0 : 0.0);
      m_ll.Push(LogLoss(pEns, y));

      if(m_eta > 0.0)
        {
         m_w[0] *= MathExp(-m_eta * LogLoss(pLin, y));
         m_w[1] *= MathExp(-m_eta * LogLoss(pRff, y));
         m_w[2] *= MathExp(-m_eta * LogLoss(pMlp, y));
         Normalise();
        }

      m_trained++;
     }

   //+---------------------------------------------------------------+
   //| One gradient step for every expert.                           |
   //+---------------------------------------------------------------+
   void              Train(const double &x[], const double y, const double weight)
     {
      if(weight <= 0.0)
         return;
      m_lin.Learn(x, y, weight);
      m_rff.Learn(x, y, weight);
      m_mlp.Learn(x, y, weight);
     }

   //+---------------------------------------------------------------+
   //| Convenience: evaluate then train on a genuinely new sample.   |
   //+---------------------------------------------------------------+
   void              Learn(const double &x[], const double y, const double weight,
                           const bool scoreFirst)
     {
      if(scoreFirst)
         Evaluate(x, y);
      Train(x, y, weight);
     }

   //--- direct access for persistence
   CLogisticModel   *Linear(void) { return(GetPointer(m_lin)); }
   CRffModel        *Rff(void)    { return(GetPointer(m_rff)); }
   CMlpModel        *Mlp(void)    { return(GetPointer(m_mlp)); }

   void              GetHedge(double &p[])
     {
      ArrayResize(p, NASML_N_EXPERTS);
      for(int i = 0; i < NASML_N_EXPERTS; i++)
         p[i] = m_w[i];
     }

   bool              SetHedge(const double &p[])
     {
      if(ArraySize(p) != NASML_N_EXPERTS)
         return(false);
      for(int i = 0; i < NASML_N_EXPERTS; i++)
         m_w[i] = p[i];
      Normalise();
      return(true);
     }
  };

#endif // __NAS100ML_ENSEMBLE_MQH__
//+------------------------------------------------------------------+

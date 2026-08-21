//+------------------------------------------------------------------+
//|                                                       Models.mqh |
//|  Three online learners that all expose the same interface:       |
//|                                                                  |
//|    CLogisticModel  - L2 regularised linear logistic regression   |
//|    CRffModel       - random Fourier features + logistic head     |
//|                      (a streaming approximation of an RBF kernel |
//|                       classifier - captures smooth non-linearity)|
//|    CMlpModel       - one hidden tanh layer, sigmoid output       |
//|                                                                  |
//|  All three are trained with AdamW on the binary cross entropy of |
//|  the triple-barrier label, and all three serialise to a flat     |
//|  double vector so weights can be saved, reloaded, or produced    |
//|  offline by tools/train_offline.py.                              |
//+------------------------------------------------------------------+
#ifndef __NAS100ML_MODELS_MQH__
#define __NAS100ML_MODELS_MQH__

#include "Utils.mqh"

//+------------------------------------------------------------------+
//| Base interface                                                   |
//+------------------------------------------------------------------+
class CModelBase
  {
protected:
   int               m_nfeat;
   double            m_lr;
   double            m_l2;

public:
                     CModelBase(void) : m_nfeat(0), m_lr(0.01), m_l2(1.0e-4) {}
   virtual          ~CModelBase(void) {}

   virtual string    Name(void) const { return("base"); }
   virtual double    Predict(const double &x[]) { return(0.5); }
   virtual void      Learn(const double &x[], const double y, const double weight) {}
   virtual int       ParamCount(void) const { return(0); }
   virtual void      GetParams(double &p[]) { ArrayResize(p, 0); }
   virtual bool      SetParams(const double &p[]) { return(false); }

   void              SetLearningRate(const double lr) { m_lr = lr; }
   void              SetL2(const double l2) { m_l2 = l2; }
  };

//+------------------------------------------------------------------+
//| Linear logistic regression                                       |
//| params: [ b , w(0..n-1) ]                                        |
//+------------------------------------------------------------------+
class CLogisticModel : public CModelBase
  {
private:
   double            m_p[];   // [0]=bias, [1..n]=weights
   double            m_g[];
   CAdam             m_adam;

public:
   virtual string    Name(void) const { return("logistic"); }

   void              Init(const int nfeat, const double lr, const double l2)
     {
      m_nfeat = nfeat;
      m_lr    = lr;
      m_l2    = l2;
      ArrayResize(m_p, nfeat + 1);
      ArrayResize(m_g, nfeat + 1);
      ArrayInitialize(m_p, 0.0);
      ArrayInitialize(m_g, 0.0);
      m_adam.Init(nfeat + 1);
     }

   virtual double    Predict(const double &x[])
     {
      double z = m_p[0];
      for(int i = 0; i < m_nfeat; i++)
         z += m_p[i + 1] * x[i];
      return(Sigmoid(z));
     }

   virtual void      Learn(const double &x[], const double y, const double weight)
     {
      double p = Predict(x);
      double d = (p - y) * weight;
      m_g[0] = d;
      for(int i = 0; i < m_nfeat; i++)
         m_g[i + 1] = d * x[i];
      ClipGradient(m_g, 5.0);
      m_adam.Step(m_p, m_g, m_lr, m_l2);
     }

   virtual int       ParamCount(void) const { return(m_nfeat + 1); }

   virtual void      GetParams(double &p[])
     {
      ArrayResize(p, ArraySize(m_p));
      ArrayCopy(p, m_p);
     }

   virtual bool      SetParams(const double &p[])
     {
      if(ArraySize(p) != m_nfeat + 1)
         return(false);
      ArrayCopy(m_p, p);
      m_adam.Reset();
      return(true);
     }
  };

//+------------------------------------------------------------------+
//| Random Fourier Features + logistic head                          |
//|                                                                  |
//|   z_j(x) = sqrt(2/D) * cos( (omega_j . x) / sigma + phi_j )      |
//|                                                                  |
//| omega and phi are drawn once and stay frozen; only the linear    |
//| head is trained. With D features this approximates an RBF kernel |
//| logistic regression at O(D) cost per bar.                        |
//|                                                                  |
//| params: [ D , sigma , omega(D*n) , phi(D) , b , w(D) ]           |
//+------------------------------------------------------------------+
class CRffModel : public CModelBase
  {
private:
   int               m_D;
   double            m_sigma;
   double            m_omega[];   // D * nfeat, row major
   double            m_phi[];     // D
   double            m_head[];    // [0]=bias, [1..D]=weights
   double            m_g[];
   double            m_z[];
   CAdam             m_adam;

   void              Project(const double &x[])
     {
      double scale = MathSqrt(2.0 / (double)m_D);
      for(int j = 0; j < m_D; j++)
        {
         double dot = 0.0;
         int base = j * m_nfeat;
         for(int i = 0; i < m_nfeat; i++)
            dot += m_omega[base + i] * x[i];
         m_z[j] = scale * MathCos(dot / m_sigma + m_phi[j]);
        }
     }

public:
                     CRffModel(void) : m_D(64), m_sigma(3.0) {}

   virtual string    Name(void) const { return("rff"); }

   void              Init(const int nfeat, const int D, const double sigma,
                          const double lr, const double l2, const int seed)
     {
      m_nfeat = nfeat;
      m_D     = MathMax(4, D);
      m_sigma = MathMax(0.1, sigma);
      m_lr    = lr;
      m_l2    = l2;

      ArrayResize(m_omega, m_D * nfeat);
      ArrayResize(m_phi,   m_D);
      ArrayResize(m_head,  m_D + 1);
      ArrayResize(m_g,     m_D + 1);
      ArrayResize(m_z,     m_D);
      ArrayInitialize(m_head, 0.0);
      ArrayInitialize(m_g,    0.0);

      MathSrand(seed);
      for(int j = 0; j < m_D; j++)
        {
         for(int i = 0; i < nfeat; i++)
            m_omega[j * nfeat + i] = RandNormal();
         m_phi[j] = RandUniform() * 2.0 * M_PI;
        }
      m_adam.Init(m_D + 1);
     }

   virtual double    Predict(const double &x[])
     {
      Project(x);
      double s = m_head[0];
      for(int j = 0; j < m_D; j++)
         s += m_head[j + 1] * m_z[j];
      return(Sigmoid(s));
     }

   virtual void      Learn(const double &x[], const double y, const double weight)
     {
      double p = Predict(x);       // fills m_z
      double d = (p - y) * weight;
      m_g[0] = d;
      for(int j = 0; j < m_D; j++)
         m_g[j + 1] = d * m_z[j];
      ClipGradient(m_g, 5.0);
      m_adam.Step(m_head, m_g, m_lr, m_l2);
     }

   virtual int       ParamCount(void) const { return(2 + m_D * m_nfeat + m_D + m_D + 1); }

   virtual void      GetParams(double &p[])
     {
      int n = 2 + m_D * m_nfeat + m_D + m_D + 1;
      ArrayResize(p, n);
      int k = 0;
      p[k++] = (double)m_D;
      p[k++] = m_sigma;
      for(int i = 0; i < m_D * m_nfeat; i++)
         p[k++] = m_omega[i];
      for(int j = 0; j < m_D; j++)
         p[k++] = m_phi[j];
      for(int j = 0; j <= m_D; j++)
         p[k++] = m_head[j];
     }

   virtual bool      SetParams(const double &p[])
     {
      if(ArraySize(p) < 3)
         return(false);
      int D = (int)MathRound(p[0]);
      if(D < 1 || D > 4096)
         return(false);
      int expect = 2 + D * m_nfeat + D + D + 1;
      if(ArraySize(p) != expect)
        {
         PrintFormat("RFF SetParams: size mismatch, got %d expected %d", ArraySize(p), expect);
         return(false);
        }

      m_D     = D;
      m_sigma = MathMax(0.1, p[1]);
      ArrayResize(m_omega, m_D * m_nfeat);
      ArrayResize(m_phi,   m_D);
      ArrayResize(m_head,  m_D + 1);
      ArrayResize(m_g,     m_D + 1);
      ArrayResize(m_z,     m_D);

      int k = 2;
      for(int i = 0; i < m_D * m_nfeat; i++)
         m_omega[i] = p[k++];
      for(int j = 0; j < m_D; j++)
         m_phi[j] = p[k++];
      for(int j = 0; j <= m_D; j++)
         m_head[j] = p[k++];

      m_adam.Init(m_D + 1);
      return(true);
     }
  };

//+------------------------------------------------------------------+
//| Multilayer perceptron: n -> H (tanh) -> 1 (sigmoid)              |
//|                                                                  |
//| params: [ H , W1(H*n) , b1(H) , W2(H) , b2 ]                     |
//+------------------------------------------------------------------+
class CMlpModel : public CModelBase
  {
private:
   int               m_H;
   double            m_p[];       // flat trainable vector
   double            m_g[];
   double            m_h[];       // hidden activations
   CAdam             m_adam;

   int               OffW1(void) const { return(0); }
   int               OffB1(void) const { return(m_H * m_nfeat); }
   int               OffW2(void) const { return(m_H * m_nfeat + m_H); }
   int               OffB2(void) const { return(m_H * m_nfeat + m_H + m_H); }

   void              Forward(const double &x[])
     {
      int w1 = OffW1(), b1 = OffB1();
      for(int j = 0; j < m_H; j++)
        {
         double s = m_p[b1 + j];
         int base = w1 + j * m_nfeat;
         for(int i = 0; i < m_nfeat; i++)
            s += m_p[base + i] * x[i];
         m_h[j] = TanhAct(s);
        }
     }

public:
                     CMlpModel(void) : m_H(24) {}

   virtual string    Name(void) const { return("mlp"); }

   void              Init(const int nfeat, const int hidden, const double lr,
                          const double l2, const int seed)
     {
      m_nfeat = nfeat;
      m_H     = MathMax(2, hidden);
      m_lr    = lr;
      m_l2    = l2;

      int n = m_H * nfeat + m_H + m_H + 1;
      ArrayResize(m_p, n);
      ArrayResize(m_g, n);
      ArrayResize(m_h, m_H);
      ArrayInitialize(m_p, 0.0);
      ArrayInitialize(m_g, 0.0);

      //--- Xavier initialisation for the hidden layer
      MathSrand(seed);
      double lim = MathSqrt(6.0 / (double)(nfeat + m_H));
      int w1 = OffW1();
      for(int i = 0; i < m_H * nfeat; i++)
         m_p[w1 + i] = (RandUniform() * 2.0 - 1.0) * lim;
      double lim2 = MathSqrt(6.0 / (double)(m_H + 1));
      int w2 = OffW2();
      for(int j = 0; j < m_H; j++)
         m_p[w2 + j] = (RandUniform() * 2.0 - 1.0) * lim2;

      m_adam.Init(n);
     }

   virtual double    Predict(const double &x[])
     {
      Forward(x);
      double s = m_p[OffB2()];
      int w2 = OffW2();
      for(int j = 0; j < m_H; j++)
         s += m_p[w2 + j] * m_h[j];
      return(Sigmoid(s));
     }

   virtual void      Learn(const double &x[], const double y, const double weight)
     {
      double p = Predict(x);        // fills m_h
      double d = (p - y) * weight;  // dL/dz_out

      int w1 = OffW1(), b1 = OffB1(), w2 = OffW2(), b2 = OffB2();

      m_g[b2] = d;
      for(int j = 0; j < m_H; j++)
        {
         m_g[w2 + j] = d * m_h[j];
         //--- backprop through tanh: dtanh = 1 - h^2
         double dh = d * m_p[w2 + j] * (1.0 - m_h[j] * m_h[j]);
         m_g[b1 + j] = dh;
         int base = w1 + j * m_nfeat;
         for(int i = 0; i < m_nfeat; i++)
            m_g[base + i] = dh * x[i];
        }
      ClipGradient(m_g, 5.0);
      m_adam.Step(m_p, m_g, m_lr, m_l2);
     }

   virtual int       ParamCount(void) const { return(1 + m_H * m_nfeat + m_H + m_H + 1); }

   virtual void      GetParams(double &p[])
     {
      int n = ArraySize(m_p);
      ArrayResize(p, n + 1);
      p[0] = (double)m_H;
      for(int i = 0; i < n; i++)
         p[i + 1] = m_p[i];
     }

   virtual bool      SetParams(const double &p[])
     {
      if(ArraySize(p) < 2)
         return(false);
      int H = (int)MathRound(p[0]);
      if(H < 1 || H > 1024)
         return(false);
      int expect = 1 + H * m_nfeat + H + H + 1;
      if(ArraySize(p) != expect)
        {
         PrintFormat("MLP SetParams: size mismatch, got %d expected %d", ArraySize(p), expect);
         return(false);
        }

      m_H = H;
      int n = expect - 1;
      ArrayResize(m_p, n);
      ArrayResize(m_g, n);
      ArrayResize(m_h, m_H);
      for(int i = 0; i < n; i++)
         m_p[i] = p[i + 1];
      m_adam.Init(n);
      return(true);
     }
  };

#endif // __NAS100ML_MODELS_MQH__
//+------------------------------------------------------------------+

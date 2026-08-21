//+------------------------------------------------------------------+
//|                                               NAS100_ML_Bot.mq5  |
//|                                                                  |
//|  Machine-learning Expert Advisor for the NASDAQ 100 CFD.         |
//|                                                                  |
//|  Pipeline, once per closed bar:                                  |
//|                                                                  |
//|    bar closes                                                    |
//|      -> resolve triple-barrier labels that matured on this bar   |
//|      -> score each matured sample out-of-sample (walk forward)   |
//|      -> update the Hedge weights over the three experts          |
//|      -> train on the new sample + a few replayed ones            |
//|      -> extract this bar's features, open a new pending sample   |
//|      -> predict P(up barrier first) and, if every risk gate and  |
//|         every confidence gate agrees, send one order             |
//|                                                                  |
//|  The models learn continuously from the instrument they trade,   |
//|  so there is no separate training run to keep in sync - though   |
//|  tools/train_offline.py can pre-train a checkpoint from exported |
//|  data if you want the EA to start with an opinion.               |
//|                                                                  |
//|  READ docs/INSTALL.md BEFORE TRADING THIS LIVE.                  |
//+------------------------------------------------------------------+
#property copyright "NAS100ML"
#property link      "https://github.com/ParadiseDevelopment/Nasdaq"
#property version   "1.00"
#property description "Online-learning ensemble (logistic + RFF kernel + MLP) for NAS100"

#include <NAS100ML/Utils.mqh>
#include <NAS100ML/FeatureEngine.mqh>
#include <NAS100ML/Scaler.mqh>
#include <NAS100ML/Models.mqh>
#include <NAS100ML/Ensemble.mqh>
#include <NAS100ML/Labeler.mqh>
#include <NAS100ML/ReplayBuffer.mqh>
#include <NAS100ML/RiskManager.mqh>
#include <NAS100ML/TradeExecutor.mqh>
#include <NAS100ML/Persistence.mqh>

//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input group "=== General ==="
input ENUM_TIMEFRAMES InpTimeframe        = PERIOD_M15;   // Working timeframe
input long            InpMagic            = 771001;       // Magic number
input ulong           InpSlippagePoints   = 30;           // Max slippage (points)
input bool            InpShowPanel        = true;         // Draw status panel on chart

input group "=== Learning ==="
input bool            InpOnlineLearning   = true;         // Keep training while live
input double          InpLearningRate     = 0.010;        // AdamW learning rate
input double          InpWeightDecay      = 0.0001;       // L2 / weight decay
input int             InpRffDim           = 64;           // Random Fourier feature count
input double          InpRffSigma         = 8.0;          // RBF kernel bandwidth
input int             InpMlpHidden        = 24;           // MLP hidden units
input double          InpHedgeEta         = 0.35;         // Expert re-weighting speed
input int             InpReplaySteps      = 4;            // Replayed samples per new sample
input int             InpReplayCapacity   = 4000;         // Replay buffer size
input int             InpMinTrainSamples  = 400;          // Warm-up before the first trade
input int             InpEvalWindow       = 300;          // Rolling accuracy window
input int             InpRandomSeed       = 20240517;     // Seed for reproducible runs

input group "=== Label (triple barrier) ==="
input int             InpLabelHorizon     = 12;           // Vertical barrier (bars)
input double          InpBarrierATR       = 1.20;         // Horizontal barriers (x ATR)
input double          InpTimeBarrierWeight= 0.50;         // Weight of time-barrier samples

input group "=== Signal gates ==="
input double          InpProbThreshold    = 0.58;         // Min P(up) to buy / max 1-P to sell
input double          InpMinRollAccuracy  = 0.52;         // Min rolling walk-forward accuracy
input double          InpMaxDisagreement  = 0.45;         // Max spread between expert opinions
input bool            InpAllowLong        = true;         // Allow long trades
input bool            InpAllowShort       = true;         // Allow short trades

input group "=== Risk ==="
input double          InpRiskPercent      = 0.50;         // Risk per trade (% of equity)
input double          InpMaxDailyLossPct  = 3.0;          // Daily loss stop (%)
input double          InpMaxDrawdownPct   = 15.0;         // Kill switch drawdown (%)
input double          InpMaxSpreadPoints  = 80.0;         // Max spread (points)
input double          InpMaxSpreadAtrFrac = 0.15;         // Max spread as fraction of ATR
input int             InpMaxTradesPerDay  = 8;            // Max entries per day
input int             InpLossStreakTrigger= 3;            // Losses in a row before cooldown
input int             InpCooldownBars     = 8;            // Cooldown length (bars)
input double          InpMaxLots          = 5.0;          // Hard volume cap

input group "=== Exits ==="
input double          InpStopATR          = 1.50;         // Initial stop (x ATR)
input double          InpTakeProfitR      = 1.60;         // Target (x initial risk)
input double          InpBreakEvenR       = 0.90;         // Lift to break even at (x R)
input double          InpBreakEvenOffsetR = 0.10;         // Break-even offset (x R)
input double          InpTrailATR         = 2.00;         // Trailing distance (x ATR)
input double          InpTrailStartR      = 1.30;         // Start trailing at (x R)
input int             InpMaxHoldBars      = 36;           // Time stop (bars, 0 = off)

input group "=== Sessions (broker server hours) ==="
input bool            InpUseSessionFilter = true;         // Only trade inside the window
input int             InpTradeStartHour   = 9;            // Trading window start hour
input int             InpTradeEndHour     = 22;           // Trading window end hour
input int             InpLondonStart      = 8;            // London open (feature)
input int             InpLondonEnd        = 17;           // London close (feature)
input int             InpNYStart          = 14;           // New York open (feature)
input int             InpNYEnd            = 23;           // New York close (feature)

input group "=== Persistence ==="
input bool            InpLoadCheckpoint   = false;        // Load weights on start
input bool            InpSaveCheckpoint   = true;         // Save weights periodically
input string          InpCheckpointFile   = "NAS100ML/model.txt";   // Checkpoint path (MQL5\Files)
input int             InpSaveEveryBars    = 500;          // Save interval (bars)
input bool            InpExportDataset    = false;        // Write training rows to CSV
input string          InpDatasetFile      = "NAS100ML/dataset.csv"; // Dataset path (MQL5\Files)

//+------------------------------------------------------------------+
//| Globals                                                          |
//+------------------------------------------------------------------+
CFeatureEngine  g_features;
CScaler         g_scaler;
CEnsemble       g_ensemble;
CLabeler        g_labeler;
CReplayBuffer   g_replay;
CRiskManager    g_risk;
CTradeExecutor  g_exec;
CDatasetWriter  g_dataset;

ENUM_TIMEFRAMES g_tf;
datetime        g_lastBarTime   = 0;
long            g_barsSinceSave = 0;

double          g_initialRisk   = 0.0;   // price distance of the entry stop
int             g_barsInTrade   = 0;

double          g_lastProb      = 0.5;
double          g_lastAtr       = 0.0;
string          g_lastAction    = "starting up";

//+------------------------------------------------------------------+
int OnInit()
  {
   g_tf = (InpTimeframe == PERIOD_CURRENT) ? (ENUM_TIMEFRAMES)Period() : InpTimeframe;

   if(!g_features.Init(_Symbol, g_tf))
      return(INIT_FAILED);
   g_features.SetSessions(InpLondonStart, InpLondonEnd, InpNYStart, InpNYEnd);

   int n = NASML_FEATURE_COUNT;

   MathSrand(InpRandomSeed);

   g_scaler.Init(n);
   g_ensemble.Init(n, InpLearningRate, InpWeightDecay,
                   InpRffDim, InpRffSigma, InpMlpHidden,
                   InpRandomSeed, InpHedgeEta, InpEvalWindow);
   g_labeler.Init(n, InpLabelHorizon, InpBarrierATR, InpTimeBarrierWeight);
   g_replay.Init(n, InpReplayCapacity);

   g_risk.Init(_Symbol);
   g_risk.Configure(InpRiskPercent, InpMaxDailyLossPct, InpMaxDrawdownPct,
                    InpMaxSpreadPoints, InpMaxSpreadAtrFrac, InpMaxTradesPerDay,
                    InpCooldownBars, InpLossStreakTrigger, InpMaxLots,
                    InpUseSessionFilter, InpTradeStartHour, InpTradeEndHour);

   g_exec.Init(_Symbol, InpMagic, InpSlippagePoints);
   g_exec.ConfigureExits(InpBreakEvenR, InpBreakEvenOffsetR, InpTrailATR, InpTrailStartR);

   if(InpLoadCheckpoint)
      NasmlLoadCheckpoint(InpCheckpointFile, g_scaler, g_ensemble, n);

   if(InpExportDataset && !g_dataset.Open(InpDatasetFile, n))
      Print("NASML: dataset export disabled (file could not be opened)");

   g_lastBarTime = iTime(_Symbol, g_tf, 0);

   PrintFormat("NASML: initialised on %s %s | features=%d experts=3 (logistic, rff-%d, mlp-%d)",
               _Symbol, EnumToString(g_tf), n, InpRffDim, InpMlpHidden);

   //--- the timer only drives the panel; in a non-visual backtest it would
   //--- just burn cycles redrawing a comment nobody sees
   if(InpShowPanel && (!MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_VISUAL_MODE)))
      EventSetTimer(1);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();

   if(InpSaveCheckpoint && g_ensemble.TrainedSamples() > 0)
      NasmlSaveCheckpoint(InpCheckpointFile, g_scaler, g_ensemble, NASML_FEATURE_COUNT);

   g_dataset.Close();
   g_features.Release();
   Comment("");

   PrintFormat("NASML: stopped (reason %d) | trained on %I64d samples | rolling acc %.3f",
               reason, g_ensemble.TrainedSamples(), g_ensemble.RollingAccuracy());
  }

//+------------------------------------------------------------------+
void OnTimer()
  {
   if(InpShowPanel)
      UpdatePanel();
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   //--- intrabar work: keep stops moving with the market
   if(g_lastAtr > 0.0 && g_initialRisk > 0.0)
      g_exec.ManageOpen(g_lastAtr, g_initialRisk);

   datetime t = iTime(_Symbol, g_tf, 0);
   if(t == g_lastBarTime || t == 0)
      return;
   g_lastBarTime = t;

   OnNewBar();
  }

//+------------------------------------------------------------------+
//| Everything that must happen exactly once per closed bar.         |
//+------------------------------------------------------------------+
void OnNewBar()
  {
   g_barsSinceSave++;

   MqlRates bar[];
   ArraySetAsSeries(bar, true);
   if(CopyRates(_Symbol, g_tf, 1, 1, bar) != 1)
      return;

   //--- 1) realised trade results feed the risk cooldown logic ------
   double closedProfit = 0.0;
   if(g_exec.PollClosedTrades(closedProfit) > 0)
     {
      g_risk.RegisterResult(closedProfit);
      g_initialRisk = 0.0;
      g_barsInTrade = 0;
     }

   g_risk.OnBar(bar[0].time);

   //--- 2) mature the pending triple-barrier samples ----------------
   g_labeler.Update(bar[0].high, bar[0].low, bar[0].close);
   ConsumeResolvedSamples(bar[0].time);

   //--- 3) build this bar's features --------------------------------
   double raw[];
   if(!g_features.Compute(raw))
     {
      g_lastAction = "warming up (indicator history)";
      return;
     }
   g_lastAtr = g_features.LastATR();

   //--- open a new pending sample anchored on this bar's close
   g_labeler.Add(bar[0].time, bar[0].close, g_lastAtr, raw);

   //--- 4) predict --------------------------------------------------
   double x[];
   g_scaler.Transform(raw, x);
   g_lastProb = g_ensemble.Predict(x);

   //--- 5) manage an existing position, or look for a new one --------
   if(g_exec.HasPosition())
     {
      //--- a position that outlived a terminal restart has no recorded
      //--- risk distance; fall back to the configured stop width so the
      //--- trailing logic still has a scale to work from
      if(g_initialRisk <= 0.0)
         g_initialRisk = InpStopATR * g_lastAtr;

      g_barsInTrade++;
      if(InpMaxHoldBars > 0 && g_barsInTrade >= InpMaxHoldBars)
        {
         g_exec.CloseAll("time stop");
         g_lastAction = "closed on time stop";
        }
      else
         g_lastAction = StringFormat("holding %s (%d bars)",
                                     g_exec.PositionDirection() > 0 ? "long" : "short",
                                     g_barsInTrade);
     }
   else
      TryEnter(bar[0].time, x);

   //--- 6) checkpoint -----------------------------------------------
   if(InpSaveCheckpoint && InpSaveEveryBars > 0 && g_barsSinceSave >= InpSaveEveryBars)
     {
      g_barsSinceSave = 0;
      if(g_ensemble.TrainedSamples() > 0)
         NasmlSaveCheckpoint(InpCheckpointFile, g_scaler, g_ensemble, NASML_FEATURE_COUNT);
     }

   if(InpShowPanel)
      UpdatePanel();
  }

//+------------------------------------------------------------------+
//| Train on every sample whose barrier resolved on this bar.        |
//|                                                                  |
//| Each sample is scored BEFORE it is learned from, so the rolling  |
//| accuracy the entry gate relies on is a genuine walk-forward      |
//| statistic rather than in-sample fit.                             |
//+------------------------------------------------------------------+
void ConsumeResolvedSamples(const datetime barTime)
  {
   int cnt = g_labeler.ResolvedCount();
   if(cnt <= 0)
      return;

   for(int i = 0; i < cnt; i++)
     {
      double rawSample[];
      double y, w;
      g_labeler.GetResolved(i, rawSample, y, w);

      if(InpExportDataset)
         g_dataset.Write(barTime, rawSample, y, w);

      //--- scaler statistics only ever see data that is already past
      g_scaler.Observe(rawSample);

      double xs[];
      g_scaler.Transform(rawSample, xs);

      if(InpOnlineLearning)
        {
         g_ensemble.Learn(xs, y, w, true);
         g_replay.Add(xs, y, w);

         //--- a few extra gradient steps over the recent past
         for(int r = 0; r < InpReplaySteps; r++)
           {
            double rx[];
            double ry, rw;
            if(!g_replay.Sample(rx, ry, rw))
               break;
            g_ensemble.Learn(rx, ry, rw, false);
           }
        }
      else
        {
         //--- frozen weights: still grade the prediction so the panel and
         //--- the accuracy gate keep reporting live out-of-sample numbers
         g_ensemble.Evaluate(xs, y);
        }
     }

   g_labeler.ClearResolved();
  }

//+------------------------------------------------------------------+
//| Confidence gates, risk gates, then one market order.             |
//+------------------------------------------------------------------+
void TryEnter(const datetime barTime, const double &x[])
  {
   if(g_ensemble.TrainedSamples() < InpMinTrainSamples)
     {
      g_lastAction = StringFormat("warm-up %I64d/%d samples",
                                  g_ensemble.TrainedSamples(), InpMinTrainSamples);
      return;
     }

   if(g_ensemble.RollingAccuracy() < InpMinRollAccuracy)
     {
      g_lastAction = StringFormat("model below accuracy floor (%.3f)",
                                  g_ensemble.RollingAccuracy());
      return;
     }

   double disagree = g_ensemble.Disagreement();
   if(disagree > InpMaxDisagreement)
     {
      g_lastAction = StringFormat("experts disagree (%.2f)", disagree);
      return;
     }

   bool wantLong  = (g_lastProb >= InpProbThreshold)       && InpAllowLong;
   bool wantShort = (g_lastProb <= 1.0 - InpProbThreshold) && InpAllowShort;

   if(!wantLong && !wantShort)
     {
      g_lastAction = StringFormat("no edge (p=%.3f)", g_lastProb);
      return;
     }

   if(!g_risk.CanTrade(barTime, g_lastAtr))
     {
      g_lastAction = "blocked: " + g_risk.BlockReason();
      return;
     }

   if(g_lastAtr <= 0.0)
      return;

   //--- geometry ----------------------------------------------------
   double stopDist = InpStopATR * g_lastAtr;
   double tpDist   = InpTakeProfitR * stopDist;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0)
      return;

   //--- size scales with how far the blended probability sits from a
   //--- coin flip, bounded so one confident signal cannot double risk
   double edge  = MathAbs(g_lastProb - 0.5) * 2.0;
   double scale = Clamp(0.6 + edge, 0.6, 1.4);

   double lots = g_risk.CalcLots(stopDist, scale);
   if(lots <= 0.0)
     {
      g_lastAction = "risk budget below minimum lot";
      return;
     }

   bool   isLong = wantLong;
   double price  = isLong ? ask : bid;
   double sl     = isLong ? price - stopDist : price + stopDist;
   double tp     = isLong ? price + tpDist   : price - tpDist;

   ENUM_ORDER_TYPE otype = isLong ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   if(!g_risk.MarginOk(otype, lots, price))
     {
      g_lastAction = "blocked: " + g_risk.BlockReason();
      return;
     }

   string comment = StringFormat("NASML p=%.3f a=%.3f", g_lastProb, g_ensemble.RollingAccuracy());

   if(g_exec.Open(isLong, lots, sl, tp, comment))
     {
      g_risk.RegisterEntry();
      g_initialRisk = stopDist;
      g_barsInTrade = 0;
      g_lastAction   = StringFormat("opened %s %.2f lots @ %.2f", isLong ? "LONG" : "SHORT",
                                    lots, price);
      PrintFormat("NASML ENTRY %s %.2f lots  p=%.3f  acc=%.3f  atr=%.2f  sl=%.2f tp=%.2f",
                  isLong ? "BUY" : "SELL", lots, g_lastProb,
                  g_ensemble.RollingAccuracy(), g_lastAtr, sl, tp);
     }
   else
      g_lastAction = "order rejected by broker";
  }

//+------------------------------------------------------------------+
//| Chart panel                                                      |
//+------------------------------------------------------------------+
void UpdatePanel()
  {
   string s = "";
   s += "NAS100 ML Bot  |  " + _Symbol + "  " + EnumToString(g_tf) + "\n";
   s += "---------------------------------------------\n";
   s += StringFormat("P(up barrier)     : %.3f\n", g_lastProb);
   s += StringFormat("Expert probs      : lin %.3f | rff %.3f | mlp %.3f\n",
                     g_ensemble.LastExpertProb(0), g_ensemble.LastExpertProb(1),
                     g_ensemble.LastExpertProb(2));
   s += StringFormat("Hedge weights     : %.2f / %.2f / %.2f\n",
                     g_ensemble.ExpertWeight(0), g_ensemble.ExpertWeight(1),
                     g_ensemble.ExpertWeight(2));
   s += StringFormat("Walk-fwd accuracy : %.3f  (n=%d)\n",
                     g_ensemble.RollingAccuracy(), g_ensemble.EvalCount());
   s += StringFormat("Rolling log loss  : %.4f\n", g_ensemble.RollingLogLoss());
   s += StringFormat("Samples trained   : %I64d\n", g_ensemble.TrainedSamples());
   s += StringFormat("Replay buffer     : %d / %d  (%.0f%% up)\n",
                     g_replay.Count(), g_replay.Capacity(), g_replay.PositiveShare() * 100.0);
   s += "---------------------------------------------\n";
   s += StringFormat("ATR               : %.2f\n", g_lastAtr);
   s += StringFormat("Trades today      : %d / %d\n", g_risk.TradesToday(), InpMaxTradesPerDay);
   s += StringFormat("Equity peak       : %.2f\n", g_risk.EquityPeak());
   s += StringFormat("Risk state        : %s\n",
                     g_risk.HaltedForGood() ? "DISABLED (max drawdown)"
                                            : (g_risk.HaltedForDay() ? "paused (daily loss)" : "active"));
   s += "Last action       : " + g_lastAction + "\n";
   Comment(s);
  }

//+------------------------------------------------------------------+
//| Optimisation criterion: reward net profit per unit of drawdown,  |
//| discount thin samples so the optimiser cannot win on 5 trades.   |
//+------------------------------------------------------------------+
double OnTester()
  {
   double trades = TesterStatistics(STAT_TRADES);
   if(trades < 40.0)
      return(0.0);

   double net     = TesterStatistics(STAT_PROFIT);
   double deposit = MathMax(TesterStatistics(STAT_INITIAL_DEPOSIT), 1.0);
   double ddMoney = TesterStatistics(STAT_EQUITY_DD);
   double pf      = MathMin(TesterStatistics(STAT_PROFIT_FACTOR), 3.0);
   if(net <= 0.0)
      return(net);

   double ddPct        = MathMax(ddMoney / deposit * 100.0, 1.0);
   double sampleFactor = MathMin(1.0, trades / 200.0);
   return((net / ddPct) * pf * sampleFactor);
  }
//+------------------------------------------------------------------+

"""
Does the feature set contain any real predictive signal?

The backtest can be fooled: a sample opens on every bar and takes the label
horizon to resolve, so neighbouring samples share nearly all of their outcome
and accuracy measured over them is inflated. This strips that out. It keeps
only NON-OVERLAPPING samples (one per horizon), splits them chronologically,
trains on the first 70% and scores the last 30%.

Read the last column. An edge inside one standard error of zero is not an
edge; SE is roughly sqrt(0.25 / n_validation), and n_validation is 30% of
n_indep. Run this BEFORE tuning anything - if there is no signal here, no
amount of parameter search on the backtest will create one.

    python3 tools/edge_test.py bars.csv
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import load_bars, build_features, atr
from train_offline import Logistic, Mlp, accuracy

if len(sys.argv) != 2:
    raise SystemExit(__doc__)

df,_ = load_bars(sys.argv[1], {})
F = build_features(df, dict(london_start=10,london_end=18,ny_start=16,ny_end=23))
A = atr(df,14).to_numpy(); H=df.high.to_numpy(); L=df.low.to_numpy(); C=df.close.to_numpy()

def labels(hz, k):
    n=len(C); y=np.full(n,np.nan)
    for i in range(260, n-hz):
        up, dn = C[i]+k*A[i], C[i]-k*A[i]
        hi=H[i+1:i+1+hz]; lo=L[i+1:i+1+hz]
        tu=np.argmax(hi>=up) if (hi>=up).any() else 10**9
        td=np.argmax(lo<=dn) if (lo<=dn).any() else 10**9
        y[i] = 1.0 if tu<td else (0.0 if td<tu else (1.0 if C[i+hz]>C[i] else 0.0))
    return y

print(f"non-overlapping, out-of-sample. edge within ~1 SE of 0 is noise.\n")
print(f"{'horizon/barrier':<18}{'n_indep':>9}{'baseline':>10}{'logit OOS':>11}{'mlp OOS':>10}{'best edge':>11}")
for hz,k in [(12,1.2),(24,1.5),(24,2.5),(48,2.5),(48,3.5),(96,4.0)]:
    y = labels(hz,k)
    idx = np.arange(260, len(C)-hz, hz)          # non-overlapping only
    idx = idx[np.isfinite(y[idx])]
    X, Y = F[idx], y[idx]
    cut=int(len(X)*.7)
    mu,sd = X[:cut].mean(0), np.maximum(X[:cut].std(0),1e-8)
    Z=np.clip((X-mu)/sd,-5,5)
    Ztr,Ytr,Zva,Yva = Z[:cut],Y[:cut],Z[cut:],Y[cut:]
    if len(Zva)<40: continue
    base=max(Yva.mean(),1-Yva.mean())
    rng=np.random.default_rng(0)
    lg=Logistic(X.shape[1]); ml=Mlp(X.shape[1],24,rng)
    W=np.ones(len(Ztr))
    for _ in range(60):
        lg.fit_epoch(Ztr,Ytr,W,.01,1e-3,32); ml.fit_epoch(Ztr,Ytr,W,.006,1e-3,32)
    a1,a2=accuracy(lg.predict(Zva),Yva),accuracy(ml.predict(Zva),Yva)
    print(f"{f'{hz}b / {k}ATR':<18}{len(X):>9}{base:>10.4f}{a1:>11.4f}{a2:>10.4f}{max(a1,a2)-base:>+11.4f}"
          f"{'':>3}(SE {np.sqrt(0.25/max(len(Yva),1)):.4f})")

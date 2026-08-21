#!/usr/bin/env python3
"""
Offline pre-trainer for NAS100_ML_Bot.

The EA learns online, so this script is optional. It is useful when you want
the bot to start with an opinion instead of spending its first few hundred
bars flat, or when you want to inspect how learnable the feature set actually
is on your own broker's history before risking anything.

Workflow
--------
1. Run the EA in the Strategy Tester with  InpExportDataset = true.
   It writes MQL5/Files/NAS100ML/dataset.csv - one row per resolved
   triple-barrier sample, with the RAW (unscaled) features.

2. Train:
       python3 tools/train_offline.py --data dataset.csv --out model.txt

3. Copy model.txt into MQL5/Files/NAS100ML/ and run the EA with
   InpLoadCheckpoint = true  (and InpMinTrainSamples lowered, since the
   warm-up counter starts from zero on every attach).

The checkpoint format and every parameter layout below mirror
MQL5/Include/NAS100ML/Persistence.mqh and Models.mqh exactly. If you change
one side, change the other.

Only numpy is required.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

CHECKPOINT_VERSION = 2
N_EXPERTS = 3


# ----------------------------------------------------------------------
# AdamW - same update rule as CAdam in Utils.mqh
# ----------------------------------------------------------------------
class AdamW:
    def __init__(self, shape, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0
        self.b1, self.b2, self.eps = beta1, beta2, eps

    def step(self, p, g, lr, weight_decay):
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * g
        self.v = self.b2 * self.v + (1.0 - self.b2) * g * g
        mhat = self.m / (1.0 - self.b1 ** self.t)
        vhat = self.v / (1.0 - self.b2 ** self.t)
        p -= lr * (mhat / (np.sqrt(vhat) + self.eps) + weight_decay * p)
        return p


def sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def log_loss(p, y, w=None):
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    ll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    if w is None:
        return float(ll.mean())
    return float((ll * w).sum() / max(w.sum(), 1e-12))


def accuracy(p, y):
    return float((((p >= 0.5).astype(float)) == y).mean())


# ----------------------------------------------------------------------
# Experts
# ----------------------------------------------------------------------
class Logistic:
    """params: [b, w(n)]"""

    def __init__(self, n):
        self.n = n
        self.p = np.zeros(n + 1)
        self.opt = AdamW(self.p.shape)

    def predict(self, X):
        return sigmoid(X @ self.p[1:] + self.p[0])

    def fit_epoch(self, X, y, w, lr, l2, batch):
        for idx in minibatches(len(X), batch):
            xb, yb, wb = X[idx], y[idx], w[idx]
            pb = sigmoid(xb @ self.p[1:] + self.p[0])
            d = (pb - yb) * wb / len(idx)
            g = np.empty_like(self.p)
            g[0] = d.sum()
            g[1:] = xb.T @ d
            self.p = self.opt.step(self.p, np.clip(g, -5, 5), lr, l2)

    def export(self):
        return self.p.copy()


class Rff:
    """params: [D, sigma, omega(D*n), phi(D), head_b, head_w(D)]"""

    def __init__(self, n, D, sigma, rng):
        self.n, self.D, self.sigma = n, D, sigma
        self.omega = rng.standard_normal((D, n))
        self.phi = rng.uniform(0.0, 2.0 * np.pi, D)
        self.head = np.zeros(D + 1)
        self.opt = AdamW(self.head.shape)

    def project(self, X):
        return np.sqrt(2.0 / self.D) * np.cos(X @ self.omega.T / self.sigma + self.phi)

    def predict(self, X):
        return sigmoid(self.project(X) @ self.head[1:] + self.head[0])

    def fit_epoch(self, X, y, w, lr, l2, batch):
        for idx in minibatches(len(X), batch):
            zb = self.project(X[idx])
            yb, wb = y[idx], w[idx]
            pb = sigmoid(zb @ self.head[1:] + self.head[0])
            d = (pb - yb) * wb / len(idx)
            g = np.empty_like(self.head)
            g[0] = d.sum()
            g[1:] = zb.T @ d
            self.head = self.opt.step(self.head, np.clip(g, -5, 5), lr, l2)

    def export(self):
        return np.concatenate(
            [[float(self.D), float(self.sigma)], self.omega.ravel(), self.phi, self.head]
        )


class Mlp:
    """params: [H, W1(H*n), b1(H), W2(H), b2]"""

    def __init__(self, n, H, rng):
        self.n, self.H = n, H
        lim1 = np.sqrt(6.0 / (n + H))
        lim2 = np.sqrt(6.0 / (H + 1))
        self.W1 = rng.uniform(-lim1, lim1, (H, n))
        self.b1 = np.zeros(H)
        self.W2 = rng.uniform(-lim2, lim2, H)
        self.b2 = np.zeros(1)
        self.oW1 = AdamW(self.W1.shape)
        self.ob1 = AdamW(self.b1.shape)
        self.oW2 = AdamW(self.W2.shape)
        self.ob2 = AdamW(self.b2.shape)

    def forward(self, X):
        h = np.tanh(X @ self.W1.T + self.b1)
        return h, sigmoid(h @ self.W2 + self.b2[0])

    def predict(self, X):
        return self.forward(X)[1]

    def fit_epoch(self, X, y, w, lr, l2, batch):
        for idx in minibatches(len(X), batch):
            xb, yb, wb = X[idx], y[idx], w[idx]
            h, pb = self.forward(xb)
            d = (pb - yb) * wb / len(idx)
            gW2 = h.T @ d
            gb2 = np.array([d.sum()])
            dh = np.outer(d, self.W2) * (1.0 - h * h)
            gW1 = dh.T @ xb
            gb1 = dh.sum(axis=0)
            self.W1 = self.oW1.step(self.W1, np.clip(gW1, -5, 5), lr, l2)
            self.b1 = self.ob1.step(self.b1, np.clip(gb1, -5, 5), lr, l2)
            self.W2 = self.oW2.step(self.W2, np.clip(gW2, -5, 5), lr, l2)
            self.b2 = self.ob2.step(self.b2, np.clip(gb2, -5, 5), lr, l2)

    def export(self):
        return np.concatenate(
            [[float(self.H)], self.W1.ravel(), self.b1, self.W2, self.b2]
        )


def minibatches(n, batch):
    order = np.random.permutation(n)
    for start in range(0, n, batch):
        yield order[start:start + batch]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_dataset(path):
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        header = fh.readline().strip().split(",")
        rows = [line.strip().split(",") for line in fh if line.strip()]

    if not rows:
        sys.exit(f"{path}: no data rows")

    if header[0] != "time" or header[-2:] != ["label", "weight"]:
        sys.exit(f"{path}: unexpected header, expected time,<features...>,label,weight")

    names = header[1:-2]
    n = len(names)

    data = np.array([[float(v) for v in r[1:]] for r in rows], dtype=float)
    X = data[:, :n]
    y = data[:, n]
    w = data[:, n + 1]
    return names, X, y, w


def split_walk_forward(X, y, w, frac):
    """Chronological split - the dataset is already in bar order."""
    cut = int(len(X) * (1.0 - frac))
    cut = max(1, min(cut, len(X) - 1))
    return (X[:cut], y[:cut], w[:cut]), (X[cut:], y[cut:], w[cut:])


# ----------------------------------------------------------------------
# Checkpoint writer - mirrors NasmlLoadCheckpoint()
# ----------------------------------------------------------------------
def write_checkpoint(path, n_feat, sections):
    with open(path, "w", encoding="ascii", newline="") as fh:
        fh.write("#NAS100ML\r\n")
        fh.write(f"VERSION {CHECKPOINT_VERSION}\r\n")
        fh.write(f"NFEAT {n_feat}\r\n")
        for tag, values in sections:
            arr = np.asarray(values, dtype=float).ravel()
            fh.write(f"SECTION {tag} {arr.size}\r\n")
            for v in arr:
                fh.write(f"{v:.12g}\r\n")
        fh.write("END\r\n")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset.csv exported by the EA")
    ap.add_argument("--out", default="model.txt", help="checkpoint file to write")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--rff-dim", type=int, default=64)
    ap.add_argument("--rff-sigma", type=float, default=8.0)
    ap.add_argument("--mlp-hidden", type=int, default=24)
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="tail fraction held out for walk-forward validation")
    ap.add_argument("--seed", type=int, default=20240517)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    names, X, y, w = load_dataset(args.data)
    n = X.shape[1]
    print(f"loaded {len(X)} samples x {n} features from {args.data}")
    print(f"class balance: {y.mean():.3f} up / {1 - y.mean():.3f} down")

    (Xtr, ytr, wtr), (Xva, yva, wva) = split_walk_forward(X, y, w, args.val_frac)
    print(f"train {len(Xtr)}  validate {len(Xva)} (chronological tail)")

    # scaler fitted on the training slice only
    mean = Xtr.mean(axis=0)
    std = np.maximum(Xtr.std(axis=0, ddof=1), 1e-8)

    def scale(A):
        return np.clip((A - mean) / std, -5.0, 5.0)

    Ztr, Zva = scale(Xtr), scale(Xva)

    lin = Logistic(n)
    rff = Rff(n, args.rff_dim, args.rff_sigma, rng)
    mlp = Mlp(n, args.mlp_hidden, rng)
    experts = [("logistic", lin), ("rff", rff), ("mlp", mlp)]

    # The MLP runs at 0.6x the base rate, exactly as CEnsemble::Init does,
    # so an offline checkpoint and continued online learning stay comparable.
    lrs = {"logistic": args.lr, "rff": args.lr, "mlp": args.lr * 0.6}

    for epoch in range(1, args.epochs + 1):
        for name, m in experts:
            m.fit_epoch(Ztr, ytr, wtr, lrs[name], args.l2, args.batch)
        if epoch % 10 == 0 or epoch == args.epochs:
            parts = []
            for name, m in experts:
                p = m.predict(Zva)
                parts.append(f"{name} acc={accuracy(p, yva):.3f} ll={log_loss(p, yva):.4f}")
            print(f"epoch {epoch:3d}  " + " | ".join(parts))

    # Hedge weights seeded from validation loss, same exponential rule the EA
    # uses online. These are only a starting point; the EA keeps updating them.
    losses = np.array([log_loss(m.predict(Zva), yva) for _, m in experts])
    hedge = np.exp(-4.0 * (losses - losses.min()))
    hedge = np.maximum(hedge / hedge.sum(), 0.05)
    hedge = hedge / hedge.sum()

    ens = sum(hw * m.predict(Zva) for hw, (_, m) in zip(hedge, experts))
    print()
    print(f"hedge weights      : " + ", ".join(f"{nm}={hw:.3f}" for hw, (nm, _) in zip(hedge, experts)))
    print(f"ensemble validation: acc={accuracy(ens, yva):.4f} logloss={log_loss(ens, yva):.4f}")
    print(f"baseline (majority): acc={max(yva.mean(), 1 - yva.mean()):.4f}")

    edge = accuracy(ens, yva) - max(yva.mean(), 1 - yva.mean())
    if edge <= 0.0:
        print("\n!! The ensemble does not beat the majority class out of sample.")
        print("!! Do not trade this checkpoint. Re-export more data or re-tune")
        print("!! the barrier/horizon before going any further.")

    write_checkpoint(
        args.out,
        n,
        [
            ("SCALER", np.concatenate([mean, std])),
            ("LOGISTIC", lin.export()),
            ("RFF", rff.export()),
            ("MLP", mlp.export()),
            ("HEDGE", hedge),
        ],
    )
    print(f"\nwrote {args.out}")
    print("copy it to  <terminal data folder>/MQL5/Files/NAS100ML/model.txt")


if __name__ == "__main__":
    main()

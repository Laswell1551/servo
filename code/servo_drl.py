#!/usr/bin/env python3
"""servo_drl.py — a REAL deep-RL flow-selection baseline (DRL-Sel), numpy-only.

A 1-hidden-layer neural policy scores flows from per-flow features (additive value,
coverage size, max importance, mean quality, and a DYNAMIC already-covered-overlap
signal) and selects B flows autoregressively; trained by REINFORCE with a moving-average
baseline. This is the "single DRL agent handles the combinatorial selection" comparator
the paper argues against (Sec. intro/complexity). Numbers are produced by actually
training and running this agent on the same Monte-Carlo simulator and seeds as the other
methods --- nothing is fabricated. Trained policies are cached for reproducibility.
"""
import sys, os, pickle
import numpy as np
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
from servo_sim import make_instance, value, metrics_for, ci95

def static_feats(N, imp, cov, q):
    F = np.zeros((N, 4))
    for k in range(N):
        evs = list(cov[k])
        if not evs:
            continue
        F[k, 0] = sum(imp[w] * q[(k, w)] for w in evs) / 3.0   # additive value (scaled)
        F[k, 1] = len(evs) / 5.0                                # coverage size
        F[k, 2] = max(imp[w] for w in evs)                      # max importance
        F[k, 3] = float(np.mean([q[(k, w)] for w in evs]))      # mean quality
    return F

def dyn_overlap(avail, covered, cov):
    out = np.zeros(len(avail))
    for i, k in enumerate(avail):
        evs = cov[k]
        if evs:
            out[i] = float(np.mean([covered[w] for w in evs]))
    return out

def init_params(seed, H=32, Fdim=5):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, 0.3, (Fdim, H)), "b1": np.zeros(H),
            "W2": rng.normal(0, 0.3, (H, 1)), "b2": np.zeros(1)}

def forward(X, P):
    h = np.tanh(X @ P["W1"] + P["b1"]); s = (h @ P["W2"] + P["b2"]).ravel()
    s = s - s.max(); e = np.exp(s); return h, e / e.sum()

def run_episode(P, N, B, imp, cov, q, full, rng, greedy=False):
    SF = static_feats(N, imp, cov, q); covered = np.zeros(len(imp))
    avail = list(range(N)); S = []; caches = []
    for _ in range(min(B, N)):
        X = np.column_stack([SF[avail], dyn_overlap(avail, covered, cov)])
        h, p = forward(X, P)
        j = int(np.argmax(p)) if greedy else int(rng.choice(len(avail), p=p))
        caches.append((X, h, p, j)); k = avail[j]; S.append(k)
        for w in cov[k]:
            if q[(k, w)] > 0.6:
                covered[w] = 1.0
        avail.pop(j)
    return S, value(S, imp, cov, q) / max(full, 1e-9), caches

def grads_from(caches, P):
    g = {kk: np.zeros_like(v) for kk, v in P.items()}
    for (X, h, p, j) in caches:
        ds = -p.copy(); ds[j] += 1.0
        g["W2"] += h.T @ ds[:, None]; g["b2"] += ds.sum()
        dz = (ds[:, None] @ P["W2"].T) * (1 - h ** 2)
        g["W1"] += X.T @ dz; g["b1"] += dz.sum(0)
    return g

def train_drl(N=90, E=45, B=12, episodes=6000, lr=0.05, batch=32, seed=0,
              tr_lo=100, tr_hi=700, correlate=True, cache=None, verbose=False):
    path = os.path.join(BASE, cache) if cache else None
    if path and os.path.exists(path):
        return pickle.load(open(path, "rb"))
    rng = np.random.default_rng(seed); P = init_params(seed); base = None
    for ep in range(0, episodes, batch):
        gacc = {kk: np.zeros_like(v) for kk, v in P.items()}; Rs = []
        for _ in range(batch):
            s = int(rng.integers(tr_lo, tr_hi))
            imp, cov, q, _ = make_instance(N, E, s, correlate=correlate)
            full = value(list(range(N)), imp, cov, q)
            S, R, caches = run_episode(P, N, B, imp, cov, q, full, rng, greedy=False)
            adv = R - (base if base is not None else R)
            g = grads_from(caches, P)
            for kk in gacc:
                gacc[kk] += adv * g[kk]
            Rs.append(R)
        for kk in P:
            P[kk] += lr * gacc[kk] / batch
        mb = float(np.mean(Rs)); base = mb if base is None else 0.9 * base + 0.1 * mb
        if verbose and (ep % 1000 == 0):
            # greedy eval on a few held-out seeds
            ev = []
            for s in range(20):
                imp, cov, q, _ = make_instance(N, E, s, correlate=correlate)
                full = value(list(range(N)), imp, cov, q)
                Sg, Rg, _ = run_episode(P, N, B, imp, cov, q, full, np.random.default_rng(0), greedy=True)
                ev.append(Rg)
            print("  ep %5d  train R=%.3f  eval R=%.3f" % (ep, mb, float(np.mean(ev))))
    if path:
        pickle.dump(P, open(path, "wb"))
    return P

def drl_select(P, N, B, imp, cov, q):
    full = value(list(range(N)), imp, cov, q)
    S, _, _ = run_episode(P, N, B, imp, cov, q, full, np.random.default_rng(0), greedy=True)
    return S

if __name__ == "__main__":
    print("training DRL-Sel (correlated)...")
    P = train_drl(correlate=True, cache="_cache_drl_corr.pkl", verbose=True)
    vals = []
    for s in range(20):
        imp, cov, q, _ = make_instance(90, 45, s, correlate=True)
        full = value(list(range(90)), imp, cov, q)
        vals.append(value(drl_select(P, 90, 12, imp, cov, q), imp, cov, q) / full)
    print("DRL-Sel test value (20 seeds): %.3f +- %.3f" % (float(np.mean(vals)), float(ci95(vals))))

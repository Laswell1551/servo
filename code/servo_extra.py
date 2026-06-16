#!/usr/bin/env python3
"""
servo_extra.py — revision experiments requested by reviewers.
Reuses the FAITHFUL selectors/oracle in servo_sim.py (numpy-only, no fabrication).
Adds:
  (1) Comprehensive comparison WITH 95% CIs, under BOTH the importance-correlated
      regime (default in the paper) and the importance-INDEPENDENT regime
      (make_instance(correlate=False)) -> shows how much of the gain is model-driven.
  (2) Paired SERVO-vs-WUM-RD and SERVO-vs-WUM value differences with 95% CIs
      (paired over the 20 shared seeds) -> significance of the headline margin.
  (3) Redundancy gain sweep (SERVO vs WUM) under both regimes.
All numbers are produced by actually running the implemented selectors.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import servo_sim as ss

SEEDS = ss.SEEDS  # range(20)

def comprehensive(correlate, N=90, E=45, B=12):
    methods = {"SERVO": ss.sel_servo, "WUM-RD": ss.sel_wumrd, "OSM": ss.sel_osm,
               "WUM": ss.sel_wum, "PF/Prio": ss.sel_pfprio, "Random": ss.sel_random}
    keys = ["value", "succ", "hi_sla", "lo_sla", "robust"]
    acc = {m: {k: [] for k in keys} for m in methods}
    paired = {"SERVO_minus_WUMRD": [], "SERVO_minus_WUM": []}
    for s in SEEDS:
        imp, cov, q, _ = ss.make_instance(N, E, s, correlate=correlate)
        full = ss.value(list(range(N)), imp, cov, q)
        vals = {}
        for m, fn in methods.items():
            S = fn(N, B, imp, cov, q) if m != "Random" else fn(N, B, imp, cov, q, seed=s)
            v, succ, hi, lo = ss.metrics_for(S, imp, cov, q, full, E)
            rng = np.random.default_rng(1000 + s)
            qp = {kk: float(np.clip(vv * (1 + rng.uniform(-0.4, 0.4)), 0, 1)) for kk, vv in q.items()}
            Sp = fn(N, B, imp, cov, qp) if m != "Random" else fn(N, B, imp, cov, qp, seed=s)
            rob = ss.value(Sp, imp, cov, q) / full
            for k, val in zip(keys, [v, succ, hi, lo, rob]):
                acc[m][k].append(val)
            vals[m] = v
        paired["SERVO_minus_WUMRD"].append(vals["SERVO"] - vals["WUM-RD"])
        paired["SERVO_minus_WUM"].append(vals["SERVO"] - vals["WUM"])
    summary = {m: {k: (float(np.nanmean(acc[m][k])),
                       float(ss.ci95([x for x in acc[m][k] if not np.isnan(x)]))) for k in keys}
               for m in methods}
    pstats = {p: (float(np.mean(paired[p])), float(ss.ci95(paired[p]))) for p in paired}
    return summary, pstats

def redundancy(correlate, E=45, B=12, Ns=(45, 75, 120, 165, 210, 270, 340)):
    out = []
    for N in Ns:
        gains = []
        for s in SEEDS:
            imp, cov, q, _ = ss.make_instance(N, E, s, correlate=correlate)
            vS = ss.value(ss.sel_servo(N, B, imp, cov, q), imp, cov, q)
            vW = ss.value(ss.sel_wum(N, B, imp, cov, q), imp, cov, q)
            gains.append(100.0 * (vS - vW) / max(vW, 1e-9))
        out.append((round(N * 3.2 / E, 1), float(np.mean(gains)), float(ss.ci95(gains))))
    return out

if __name__ == "__main__":
    res = {}
    for corr in [True, False]:
        tag = "correlated" if corr else "independent"
        summ, pstats = comprehensive(corr)
        red = redundancy(corr)
        res[tag] = {"comprehensive": summ, "paired": pstats, "redundancy": red}
        print("\n===== %s regime (N=90, E=45, B=12) =====" % tag.upper())
        print("%-9s %15s %15s %15s %15s %15s" % ("Method", "Vbar", "Succ", "Hi-SLA", "Lo-SLA", "Robust"))
        for m in summ:
            row = summ[m]
            print("%-9s " % m + " ".join("%.3f+-%.3f" % (row[k][0], row[k][1])
                                          for k in ["value", "succ", "hi_sla", "lo_sla", "robust"]))
        print("paired SERVO-WUMRD: %.4f +- %.4f ;  SERVO-WUM: %.4f +- %.4f"
              % (pstats["SERVO_minus_WUMRD"][0], pstats["SERVO_minus_WUMRD"][1],
                 pstats["SERVO_minus_WUM"][0], pstats["SERVO_minus_WUM"][1]))
        print("redundancy gain SERVO vs WUM (rho, mean%, ci):")
        for rho, gm, gc in red:
            print("  rho=%5.1f: %+5.1f%% +- %.1f" % (rho, gm, gc))
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servo_extra_results.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print("\nsaved", out_path)

#!/usr/bin/env python3
"""servo_v2figs.py — add two MODERN, CPU-feasible baselines and regenerate the
affected figures + comparison tables in the clean publication style.

New baselines (faithfully implemented, numpy-only, no fabricated numbers):
  * VoI  : Value-of-Information greedy with noisy-OR (soft-confidence) fusion,
           Where2comm-style spatial-confidence selection [hu2022where2comm, krause2007voi].
           Distinct value model from SERVO (soft-OR vs hard-max) -> rewards a little
           redundancy, so it trails SERVO on importance-weighted coverage.
  * CTS  : Combinatorial Thompson Sampling online learner [wang2018cts] with Gaussian
           conjugate posteriors over per-(flow,event) quality and NOISY semi-bandit
           feedback -> a modern online competitor (replaces/augments plain-UCB MAB-CP).

Regenerates: fig_coverage_vs_budget, fig_value_vs_rho (now incl. VoI), fig_comprehensive
(9 methods), fig_drift (incl. CTS); prints Table III (correlated) and tab:indep
(importance-independent) for all methods + paired significance; dumps servo_v2_numbers.json.
"""
import sys, os, time, heapq, json
import numpy as np
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from servo_sim import (make_instance, value, covered_mask, metrics_for, ci95,
                       sel_servo, sel_wum, sel_wumrd, sel_osm, sel_pfprio, sel_random)
from servo_drl import train_drl, drl_select

SEEDS = list(range(20)); OUT = BASE; E = 45
STY = {"SERVO": ("#0E9C73", "o"), "VoI": ("#17A398", "D"), "WUM-RD": ("#E0883C", "s"),
       "OSM": ("#3C6E9C", "^"), "CTS": ("#8E5BA6", "*"), "MAB-CP": ("#B26E86", "v"),
       "WUM": ("#BBA53D", "P"), "PF/Prio": ("#C44E52", ">"), "Random": ("#7F7F7F", "X"),
       "DRL": ("#8C564B", "p"), "SERVO-NoEvo": ("#6FB39A", "o")}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.2,
    "legend.fontsize": 6.8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.9,
    "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e7eaed", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.framealpha": 0.9, "legend.edgecolor": "#cfd4d8", "lines.solid_capstyle": "round"})

def Lp(ax, x, y, m, ci=None, emph=False, label=None):
    c, mk = STY[m]; lw = 2.4 if emph else 1.5; ms = 5.4 if emph else 4.0
    x = np.asarray(x, float); y = np.asarray(y, float)
    if ci is not None:
        ci = np.asarray(ci, float); ax.fill_between(x, y - ci, y + ci, color=c, alpha=0.13, lw=0, zorder=2)
    ax.plot(x, y, color=c, lw=lw, marker=mk, ms=ms, mfc=c, mec="white", mew=0.8,
            label=(label or m), zorder=(6 if emph else 4))

def save(fig, n):
    fig.savefig(f"{OUT}/{n}.pdf"); fig.savefig(f"{OUT}/{n}.png"); plt.close(fig); print("  saved", n)

# ---------------- new selectors ----------------
def greedy_value(N, B, imp, cov, qq):
    S, vS = [], 0.0
    for _ in range(B):
        bk, bg = -1, -1e9
        for k in range(N):
            if k in S:
                continue
            g = value(S + [k], imp, cov, qq) - vS
            if g > bg:
                bg, bk = g, k
        if bk < 0:
            break
        S.append(bk); vS = value(S, imp, cov, qq)
    return S

def sel_voi(N, B, imp, cov, q):
    """Value-of-Information greedy: noisy-OR (soft-confidence) coverage fusion."""
    pm = np.ones(len(imp)); S = []
    for _ in range(B):
        bk, bg = -1, -1e9
        for k in range(N):
            if k in S:
                continue
            g = sum(imp[w] * pm[w] * q[(k, w)] for w in cov[k])
            if g > bg:
                bg, bk = g, k
        if bk < 0:
            break
        S.append(bk)
        for w in cov[bk]:
            pm[w] *= (1.0 - q[(bk, w)])
    return S

def sel_cts(N, B, imp, cov, q, T=40, seed=0):
    """Combinatorial Thompson Sampling: Gaussian-conjugate posteriors over q_k(w),
       noisy semi-bandit feedback; samples each slot and selects greedily."""
    rng = np.random.default_rng(5000 + seed)
    pm0, pv, ov = 0.8, 0.05, 0.12 ** 2
    mu = {}; prec = {}
    for k in range(N):
        for w in cov[k]:
            mu[(k, w)] = pm0; prec[(k, w)] = 1.0 / pv
    last = list(range(min(B, N)))
    for _ in range(T):
        qq = {}
        for kw, mk in mu.items():
            qq[kw] = min(1.0, max(0.0, rng.normal(mk, np.sqrt(1.0 / prec[kw]))))
        S = greedy_value(N, B, imp, cov, qq); last = S
        for k in S:
            for w in cov[k]:
                obs = q[(k, w)] + rng.normal(0, 0.12)
                p0, m0 = prec[(k, w)], mu[(k, w)]
                p1 = p0 + 1.0 / ov
                mu[(k, w)] = (m0 * p0 + obs / ov) / p1; prec[(k, w)] = p1
    return last

OFFLINE = {"SERVO": sel_servo, "VoI": sel_voi, "WUM-RD": sel_wumrd, "OSM": sel_osm,
           "WUM": sel_wum, "PF/Prio": sel_pfprio, "Random": sel_random}
LINE = ["SERVO", "VoI", "WUM-RD", "OSM", "WUM", "PF/Prio", "Random"]
COMP = ["SERVO", "VoI", "WUM-RD", "OSM", "CTS", "DRL", "MAB-CP", "WUM", "PF/Prio", "Random"]

# ---------------- comprehensive (all methods, with CI + robustness + runtime) ----------------
def comprehensive_all(correlate, P_drl, N=90, B=12):
    keys = ["value", "succ", "hi_sla", "lo_sla", "robust", "runtime"]
    acc = {m: {k: [] for k in keys} for m in COMP}
    paired = {"SERVO_VoI": [], "SERVO_WUMRD": []}
    for s in SEEDS:
        imp, cov, q, _ = make_instance(N, E, s, correlate=correlate); full = value(list(range(N)), imp, cov, q)
        rng = np.random.default_rng(1000 + s)
        qp = {kk: float(np.clip(vv * (1 + rng.uniform(-0.4, 0.4)), 0, 1)) for kk, vv in q.items()}
        vraw = {}
        for m, fn in OFFLINE.items():
            t0 = time.perf_counter()
            S = fn(N, B, imp, cov, q) if m != "Random" else fn(N, B, imp, cov, q, seed=s)
            rt = (time.perf_counter() - t0) * 1000
            v, succ, hi, lo = metrics_for(S, imp, cov, q, full, E)
            Sp = fn(N, B, imp, cov, qp) if m != "Random" else fn(N, B, imp, cov, qp, seed=s)
            rob = value(Sp, imp, cov, q) / full
            for kk, val in zip(keys, [v, succ, hi, lo, rob, rt]):
                acc[m][kk].append(val)
            vraw[m] = v
        # MAB-CP (plain UCB, online)
        T = 40; muu = np.zeros(N); cnt = np.full(N, 1e-6); tt = 0; lastm = list(range(B))
        t0 = time.perf_counter()
        for _ in range(T):
            tt += 1; ucb = muu + np.sqrt(2 * np.log(tt) / cnt); Sm = list(np.argsort(-ucb)[:B]); lastm = Sm
            for k in Sm:
                fb = value([k], imp, cov, q); cnt[k] += 1; muu[k] += (fb - muu[k]) / cnt[k]
        rtm = (time.perf_counter() - t0) * 1000 / T
        v, succ, hi, lo = metrics_for(lastm, imp, cov, q, full, E)
        for kk, val in zip(keys, [v, succ, hi, lo, v, rtm]):
            acc["MAB-CP"][kk].append(val)
        # CTS (online)
        t0 = time.perf_counter(); Sc = sel_cts(N, B, imp, cov, q, T=40, seed=s)
        rtc = (time.perf_counter() - t0) * 1000 / 40
        v, succ, hi, lo = metrics_for(Sc, imp, cov, q, full, E)
        for kk, val in zip(keys, [v, succ, hi, lo, v, rtc]):
            acc["CTS"][kk].append(val)
        # DRL-Sel (trained neural policy-gradient selector; offline inference)
        t0 = time.perf_counter(); Sd = drl_select(P_drl, N, B, imp, cov, q)
        rtd = (time.perf_counter() - t0) * 1000
        v, succ, hi, lo = metrics_for(Sd, imp, cov, q, full, E)
        robd = value(drl_select(P_drl, N, B, imp, cov, qp), imp, cov, q) / full
        for kk, val in zip(keys, [v, succ, hi, lo, robd, rtd]):
            acc["DRL"][kk].append(val)
        paired["SERVO_VoI"].append(vraw["SERVO"] - vraw["VoI"])
        paired["SERVO_WUMRD"].append(vraw["SERVO"] - vraw["WUM-RD"])
    summ = {m: {k: (float(np.nanmean(acc[m][k])), float(ci95([x for x in acc[m][k] if not np.isnan(x)])))
                for k in keys} for m in COMP}
    ps = {p: (float(np.mean(paired[p])), float(ci95(paired[p]))) for p in paired}
    return summ, ps

print("training DRL-Sel policies (cached after first run)...")
P_corr = train_drl(correlate=True, cache="_cache_drl_corr.pkl")
P_indep = train_drl(correlate=False, cache="_cache_drl_indep.pkl")
print("comprehensive (correlated)..."); comp_c, paired_c = comprehensive_all(True, P_corr)
print("comprehensive (independent)..."); comp_i, paired_i = comprehensive_all(False, P_indep)

def show(tag, summ, ps):
    print(f"\n== {tag} (N=90,E=45,B=12; mean +- 95%CI) ==")
    print("%-9s %13s %13s %13s %13s %13s %8s" % ("method", "Vbar", "Task", "Hi-SLA", "Lo-SLA", "Robust", "ms"))
    for m in COMP:
        r = summ[m]
        print("%-9s %s %s %s %s %s %7.2f" % (m,
              "%.3f+-%.3f" % r["value"], "%.3f+-%.3f" % r["succ"], "%.3f+-%.3f" % r["hi_sla"],
              "%.3f+-%.3f" % r["lo_sla"], "%.3f+-%.3f" % r["robust"], r["runtime"][0]))
    print("paired SERVO-VoI=%.4f+-%.4f  SERVO-WUMRD=%.4f+-%.4f" %
          (ps["SERVO_VoI"][0], ps["SERVO_VoI"][1], ps["SERVO_WUMRD"][0], ps["SERVO_WUMRD"][1]))
show("CORRELATED", comp_c, paired_c); show("INDEPENDENT", comp_i, paired_i)

# ---------------- dense budget & rho sweeps (incl. VoI) ----------------
print("\nbudget sweep..."); budgets = list(range(2, 33, 1))
e1 = {m: {"mean": [], "ci": []} for m in LINE}
for B in budgets:
    per = {m: [] for m in LINE}
    for s in SEEDS:
        imp, cov, q, _ = make_instance(90, E, s); full = value(list(range(90)), imp, cov, q)
        for m in LINE:
            S = OFFLINE[m](90, B, imp, cov, q) if m != "Random" else OFFLINE[m](90, B, imp, cov, q, seed=s)
            per[m].append(value(S, imp, cov, q) / full)
    for m in LINE:
        e1[m]["mean"].append(np.mean(per[m])); e1[m]["ci"].append(ci95(per[m]))

print("rho sweep..."); Ns = sorted(set(int(round(x)) for x in np.linspace(45, 340, 18))); B = 12
rhos = []; vr = {m: {"mean": [], "ci": []} for m in LINE}
for N in Ns:
    rhos.append(round(N * 3.2 / E, 1)); per = {m: [] for m in LINE}
    for s in SEEDS:
        imp, cov, q, _ = make_instance(N, E, s); full = value(list(range(N)), imp, cov, q)
        for m in LINE:
            S = OFFLINE[m](N, B, imp, cov, q) if m != "Random" else OFFLINE[m](N, B, imp, cov, q, seed=s)
            per[m].append(value(S, imp, cov, q) / full)
    for m in LINE:
        vr[m]["mean"].append(np.mean(per[m])); vr[m]["ci"].append(ci95(per[m]))

# ---------------- drift (incl. CTS) ----------------
print("drift...");
def run_drift(N=90, B=12, T=160, drift_slots=(53, 107), runs=10):
    track = {m: [] for m in ["SERVO", "SERVO-NoEvo", "WUM-RD", "MAB-CP", "CTS"]}
    regret = {m: [] for m in track}
    for rs in range(runs):
        imp0, cov, q, _ = make_instance(N, E, rs); imp_cur = imp0.copy(); imp_stale = imp0.copy()
        muu = np.zeros(N); cnt = np.full(N, 1e-6); tt = 0           # MAB
        cmu = {}; cprec = {}                                       # CTS
        for k in range(N):
            for w in cov[k]:
                cmu[(k, w)] = 0.8; cprec[(k, w)] = 1.0 / 0.05
        rng = np.random.default_rng(9000 + rs)
        per = {m: [] for m in track}; reg = {m: [] for m in track}; c = {m: 0.0 for m in track}
        for t in range(T):
            if t in drift_slots:
                perm = np.random.default_rng(7000 + rs * 10 + t).permutation(E); imp_cur = imp_cur[perm]
            fullc = value(list(range(N)), imp_cur, cov, q)
            vS = value(sel_servo(N, B, imp_cur, cov, q), imp_cur, cov, q)
            vN = value(sel_servo(N, B, imp_stale, cov, q), imp_cur, cov, q)
            vR = value(sel_wumrd(N, B, imp_cur, cov, q), imp_cur, cov, q)
            tt += 1; ucb = muu + np.sqrt(2 * np.log(tt) / cnt); Sm = list(np.argsort(-ucb)[:B])
            vM = value(Sm, imp_cur, cov, q)
            for k in Sm:
                fb = value([k], imp_cur, cov, q); cnt[k] += 1; muu[k] += (fb - muu[k]) / cnt[k]
            qq = {kw: min(1.0, max(0.0, rng.normal(cmu[kw], np.sqrt(1.0 / cprec[kw])))) for kw in cmu}
            Sc = greedy_value(N, B, imp_cur, cov, qq); vC = value(Sc, imp_cur, cov, q)
            for k in Sc:
                for w in cov[k]:
                    obs = q[(k, w)] + rng.normal(0, 0.12); p0, m0 = cprec[(k, w)], cmu[(k, w)]
                    p1 = p0 + 1.0 / 0.0144; cmu[(k, w)] = (m0 * p0 + obs / 0.0144) / p1; cprec[(k, w)] = p1
            for m, vv in zip(track, [vS, vN, vR, vM, vC]):
                per[m].append(vv / max(fullc, 1e-9)); c[m] += (fullc - vv); reg[m].append(c[m])
        for m in track:
            track[m].append(np.array(per[m])); regret[m].append(np.array(reg[m]))
    tm = {m: np.mean(np.vstack(track[m]), axis=0) for m in track}
    rm = {m: np.mean(np.vstack(regret[m]), axis=0) for m in track}
    return tm, rm, list(drift_slots)
track, regret, ds = run_drift()

# ================= render =================
print("rendering...")
# budget
fig, ax = plt.subplots(figsize=(3.5, 2.75))
for m in LINE:
    Lp(ax, budgets, e1[m]["mean"], m, ci=e1[m]["ci"], emph=(m == "SERVO"))
ax.set_xlabel("admission budget $B$ (flows)"); ax.set_ylabel(r"normalised delivered value $\bar V$")
ax.set_xlim(budgets[0], budgets[-1]); ax.set_ylim(0, 1.02)
ax.legend(ncol=2, loc="lower right", handlelength=1.5, columnspacing=0.9, labelspacing=0.25, borderpad=0.4)
save(fig, "fig_coverage_vs_budget")
# rho
fig, ax = plt.subplots(figsize=(3.5, 2.75))
for m in LINE:
    Lp(ax, rhos, vr[m]["mean"], m, ci=vr[m]["ci"], emph=(m == "SERVO"))
ax.set_xlabel(r"redundancy $\rho$ (mean flows per event)"); ax.set_ylabel(r"normalised delivered value $\bar V$")
ax.set_xlim(min(rhos), max(rhos)); ax.set_ylim(0, 1.02)
ax.legend(ncol=2, loc="lower left", handlelength=1.5, columnspacing=0.9, labelspacing=0.25, borderpad=0.4)
save(fig, "fig_value_vs_rho")
# comprehensive (9 methods, 6 panels)
panels = [("value", r"normalised value $\bar V$", False), ("succ", "semantic-task success", False),
          ("hi_sla", "high-tier SLA", False), ("lo_sla", "low-tier SLA", False),
          ("robust", r"robustness ($\pm$40\%)", False), ("runtime", "selection runtime (ms)", True)]
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3))
xs = np.arange(len(COMP))
for ax, (key, ttl, logy) in zip(axes.flat, panels):
    vals = [comp_c[m][key][0] for m in COMP]; errs = [comp_c[m][key][1] for m in COMP]
    cols = [STY[m][0] for m in COMP]
    edgec = ["#1a1a1a" if m == "SERVO" else "white" for m in COMP]
    ax.bar(xs, vals, color=cols, edgecolor=edgec, linewidth=[1.3 if m == "SERVO" else 0.5 for m in COMP], zorder=3)
    ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor="#3a3f45", elinewidth=0.8, capsize=2, zorder=8)
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, max(vals) * 1.22)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + max(vals) * 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=5.2, color="#333", zorder=9)
    ax.set_xticks(xs); ax.set_xticklabels(COMP, rotation=45, ha="right", fontsize=5.8)
    ax.set_title(ttl, fontsize=8.2)
fig.tight_layout(w_pad=1.4, h_pad=1.7)
save(fig, "fig_comprehensive")
# drift (2 panels incl CTS)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
T = len(next(iter(track.values()))); xT = np.arange(T); seq = ["SERVO", "SERVO-NoEvo", "WUM-RD", "MAB-CP", "CTS"]
for d in ds:
    a1.axvline(d, color="#ccd1d6", ls=":", lw=1.0, zorder=1); a2.axvline(d, color="#ccd1d6", ls=":", lw=1.0, zorder=1)
for m in seq:
    a1.plot(xT, track[m], color=STY[m][0], lw=(2.3 if m == "SERVO" else 1.5), label=m, zorder=(6 if m == "SERVO" else 4))
a1.set_xlabel("time slot"); a1.set_ylabel("delivered value / full"); a1.set_title("(a) Value tracking under drift", fontsize=8.6)
a1.legend(ncol=2, loc="lower left", handlelength=1.6)
for m in seq:
    a2.plot(xT, regret[m], color=STY[m][0], lw=(2.3 if m == "SERVO" else 1.5), label=m, zorder=(6 if m == "SERVO" else 4))
a2.set_xlabel("time slot"); a2.set_ylabel("cumulative value shortfall"); a2.set_title("(b) Cumulative regret", fontsize=8.6)
a2.legend(loc="upper left", handlelength=1.6)
fig.tight_layout(w_pad=1.8)
save(fig, "fig_drift")

# dump numbers
out = {"correlated": {m: {k: comp_c[m][k] for k in comp_c[m]} for m in COMP},
       "independent": {m: {k: comp_i[m][k] for k in comp_i[m]} for m in COMP},
       "paired_correlated": paired_c, "paired_independent": paired_i,
       "drift_final_regret": {m: float(regret[m][-1]) for m in track},
       "drift_mean_value": {m: float(track[m].mean()) for m in track}}
json.dump(out, open(f"{OUT}/servo_v2_numbers.json", "w"), indent=2)
print("\ndrift final regret:", {m: round(float(regret[m][-1]), 1) for m in track})
print("DONE")

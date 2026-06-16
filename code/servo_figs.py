#!/usr/bin/env python3
"""servo_figs.py — denser sweeps + clean publication-grade style for the six
parameter/scalability/redundancy figures.

Reuses the FAITHFUL selectors/oracle in servo_sim.py and the SAME 20 seeds, so the
operating points already reported in the paper are reproduced exactly; we only add
many more points (and render crisp vector PDF). No fabricated data.
Figures: fig_coverage_vs_budget, fig_value_vs_rho, fig_gain_vs_redundancy,
         fig_scalability, fig_diminishing, fig_approx_ratio.
"""
import sys, os, time, heapq
import numpy as np
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from servo_sim import (make_instance, value, ci95, sel_servo, sel_wum, sel_wumrd,
                       sel_osm, sel_pfprio, sel_random, sel_opt)

SEEDS = list(range(20)); OUT = BASE; E = 45
ORDER = ["SERVO", "WUM-RD", "OSM", "WUM", "PF/Prio", "Random"]
FN = {"SERVO": sel_servo, "WUM-RD": sel_wumrd, "OSM": sel_osm, "WUM": sel_wum,
      "PF/Prio": sel_pfprio, "Random": sel_random}
# crisp, colour-blind-friendly palette + distinct markers (B/W-readable)
STY = {"SERVO": ("#0E9C73", "o"), "WUM-RD": ("#E0883C", "s"), "OSM": ("#3C6E9C", "^"),
       "WUM": ("#7E6BB0", "D"), "MAB-CP": ("#B26E86", "v"), "PF/Prio": ("#C44E52", "P"),
       "Random": ("#7F7F7F", "X")}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7.0, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.9, "savefig.dpi": 600, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e7eaed", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.framealpha": 0.9, "legend.edgecolor": "#cfd4d8", "lines.solid_capstyle": "round",
})

def L(ax, x, y, m, ci=None, emph=False, label=None):
    c, mk = STY[m]; lw = 2.4 if emph else 1.5; ms = 5.4 if emph else 4.0
    x = np.asarray(x, float); y = np.asarray(y, float)
    if ci is not None:
        ci = np.asarray(ci, float)
        ax.fill_between(x, y - ci, y + ci, color=c, alpha=0.13, lw=0, zorder=2)
    ax.plot(x, y, color=c, lw=lw, marker=mk, ms=ms, mfc=c, mec="white", mew=0.8,
            label=(label or m), zorder=(6 if emph else 4))

def save(fig, n):
    fig.savefig(f"{OUT}/{n}.pdf"); fig.savefig(f"{OUT}/{n}.png"); plt.close(fig); print("  saved", n)

def sel_servo_plain(N, B, imp, cov, q):
    Sx, vS = [], 0.0
    for _ in range(B):
        bk, bg = -1, -1.0
        for k in range(N):
            if k in Sx:
                continue
            g = value(Sx + [k], imp, cov, q) - vS
            if g > bg:
                bg, bk = g, k
        if bk < 0:
            break
        Sx.append(bk); vS = value(Sx, imp, cov, q)
    return Sx

# ---------------- E1: value vs budget (31 points, was 8) ----------------
print("E1 value vs budget...")
budgets = list(range(2, 33, 1))
e1 = {m: {"mean": [], "ci": []} for m in ORDER}
for B in budgets:
    per = {m: [] for m in ORDER}
    for s in SEEDS:
        imp, cov, q, _ = make_instance(90, E, s); full = value(list(range(90)), imp, cov, q)
        for m in ORDER:
            Ssel = FN[m](90, B, imp, cov, q) if m != "Random" else FN[m](90, B, imp, cov, q, seed=s)
            per[m].append(value(Ssel, imp, cov, q) / full)
    for m in ORDER:
        e1[m]["mean"].append(np.mean(per[m])); e1[m]["ci"].append(ci95(per[m]))

# ---------------- E2/E3: value & gain vs redundancy (18 points, was 7) ----------------
print("E2/E3 value & gain vs rho...")
Ns = sorted(set(int(round(x)) for x in np.linspace(45, 340, 18)))
B = 12; rhos = []; vr = {m: {"mean": [], "ci": []} for m in ORDER}; gain_mean = []; gain_ci = []
for N in Ns:
    rhos.append(round(N * 3.2 / E, 1)); per = {m: [] for m in ORDER}; gains = []
    for s in SEEDS:
        imp, cov, q, _ = make_instance(N, E, s); full = value(list(range(N)), imp, cov, q)
        raw = {}
        for m in ORDER:
            Ssel = FN[m](N, B, imp, cov, q) if m != "Random" else FN[m](N, B, imp, cov, q, seed=s)
            rv = value(Ssel, imp, cov, q); raw[m] = rv; per[m].append(rv / full)
        gains.append(100.0 * (raw["SERVO"] - raw["WUM"]) / max(raw["WUM"], 1e-9))
    for m in ORDER:
        vr[m]["mean"].append(np.mean(per[m])); vr[m]["ci"].append(ci95(per[m]))
    gain_mean.append(np.mean(gains)); gain_ci.append(ci95(gains))

# ---------------- E4: scalability (10 points, was 6) ----------------
print("E4 scalability...")
Ns_sc = [60, 120, 180, 260, 340, 420, 500, 580, 640]
lazy, plain = [], []
for N in Ns_sc:
    tl, tp = [], []
    for s in range(8):
        imp, cov, q, _ = make_instance(N, E, s)
        t0 = time.perf_counter(); sel_servo(N, 16, imp, cov, q); tl.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); sel_servo_plain(N, 16, imp, cov, q); tp.append((time.perf_counter() - t0) * 1000)
    lazy.append(np.mean(tl)); plain.append(np.mean(tp))

# ---------------- E5: diminishing marginal gains (28 iters) ----------------
print("E5 diminishing...")
curves = []
for s in SEEDS:
    imp, cov, q, _ = make_instance(120, E, s); Sx, vp, g = [], 0.0, []
    heap = [(-value([k], imp, cov, q), k, 0) for k in range(120)]; heapq.heapify(heap)
    while heap and len(Sx) < 28:
        ng, k, st = heapq.heappop(heap)
        if st == len(Sx):
            gg = value(Sx + [k], imp, cov, q) - vp; Sx.append(k); vp += gg; g.append(gg)
        else:
            heapq.heappush(heap, (-(value(Sx + [k], imp, cov, q) - vp), k, len(Sx)))
    curves.append(g)
mlen = min(len(c) for c in curves); arr = np.array([c[:mlen] for c in curves])
dmean = arr.mean(0); dci = 1.96 * arr.std(0, ddof=1) / np.sqrt(len(arr))

# ---------------- E6: approximation ratio (200 instances, was 40) ----------------
print("E6 approx ratio...")
ratios = []
for s in range(200):
    imp, cov, q, _ = make_instance(16, 20, s)
    vg = value(sel_servo(16, 4, imp, cov, q), imp, cov, q)
    vo = value(sel_opt(16, 4, imp, cov, q), imp, cov, q)
    ratios.append(vg / max(vo, 1e-9))
ratios = np.array(ratios)

print("rendering figures...")
# 1) value vs budget
fig, ax = plt.subplots(figsize=(3.5, 2.75))
for m in ORDER:
    L(ax, budgets, e1[m]["mean"], m, ci=e1[m]["ci"], emph=(m == "SERVO"))
ax.set_xlabel("admission budget $B$ (flows)"); ax.set_ylabel(r"normalised delivered value $\bar V$")
ax.set_xlim(budgets[0], budgets[-1]); ax.set_ylim(0.0, 1.02)
ax.legend(ncol=2, loc="lower right", handlelength=1.6, columnspacing=1.0, labelspacing=0.25, borderpad=0.4)
save(fig, "fig_coverage_vs_budget")

# 2) value vs redundancy
fig, ax = plt.subplots(figsize=(3.5, 2.75))
for m in ORDER:
    L(ax, rhos, vr[m]["mean"], m, ci=vr[m]["ci"], emph=(m == "SERVO"))
ax.set_xlabel(r"redundancy $\rho$ (mean flows per event)"); ax.set_ylabel(r"normalised delivered value $\bar V$")
ax.set_xlim(min(rhos), max(rhos)); ax.set_ylim(0.0, 1.02)
ax.legend(ncol=2, loc="lower left", handlelength=1.6, columnspacing=1.0, labelspacing=0.25, borderpad=0.4)
save(fig, "fig_value_vs_rho")

# 3) gain vs redundancy
fig, ax = plt.subplots(figsize=(3.5, 2.75))
gm = np.array(gain_mean); gc = np.array(gain_ci)
ax.fill_between(rhos, gm - gc, gm + gc, color=STY["SERVO"][0], alpha=0.15, lw=0)
ax.plot(rhos, gm, color=STY["SERVO"][0], lw=2.4, marker="o", ms=5.0, mfc=STY["SERVO"][0],
        mec="white", mew=0.8, zorder=6)
ax.set_xlabel(r"redundancy $\rho$ (mean flows per event)"); ax.set_ylabel(r"SERVO value gain over WUM (\%)")
ax.set_xlim(min(rhos), max(rhos)); ax.set_ylim(0, max(gm + gc) * 1.18)
ax.annotate("gain grows\nwith redundancy", xy=(rhos[-2], gm[-2]), xytext=(rhos[len(rhos)//4], max(gm + gc) * 0.55),
            fontsize=7.5, color="#2a2f34", ha="center",
            arrowprops=dict(arrowstyle="-|>", color="#2a2f34", lw=1.0, shrinkA=2, shrinkB=4, mutation_scale=10))
save(fig, "fig_gain_vs_redundancy")

# 4) scalability
fig, ax = plt.subplots(figsize=(3.5, 2.75))
ax.plot(Ns_sc, plain, color=STY["PF/Prio"][0], lw=1.7, marker="s", ms=4.4, mfc=STY["PF/Prio"][0],
        mec="white", mew=0.8, label="plain greedy", zorder=4)
ax.plot(Ns_sc, lazy, color=STY["SERVO"][0], lw=2.4, marker="o", ms=5.0, mfc=STY["SERVO"][0],
        mec="white", mew=0.8, label="lazy greedy (CELF)", zorder=6)
ax.set_xlabel("number of flows $N$"); ax.set_ylabel("selection runtime (ms)")
ax.set_xlim(min(Ns_sc), max(Ns_sc)); ax.set_ylim(0, max(plain) * 1.18)
sp = plain[-1] / max(lazy[-1], 1e-9)
ax.annotate(rf"${sp:.1f}\times$ faster", xy=(Ns_sc[-1], lazy[-1]), xytext=(Ns_sc[-4], max(plain) * 0.62),
            fontsize=8.0, color=STY["SERVO"][0], fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="-|>", color=STY["SERVO"][0], lw=1.1, shrinkA=2, shrinkB=4, mutation_scale=10))
ax.legend(loc="upper left", handlelength=1.8, borderpad=0.4)
save(fig, "fig_scalability")

# 5) diminishing
fig, ax = plt.subplots(figsize=(3.5, 2.75))
x = np.arange(1, len(dmean) + 1)
ax.fill_between(x, dmean - dci, dmean + dci, color=STY["SERVO"][0], alpha=0.15, lw=0)
ax.plot(x, dmean, color=STY["SERVO"][0], lw=2.2, marker="o", ms=4.6, mfc=STY["SERVO"][0],
        mec="white", mew=0.8, zorder=6)
ax.set_xlabel("greedy iteration"); ax.set_ylabel(r"marginal value gain $\Delta(e\,|\,S)$")
ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.set_xlim(1, len(dmean)); ax.set_ylim(0, max(dmean) * 1.16)
ax.annotate("diminishing returns\n(submodular structure)", xy=(x[3], dmean[3]),
            xytext=(x[len(x)//2], max(dmean) * 0.72), fontsize=7.5, color="#2a2f34", ha="center",
            arrowprops=dict(arrowstyle="-|>", color="#2a2f34", lw=1.0, shrinkA=2, shrinkB=4, mutation_scale=10))
save(fig, "fig_diminishing")

# 6) approximation ratio
fig, ax = plt.subplots(figsize=(3.5, 2.75))
n, bins, patches = ax.hist(ratios, bins=22, color=STY["SERVO"][0], edgecolor="white", linewidth=0.7, zorder=3)
for p, h in zip(patches, n):
    p.set_alpha(0.45 + 0.5 * h / max(n))
ax.axvline(0.6321, color=STY["WUM-RD"][0], ls="--", lw=1.6, zorder=4)
ax.axvline(float(ratios.mean()), color="#2a2f34", ls=":", lw=1.4, zorder=4)
ax.set_xlabel(r"empirical ratio $V(S_{\mathrm{greedy}})/V(S^\star)$"); ax.set_ylabel("count")
ymax = max(n) * 1.16; ax.set_ylim(0, ymax)
ax.text(0.6321, ymax * 0.92, r"$1-1/e$", color=STY["WUM-RD"][0], fontsize=8, ha="center", fontweight="bold")
ax.text(float(ratios.mean()), ymax * 0.80, f"mean {ratios.mean():.3f}", color="#2a2f34", fontsize=7.5,
        ha="center", fontweight="bold")
save(fig, "fig_approx_ratio")

print("approx: mean=%.3f min=%.3f frac>1-1/e=%.0f%%  (n=%d)"
      % (ratios.mean(), ratios.min(), (ratios > 0.6321).mean() * 100, len(ratios)))
import json
json.dump({"budget12_servo": float(e1["SERVO"]["mean"][budgets.index(12)]),
           "gain_rho_min_pct": float(gain_mean[0]), "gain_rho_max_pct": float(gain_mean[-1]),
           "scal_N": int(Ns_sc[-1]), "scal_lazy_ms": float(lazy[-1]), "scal_plain_ms": float(plain[-1]),
           "scal_speedup": float(plain[-1] / max(lazy[-1], 1e-9)),
           "approx_mean": float(ratios.mean()), "approx_min": float(ratios.min())},
          open(f"{OUT}/servo_figs_numbers.json", "w"), indent=2)
print("scalability @N=%d: lazy=%.1f ms, plain=%.1f ms, speedup=%.1fx"
      % (Ns_sc[-1], lazy[-1], plain[-1], plain[-1] / max(lazy[-1], 1e-9)))
print("DONE")

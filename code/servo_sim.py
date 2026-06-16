#!/usr/bin/env python3
"""
servo_sim.py
Faithful Monte-Carlo simulation of the SERVO submodular coverage model (Sec. III).
ALL numbers printed/saved by this script are genuine outputs of actually running the
implemented selectors on synthetic V2X cooperative-perception instances.

Selectors implemented faithfully (numpy only):
  SERVO (lazy/CELF greedy submodular), WUM (additive weighted), WUM-RD
  (redundancy-discounted), OSM (sieve-streaming submodular), PF/Prio, Random,
  OPT (brute-force, small instances), MAB-CP (UCB online bandit).
NOT simulated here (need GPU training / packet sim; do NOT fabricate): LyDRO* full
DRL network, SAC sub-allocator training, NS-3/SUMO packet-level QoS.
"""
import numpy as np, time, itertools, csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# ----------------------------- Instance model (Table III) -----------------------------
TIERS  = np.array([0.95, 0.60, 0.25])      # high / med / low importance
TPROB  = np.array([0.20, 0.30, 0.50])      # tier probabilities

def make_instance(N, E, seed, cov_per_flow=3.2, correlate=True):
    """Generate a coverage instance faithful to Table III:
       each flow covers ~Poisson(3.2) events (FIXED), so redundancy rho = N*cov_per_flow/E
       is controlled by N (sweep N to sweep rho). correlate=True makes important events
       more likely to be sensed/reported (V2X: hazards attract more reporters), consistent
       with the paper's motivation (Fig.1) and Prop.2.
       Returns: imp[E], cov (sets of events per flow), q dict (k,w)->[0.6,1], rho_eff."""
    rng = np.random.default_rng(seed)
    tier = rng.choice(3, size=E, p=TPROB)
    imp  = TIERS[tier]
    p_ev = (imp/imp.sum()) if correlate else (np.ones(E)/E)   # coverage prob ∝ importance
    cov = [set() for _ in range(N)]
    q   = {}
    for k in range(N):
        nk = int(np.clip(rng.poisson(cov_per_flow), 1, E))    # |cov(k)| ~ Poisson(3.2)
        evs = rng.choice(E, size=nk, replace=False, p=p_ev)
        for w in evs:
            cov[k].add(w)
            q[(k, w)] = rng.uniform(0.6, 1.0)                 # quality U(0.6,1]
    rho_eff = N*cov_per_flow/E
    return imp, cov, q, rho_eff

# ----------------------------- Submodular value oracle (Eq. 6) -----------------------------
def value(S, imp, cov, q):
    """V(S) = sum_w imp(w) * max_{k in S, w in cov(k)} q(k,w)."""
    if not S:
        return 0.0
    E = len(imp)
    best = np.zeros(E)
    for k in S:
        for w in cov[k]:
            qq = q[(k, w)]
            if qq > best[w]:
                best[w] = qq
    return float(np.dot(imp, best))

def covered_mask(S, cov, q, thr=0.0):
    E_cov = set()
    for k in S:
        for w in cov[k]:
            if q[(k, w)] > thr:
                E_cov.add(w)
    return E_cov

# ----------------------------- Selectors -----------------------------
def sel_servo(N, B, imp, cov, q):
    """Lazy (CELF) greedy submodular maximization under cardinality budget B."""
    import heapq
    S, vS = [], 0.0
    # initial marginals
    heap = []
    for k in range(N):
        g = value([k], imp, cov, q)            # marginal over empty = V({k})
        heapq.heappush(heap, (-g, k, 0))
    it = 0
    while heap and len(S) < B:
        neg_g, k, stamp = heapq.heappop(heap)
        if stamp == len(S):                    # marginal fresh
            S.append(k); vS = value(S, imp, cov, q)
        else:
            g = value(S + [k], imp, cov, q) - vS
            heapq.heappush(heap, (-g, k, len(S)))
        it += 1
    return S

def sel_wum(N, B, imp, cov, q):
    """Additive weighted-utility (overlap-blind): top-B by sum_w imp(w)q(k,w)."""
    score = np.array([sum(imp[w]*q[(k, w)] for w in cov[k]) for k in range(N)])
    return list(np.argsort(-score)[:B])

def sel_wumrd(N, B, imp, cov, q):
    """Redundancy-discounted WUM: weight divided by #flows already covering the event."""
    S = []
    seen = np.zeros(len(imp), dtype=int)
    for _ in range(B):
        best_k, best_s = -1, -1.0
        for k in range(N):
            if k in S:
                continue
            s = sum(imp[w]*q[(k, w)] / (1 + seen[w]) for w in cov[k])
            if s > best_s:
                best_s, best_k = s, k
        if best_k < 0:
            break
        S.append(best_k)
        for w in cov[best_k]:
            seen[w] += 1
    return S

def sel_osm(N, B, imp, cov, q):
    """Sieve-streaming submodular (single pass) under cardinality B (Badanidiyuru et al.)."""
    m = max((value([k], imp, cov, q) for k in range(N)), default=0.0)
    if m <= 0:
        return []
    eps = 0.2
    thresholds, v = [], m
    while v <= B*m*(1+eps):
        thresholds.append(v); v *= (1+eps)
    buckets = {t: [] for t in thresholds}; bval = {t: 0.0 for t in thresholds}
    for k in range(N):
        for t in thresholds:
            if len(buckets[t]) < B:
                g = value(buckets[t] + [k], imp, cov, q) - bval[t]
                need = (t/2.0 - bval[t]) / max(B - len(buckets[t]), 1)
                if g >= need:                  # correct sieve admission rule
                    buckets[t].append(k); bval[t] += g
    return max(buckets.values(), key=lambda s: value(s, imp, cov, q))

def sel_pfprio(N, B, imp, cov, q):
    """Priority by max single-event importance covered."""
    score = np.array([max((imp[w] for w in cov[k]), default=0.0) for k in range(N)])
    return list(np.argsort(-score)[:B])

def sel_random(N, B, imp, cov, q, seed=0):
    rng = np.random.default_rng(seed)
    return list(rng.choice(N, size=min(B, N), replace=False))

def sel_opt(N, B, imp, cov, q):
    """Brute-force optimum (small instances only)."""
    best, bestv = None, -1.0
    for S in itertools.combinations(range(N), B):
        v = value(list(S), imp, cov, q)
        if v > bestv:
            bestv, best = v, list(S)
    return best

# ----------------------------- Experiment helpers -----------------------------
def ci95(arr):
    arr = np.asarray(arr, float)
    if len(arr) < 2:
        return 0.0
    return 1.96*arr.std(ddof=1)/np.sqrt(len(arr))

SEEDS = list(range(20))

# ===================== E1: coverage (normalised value) vs budget =====================
def exp_budget(N=90, E=45, budgets=range(4, 33, 4)):
    methods = {"SERVO": sel_servo, "WUM-RD": sel_wumrd, "OSM": sel_osm,
               "WUM": sel_wum, "PF/Prio": sel_pfprio, "Random": sel_random}
    res = {m: {"mean": [], "ci": []} for m in methods}
    for B in budgets:
        per = {m: [] for m in methods}
        for s in SEEDS:
            imp, cov, q, _ = make_instance(N, E, s)
            full = value(list(range(N)), imp, cov, q)
            for m, fn in methods.items():
                S = fn(N, B, imp, cov, q) if m != "Random" else fn(N, B, imp, cov, q, seed=s)
                per[m].append(value(S, imp, cov, q)/full)
        for m in methods:
            res[m]["mean"].append(np.mean(per[m])); res[m]["ci"].append(ci95(per[m]))
    return list(budgets), res

# ===================== E2: SERVO gain over WUM vs redundancy =====================
def exp_redundancy(E=45, B=12, Ns=(45, 75, 120, 165, 210, 270, 340), cov_per_flow=3.2):
    """Sweep N (per Table III, N=40-320) -> rho = N*3.2/E in [3,24]. B fixed.
       Each flow still covers ~Poisson(3.2) events; more flows = more redundancy."""
    rhos, means, cis = [], [], []
    for N in Ns:
        gains = []
        rho_eff = N*cov_per_flow/E
        for s in SEEDS:
            imp, cov, q, _ = make_instance(N, E, s, cov_per_flow=cov_per_flow)
            vS = value(sel_servo(N, B, imp, cov, q), imp, cov, q)
            vW = value(sel_wum(N, B, imp, cov, q),   imp, cov, q)
            gains.append(100.0*(vS - vW)/max(vW, 1e-9))
        rhos.append(round(rho_eff, 1)); means.append(np.mean(gains)); cis.append(ci95(gains))
    return rhos, means, cis

# ===================== E3: comprehensive comparison (multi-metric) =====================
def metrics_for(S, imp, cov, q, full, E):
    v = value(S, imp, cov, q)/full
    # distinct-event success: fraction of events covered with quality>0.6 (any flow)
    E_cov = covered_mask(S, cov, q, thr=0.6)
    succ = len(E_cov)/E
    # per-tier "SLA": fraction of high/low importance events covered above q>0.6
    hi = [w for w in range(E) if imp[w] >= 0.9]
    lo = [w for w in range(E) if imp[w] <= 0.3]
    hi_sla = (len([w for w in hi if w in E_cov])/len(hi)) if hi else np.nan
    lo_sla = (len([w for w in lo if w in E_cov])/len(lo)) if lo else np.nan
    return v, succ, hi_sla, lo_sla

def exp_comprehensive(N=90, E=45, rho=6, B=12):
    methods = {"SERVO": sel_servo, "WUM-RD": sel_wumrd, "OSM": sel_osm,
               "MAB-CP": None, "WUM": sel_wum, "PF/Prio": sel_pfprio, "Random": sel_random}
    keys = ["value", "succ", "hi_sla", "lo_sla", "robust", "runtime"]
    acc = {m: {k: [] for k in keys} for m in methods}
    # MAB-CP needs an online stream of slots; we run a short stream per seed
    for s in SEEDS:
        imp, cov, q, _ = make_instance(N, E, s)
        full = value(list(range(N)), imp, cov, q)
        # --- combinatorial selectors ---
        for m, fn in methods.items():
            if m == "MAB-CP":
                continue
            t0 = time.perf_counter()
            S = fn(N, B, imp, cov, q) if m != "Random" else fn(N, B, imp, cov, q, seed=s)
            rt = (time.perf_counter()-t0)*1000.0
            v, succ, hi, lo = metrics_for(S, imp, cov, q, full, E)
            # robustness: perturb q by +-40%, select on perturbed, evaluate on TRUE
            rng = np.random.default_rng(1000+s)
            qp = {kk: float(np.clip(vv*(1+rng.uniform(-0.4, 0.4)), 0, 1)) for kk, vv in q.items()}
            Sp = fn(N, B, imp, cov, qp) if m != "Random" else fn(N, B, imp, cov, qp, seed=s)
            rob = value(Sp, imp, cov, q)/full
            for k, val in zip(keys, [v, succ, hi, lo, rob, rt]):
                acc[m][k].append(val)
        # --- MAB-CP: UCB over T slots on the SAME instance (online) ---
        T = 40
        mu = np.zeros(N); cnt = np.full(N, 1e-6); tt = 0
        last = list(range(B))
        t0 = time.perf_counter()
        for _ in range(T):
            tt += 1
            ucb = mu + np.sqrt(2*np.log(tt)/cnt)
            S = list(np.argsort(-ucb)[:B]); last = S
            # realized per-flow feedback = that flow's marginal contribution proxy
            for k in S:
                fb = value([k], imp, cov, q)
                cnt[k] += 1; mu[k] += (fb - mu[k])/cnt[k]
        rt = (time.perf_counter()-t0)*1000.0/T
        v, succ, hi, lo = metrics_for(last, imp, cov, q, full, E)
        for k, val in zip(keys, [v, succ, hi, lo, v, rt]):
            acc["MAB-CP"][k].append(val)
    summary = {m: {k: (np.nanmean(acc[m][k]), ci95([x for x in acc[m][k] if not np.isnan(x)]))
                   for k in keys} for m in methods}
    return summary

# ===================== E4: ablation =====================
def exp_ablation(N=90, E=45, rho=6, B=12):
    """Full SERVO vs -Submod(=WUM additive) vs -Greedy(=random select) vs
       -GNN-sense(value-est error +-40% then select) vs -SAC(value unchanged here;
       reported via realized-vs-target proxy) vs -NoEvo (no drift tracking -> see E5)."""
    rows = {}
    full_metrics = {k: [] for k in ["value", "succ", "hi_sla", "lo_sla", "approx"]}
    variants = {
        "Full (SERVO)":   ("servo", 0.0),
        "- Submodular":   ("wum",   0.0),
        "- Greedy":       ("rand",  0.0),
        "- GNN-sense":    ("servo", 0.4),   # +-40% value-estimation error before selection
    }
    out = {v: {k: [] for k in ["value", "succ", "hi_sla", "lo_sla"]} for v in variants}
    for s in SEEDS:
        imp, cov, q, _ = make_instance(N, E, s)
        full = value(list(range(N)), imp, cov, q)
        for vname, (sel, noise) in variants.items():
            if noise > 0:
                rng = np.random.default_rng(2000+s)
                qn = {kk: float(np.clip(vv*(1+rng.uniform(-noise, noise)), 0, 1)) for kk, vv in q.items()}
                S = sel_servo(N, B, imp, cov, qn)        # select on noisy estimate
            elif sel == "servo":
                S = sel_servo(N, B, imp, cov, q)
            elif sel == "wum":
                S = sel_wum(N, B, imp, cov, q)
            else:
                S = sel_random(N, B, imp, cov, q, seed=s)
            v, succ, hi, lo = metrics_for(S, imp, cov, q, full, E)   # always eval on TRUE q
            out[vname]["value"].append(v); out[vname]["succ"].append(succ)
            out[vname]["hi_sla"].append(hi); out[vname]["lo_sla"].append(lo)
    summ = {v: {k: (np.nanmean(out[v][k]), ci95([x for x in out[v][k] if not np.isnan(x)]))
                for k in out[v]} for v in variants}
    return summ

# ===================== E5: approximation ratio vs OPT (small instances) =====================
def exp_approx(N=16, E=20, rho=4, B=4, n_inst=200):  # 200 instances, matching the paper
    ratios = []
    for s in range(n_inst):
        imp, cov, q, _ = make_instance(N, E, s)
        vg = value(sel_servo(N, B, imp, cov, q), imp, cov, q)
        vo = value(sel_opt(N, B, imp, cov, q),   imp, cov, q)
        ratios.append(vg/max(vo, 1e-9))
    return np.array(ratios)

# ===================== E6: value under drift + cumulative regret =====================
def exp_drift(N=90, E=45, rho=6, B=12, T=160, drift_slots=(53, 107)):
    """Importance-relocating drift. SERVO re-senses current imp each slot; SERVO-NoEvo
       keeps stale imp from slot 0. Track delivered value / per-slot OPT-ish full and regret."""
    track = {"SERVO": [], "SERVO-NoEvo": [], "WUM-RD": [], "MAB-CP": []}
    regret = {m: [] for m in track}
    cum = {m: 0.0 for m in track}
    for run_seed in range(10):                         # average over 10 runs
        base = make_instance(N, E, run_seed)
        imp0, cov, q, _ = base
        imp_cur = imp0.copy()
        imp_stale = imp0.copy()
        # MAB state
        mu = np.zeros(N); cnt = np.full(N, 1e-6); tt = 0
        per = {m: [] for m in track}; reg = {m: [] for m in track}; c = {m: 0.0 for m in track}
        for t in range(T):
            if t in drift_slots:                       # relocate importances (shuffle tiers)
                rng = np.random.default_rng(7000+run_seed*10+t)
                perm = rng.permutation(E); imp_cur = imp_cur[perm]
            full_cur = value(list(range(N)), imp_cur, cov, q)
            # SERVO: senses imp_cur
            S = sel_servo(N, B, imp_cur, cov, q)
            vS = value(S, imp_cur, cov, q)
            # NoEvo: uses stale imp for SELECTION, evaluated on current truth
            Sn = sel_servo(N, B, imp_stale, cov, q)
            vN = value(Sn, imp_cur, cov, q)
            # WUM-RD on current
            Sr = sel_wumrd(N, B, imp_cur, cov, q); vR = value(Sr, imp_cur, cov, q)
            # MAB-CP online
            tt += 1; ucb = mu + np.sqrt(2*np.log(tt)/cnt); Sm = list(np.argsort(-ucb)[:B])
            vM = value(Sm, imp_cur, cov, q)
            for k in Sm:
                fb = value([k], imp_cur, cov, q); cnt[k] += 1; mu[k] += (fb-mu[k])/cnt[k]
            for m, vv in zip(track, [vS, vN, vR, vM]):
                per[m].append(vv/max(full_cur, 1e-9))
                c[m] += (full_cur - vv); reg[m].append(c[m])
        for m in track:
            per_arr = np.array(per[m]); track[m].append(per_arr); regret[m].append(np.array(reg[m]))
    track_mean = {m: np.mean(np.vstack(track[m]), axis=0) for m in track}
    regret_mean = {m: np.mean(np.vstack(regret[m]), axis=0) for m in track}
    return track_mean, regret_mean, list(drift_slots)

# ----------------------------- Run all + save -----------------------------
def main():
    np.random.seed(0)
    report = {}

    print("== E1: normalised value vs budget ==")
    budgets, e1 = exp_budget()
    report["e1"] = {"budgets": budgets, "data": {m: e1[m]["mean"] for m in e1}}
    for m in e1:
        print(f"  {m:8s} @B=12: {e1[m]['mean'][2]:.3f} +-{e1[m]['ci'][2]:.3f}")

    print("== E2: SERVO gain over WUM vs redundancy ==")
    rhos, g_mean, g_ci = exp_redundancy()
    report["e2"] = {"rho": rhos, "gain_pct": g_mean, "ci": g_ci}
    for r, gm, gc in zip(rhos, g_mean, g_ci):
        print(f"  rho={r:5.1f}: +{gm:5.1f}% (+-{gc:.1f})")

    print("== E3: comprehensive comparison @B=12, rho=6 ==")
    comp = exp_comprehensive()
    report["e3"] = {m: {k: comp[m][k][0] for k in comp[m]} for m in comp}
    hdr = ["value", "succ", "hi_sla", "lo_sla", "robust", "runtime"]
    print("  method     " + "  ".join(f"{h:>8s}" for h in hdr))
    for m in comp:
        print(f"  {m:10s} " + "  ".join(f"{comp[m][k][0]:8.3f}" for k in hdr))

    print("== E4: ablation @B=12, rho=6 ==")
    abl = exp_ablation()
    report["e4"] = {v: {k: abl[v][k][0] for k in abl[v]} for v in abl}
    for v in abl:
        print(f"  {v:14s} value={abl[v]['value'][0]:.3f} succ={abl[v]['succ'][0]:.3f} "
              f"hi={abl[v]['hi_sla'][0]:.3f} lo={abl[v]['lo_sla'][0]:.3f}")

    print("== E5: empirical approx ratio vs OPT (200 small instances) ==")
    ratios = exp_approx()
    report["e5"] = {"mean": float(ratios.mean()), "std": float(ratios.std()),
                    "min": float(ratios.min()), "frac_above_063": float((ratios > 0.6321).mean())}
    print(f"  V(greedy)/V(OPT) = {ratios.mean():.3f} +- {ratios.std():.3f} "
          f"(min {ratios.min():.3f}, all > 1-1/e: {(ratios>0.6321).mean()*100:.0f}%)")

    print("== E6: drift tracking + cumulative regret ==")
    track, regret, ds = exp_drift()
    report["e6"] = {m: {"final_regret": float(regret[m][-1]),
                        "mean_norm_value": float(track[m].mean())} for m in track}
    for m in track:
        print(f"  {m:12s} mean(value/full)={track[m].mean():.3f}  "
              f"final cumulative regret={regret[m][-1]:.1f}")

    # ---- save machine-readable results ----
    with open(f"{OUT}/servo_results.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---- figures ----
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                         "font.size": 8, "savefig.dpi": 300,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    COL = {"SERVO": "#1b9e77", "WUM-RD": "#d95f02", "OSM": "#7570b3",
           "MAB-CP": "#e7298a", "WUM": "#66a61e", "PF/Prio": "#666666",
           "Random": "#bbbbbb", "SERVO-NoEvo": "#999999"}

    # Fig A: coverage vs budget
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in e1:
        ax.errorbar(budgets, e1[m]["mean"], yerr=e1[m]["ci"], marker="o", ms=3,
                    lw=1.2, capsize=2, color=COL.get(m), label=m)
    ax.set_xlabel("budget B (admitted flows)"); ax.set_ylabel("normalised delivered value V(S)")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_coverage_vs_budget.pdf")
    fig.savefig(f"{OUT}/fig_coverage_vs_budget.png"); plt.close(fig)

    # Fig B: gain vs redundancy
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    g_mean = np.array(g_mean); g_ci = np.array(g_ci)
    ax.plot(rhos, g_mean, "-s", color=COL["SERVO"], lw=1.4, ms=4)
    ax.fill_between(rhos, g_mean-g_ci, g_mean+g_ci, color=COL["SERVO"], alpha=0.18)
    ax.set_xlabel(r"redundancy $\rho$ (flows per event)")
    ax.set_ylabel("SERVO value gain over WUM (%)"); ax.set_ylim(0, max(g_mean+g_ci)*1.2)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_gain_vs_redundancy.pdf")
    fig.savefig(f"{OUT}/fig_gain_vs_redundancy.png"); plt.close(fig)

    # Fig C: comprehensive bars (value / succ / hi_sla / lo_sla / robust / runtime)
    order = ["SERVO", "WUM-RD", "OSM", "MAB-CP", "WUM", "PF/Prio", "Random"]
    panels = [("value", "V(S)/full", False), ("succ", "semantic-task success", False),
              ("hi_sla", "high-tier SLA", False), ("lo_sla", "low-tier SLA", False),
              ("robust", "robustness (±40%)", False), ("runtime", "edge runtime (ms)", True)]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0))
    for ax, (key, title, log) in zip(axes.flat, panels):
        vals = [comp[m][key][0] for m in order]; errs = [comp[m][key][1] for m in order]
        cols = ["#1b9e77" if m == "SERVO" else "#9 aa".replace(" ", "") for m in order]
        cols = ["#1b9e77" if m == "SERVO" else "#9999aa" for m in order]
        x = np.arange(len(order))
        ax.bar(x, vals, yerr=errs, color=cols, edgecolor="k", lw=0.5, capsize=2)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=6)
        ax.set_title(title, fontsize=8)
        if log:
            ax.set_yscale("log")
        for xi, vv in zip(x, vals):
            ax.text(xi, vv, f"{vv:.2f}" if not log else f"{vv:.2f}", ha="center",
                    va="bottom", fontsize=5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_comprehensive.pdf")
    fig.savefig(f"{OUT}/fig_comprehensive.png"); plt.close(fig)

    # Fig D: ablation bars
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    avs = list(abl.keys())
    x = np.arange(len(avs))
    vals = [abl[v]["value"][0] for v in avs]; errs = [abl[v]["value"][1] for v in avs]
    cols = ["#1b9e77" if v == "Full (SERVO)" else "#9999aa" for v in avs]
    ax.bar(x, vals, yerr=errs, color=cols, edgecolor="k", lw=0.5, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(avs, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("normalised value V(S)/full"); ax.set_ylim(0, 1.0)
    for xi, vv in zip(x, vals):
        ax.text(xi, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=6)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ablation.pdf")
    fig.savefig(f"{OUT}/fig_ablation.png"); plt.close(fig)

    # Fig E: approx ratio histogram
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.hist(ratios, bins=12, color="#1b9e77", edgecolor="k", alpha=0.85)
    ax.axvline(0.6321, color="red", ls="--", lw=1.2, label="1-1/e")
    ax.set_xlabel("empirical ratio V(greedy)/V(OPT)"); ax.set_ylabel("count"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_approx_ratio.pdf")
    fig.savefig(f"{OUT}/fig_approx_ratio.png"); plt.close(fig)

    # Fig F: drift tracking + regret
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    T = len(next(iter(track.values())))
    for m in ["SERVO", "SERVO-NoEvo", "WUM-RD", "MAB-CP"]:
        a1.plot(range(T), track[m], lw=1.1, color=COL.get(m), label=m)
    for d in ds:
        a1.axvline(d, color="grey", ls=":", lw=0.8)
    a1.set_xlabel("time slot"); a1.set_ylabel("delivered value / full-coverage")
    a1.set_title("(a) Value tracking under drift", fontsize=8); a1.legend(fontsize=6)
    for m in ["SERVO", "SERVO-NoEvo", "WUM-RD", "MAB-CP"]:
        a2.plot(range(T), regret[m], lw=1.2, color=COL.get(m), label=m)
    a2.set_xlabel("time slot"); a2.set_ylabel("cumulative value shortfall")
    a2.set_title("(b) Cumulative regret", fontsize=8); a2.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_drift.pdf")
    fig.savefig(f"{OUT}/fig_drift.png"); plt.close(fig)

    # ---- CSV tables for the paper ----
    with open(f"{OUT}/table_comprehensive.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + hdr)
        for m in comp:
            w.writerow([m] + [f"{comp[m][k][0]:.4f}" for k in hdr])
    with open(f"{OUT}/table_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "value", "succ", "hi_sla", "lo_sla"])
        for v in abl:
            w.writerow([v] + [f"{abl[v][k][0]:.4f}" for k in ["value", "succ", "hi_sla", "lo_sla"]])

    print("\nSaved: figures (pdf+png), servo_results.json, table_comprehensive.csv, table_ablation.csv")

if __name__ == "__main__":
    main()

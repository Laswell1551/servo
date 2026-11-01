"""Corrected-physics, evidence-producing SERVO consistency revision.

No legacy result is overwritten. Run from the project root:
  python experiments/revision_consistency.py --train-steps 36000
The independent snapshot model has no queue/carry-over claim. Serial latency
includes measured orchestration time; component-only results are retained too.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import cvxpy as cp
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from scipy.stats import t as student_t
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import e2e_servo as b

ROOT = b.ROOT
OUT = ROOT / "experiments/results_consistency"
CKPT = ROOT / "experiments/checkpoints_consistency"
MAX_B = 16


def context(scene, selected, q, imp, include_truth=False, with_credit=True):
    ctx = b._allocation_arrays(scene, selected, MAX_B, q, imp)
    if not with_credit and not include_truth:
        return ctx
    m, nw = len(selected), len(scene.event_x)
    pred = np.zeros((MAX_B, nw)); truth = np.zeros_like(pred)
    for j, k in enumerate(selected):
        for w, e in scene.flow_event_edges[k].items():
            pred[j, w] = q[e]
            if include_truth: truth[j, w] = scene.q_true[e]
    best = pred[:m].max(axis=0) if m else np.zeros(nw)
    credit = np.zeros(MAX_B)
    total = max(float(best @ imp), 1e-12)
    for j in range(m):
        others = np.delete(pred[:m], j, axis=0)
        without = others.max(axis=0) if len(others) else np.zeros(nw)
        credit[j] = float((best - without) @ imp) / total
    ctx.update(pred_matrix=pred, event_imp=np.asarray(imp), coverage_credit=credit)
    if include_truth:
        ctx.update(truth_matrix=truth, truth_imp=scene.imp_true.copy())
    return ctx


def observation(ctx):
    old = b.allocation_observation(ctx, MAX_B)
    return np.concatenate([old, ctx["coverage_credit"]]).astype(np.float32)


def soft_coverage(ctx, satisfaction):
    m = int(ctx["m"][0])
    quality = ctx["truth_matrix"][:m]
    den = float(quality.max(axis=0) @ ctx["truth_imp"]) if m else 0.0
    num = float((quality * satisfaction[:, None]).max(axis=0) @ ctx["truth_imp"]) if m else 0.0
    return num / den if den > 0 else 0.0


class CoverageEnv(gym.Env):
    def __init__(self, pool, reward_kind, seed=2026):
        self.pool = pool; self.kind = reward_kind
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1, 1, (2 * MAX_B,), np.float32)
        self.observation_space = spaces.Box(-2, 2, (12 * MAX_B + 3,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ctx = self.pool[int(self.rng.integers(len(self.pool)))]
        return observation(self.ctx), {}

    def step(self, action):
        p, f = b.action_to_resources(action, self.ctx, MAX_B)
        out = b.allocation_outcome(self.ctx, p, f)
        m = int(self.ctx["m"][0]); lat = np.asarray(out["latency"]) + 0.005
        deadline = self.ctx["deadline"][:m]
        sl = 1 / (1 + np.exp(np.clip(8 * (lat / deadline - 1), -30, 30)))
        sr = 1 / (1 + np.exp(np.clip(-2 * (out["sinr_db"] - self.ctx["rel_db"][:m]), -30, 30)))
        satisfaction = sl * sr
        if self.kind == "coverage":
            utility = soft_coverage(self.ctx, satisfaction)
        else:
            weights = self.ctx["importance"][:m] * self.ctx["q"][:m]
            utility = float(weights @ satisfaction / max(weights.sum(), 1e-12))
        complete = (lat <= deadline) & (out["sinr_db"] >= self.ctx["rel_db"][:m])
        reward = 1.6 * utility + 0.7 * complete.mean() \
            - 0.010 * min(float(np.mean(out["energy"])), 8) \
            - 0.25 * min(float(np.maximum(lat / deadline - 1, 0).mean()), 4)
        return observation(self.ctx), float(reward), True, False, {}


def train_models(gnn, steps, force=False):
    models = {}
    missing = [kind for kind in ["coverage", "additive"]
               if force or not (CKPT / f"sac_{kind}_{steps}.zip").exists()]
    pool = []
    if missing:
        rng = np.random.default_rng(2026)
        for j in range(384):
            cfg = b.SimConfig(n_vehicles=int(rng.choice([36, 72, 128])),
                              n_events=int(rng.choice([20, 36, 64])),
                              n_nodes=int(rng.choice([4, 8])),
                              admission_budget=int(rng.integers(4, MAX_B + 1)))
            spec = b.Scenario(capacity_cv=float(rng.uniform(0, .65)),
                              congestion_multiplier=float(rng.uniform(.8, 1.8)))
            sc = b.make_scene(10000 + j, j % 120, cfg, spec)
            q, imp = b.predict_scene(gnn, sc)
            chosen = b.select_flows(sc, q, imp, cfg.admission_budget, "submodular", rng)
            if chosen:
                pool.append(context(sc, chosen, q, imp, include_truth=True))
        print(f"Training contexts: {len(pool)}; max padded admission={MAX_B}", flush=True)
    for kind in ["coverage", "additive"]:
        path = CKPT / f"sac_{kind}_{steps}"
        if kind in missing:
            b.seed_everything(2026)
            env = CoverageEnv(pool, kind)
            model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=max(steps, 5000),
                        learning_starts=min(1000, steps // 5), batch_size=128,
                        gamma=0, tau=.01, train_freq=1, gradient_steps=1,
                        policy_kwargs={"net_arch": [128, 128]}, seed=2026, verbose=0)
            print(f"Training {kind} reward, {steps} steps", flush=True)
            model.learn(total_timesteps=steps)
            model.save(str(path))
        else:
            model = SAC.load(str(path))
        models[kind] = model
    return models


def equal_resources(ctx, fractions=None):
    m = int(ctx["m"][0]); nodes = ctx["node"][:m].astype(int)
    p = np.zeros(m); f = np.zeros(m)
    for n in range(len(ctx["node_cpu"])):
        ii = np.flatnonzero(nodes == n)
        if len(ii):
            pf, ff = fractions[n] if fractions is not None else (1., 1.)
            p[ii] = ctx["node_power"][n] * pf / len(ii)
            f[ii] = ctx["node_cpu"][n] * ff / len(ii)
    return p, f


def convex_resources(ctx):
    """Actual P_fix energy minimum, with interference frozen at Equal's power.

    Exponential cones represent tau*exp(c/tau) exactly. Infeasibility is
    recorded; a budget-feasible Reference fallback is explicitly distinguished.
    """
    m = int(ctx["m"][0]); nodes = ctx["node"][:m].astype(int)
    p0, _ = equal_resources(ctx)
    gain = ctx["gain"][:m]; received = p0 * gain
    interf = np.array([ctx["alpha"][0] * (received[nodes == n].sum() - received[j])
                       for j, n in enumerate(nodes)])
    a = (ctx["noise"][0] + interf) / np.maximum(gain, 1e-30)
    counts = np.array([np.sum(nodes == n) for n in nodes])
    bw = ctx["bandwidth"][0] / counts
    c = np.log(2) * ctx["data"][:m] / bw / .1
    cycles = ctx["cycles"][:m] / 1e9
    u = cp.Variable(m, pos=True); v = cp.Variable(m, pos=True)
    txenergy = cp.Variable(m, nonneg=True)
    power = cp.multiply(a, cp.exp(cp.multiply(c, cp.inv_pos(u))) - 1)
    cpu = cp.multiply(cycles / .1, cp.inv_pos(v))
    compenergy = cp.multiply(2 * cycles ** 3, cp.square(cp.inv_pos(v)))
    cons = [u + v <= ctx["deadline"][:m] / .1,
            u <= ctx["data"][:m] / (bw * np.log2(1 + 10 ** (ctx["rel_db"][:m] / 10))) / .1]
    for j in range(m):
        cons.append(cp.ExpCone(c[j], u[j], txenergy[j] / (a[j] * .1) + u[j]))
    for n in np.unique(nodes):
        ii = np.flatnonzero(nodes == n)
        cons += [cp.sum(power[ii]) <= ctx["node_power"][n],
                 cp.sum(cpu[ii]) <= ctx["node_cpu"][n] / 1e9]
    problem = cp.Problem(cp.Minimize(cp.sum(txenergy + compenergy)), cons)
    try:
        problem.solve(solver="CLARABEL", max_iter=150)
        ok = problem.status == cp.OPTIMAL
        if ok:
            pv = np.asarray(power.value).ravel(); fv = np.asarray(cpu.value).ravel() * 1e9
            predicted_lat = .1 * (u.value + v.value)
            predicted_sinr = pv * gain / (ctx["noise"][0] + interf)
            ok = bool(np.all(predicted_lat <= ctx["deadline"][:m] + 1e-5)
                      and np.all(predicted_sinr >= 10 ** (ctx["rel_db"][:m] / 10) - 1e-5))
            for n in np.unique(nodes):
                ii = nodes == n
                ok &= pv[ii].sum() <= ctx["node_power"][n] * (1 + 1e-5)
                ok &= fv[ii].sum() <= ctx["node_cpu"][n] * (1 + 1e-5)
            if ok:
                return pv, fv, True
    except (cp.error.SolverError, ValueError, FloatingPointError):
        pass
    p, f = b.action_to_resources(np.zeros(2 * MAX_B), ctx, MAX_B)
    return p, f, False


def allocate(ctx, mode, models):
    t0 = time.perf_counter(); status = np.nan
    if mode in ["sac", "reward-add", "equal-match"]:
        model = models["additive" if mode == "reward-add" else "coverage"]
        action, _ = model.predict(observation(ctx), deterministic=True)
        p, f = b.action_to_resources(action, ctx, MAX_B)
        if mode == "equal-match":
            nodes = ctx["node"][:len(p)].astype(int); fractions = {}
            for n in np.unique(nodes):
                ii = nodes == n
                fractions[n] = (p[ii].sum() / ctx["node_power"][n],
                                f[ii].sum() / ctx["node_cpu"][n])
            p, f = equal_resources(ctx, fractions)
    elif mode == "equal":
        p, f = equal_resources(ctx)
    elif mode == "reference":
        p, f = b.action_to_resources(np.zeros(2 * MAX_B), ctx, MAX_B)
    elif mode == "fi-cvx":
        p, f, status = convex_resources(ctx)
    elif mode == "fi-1":
        m = int(ctx["m"][0]); nodes = ctx["node"][:m].astype(int)
        p = np.zeros(m); f = np.zeros(m)
        for n in np.unique(nodes):
            ii = np.flatnonzero(nodes == n)
            rp = ctx["data"][ii] / np.maximum(ctx["deadline"][ii] * ctx["gain"][ii], 1e-30)
            rf = ctx["cycles"][ii] / ctx["deadline"][ii]
            p[ii] = ctx["node_power"][n] * rp / rp.sum()
            f[ii] = ctx["node_cpu"][n] * rf / rf.sum()
    else:
        raise ValueError(mode)
    ms = (time.perf_counter() - t0) * 1000
    # Remove solver tolerance overflow without claiming SLA preservation.
    nodes = ctx["node"][:len(p)].astype(int)
    for n in np.unique(nodes):
        ii = nodes == n
        p[ii] *= min(1., ctx["node_power"][n] / max(p[ii].sum(), 1e-30))
        f[ii] *= min(1., ctx["node_cpu"][n] / max(f[ii].sum(), 1e-30))
    return b.allocation_outcome(ctx, p, f), ms, status, p, f


METHODS = {
    "SERVO": ("gnn", "submodular", "sac", "ph"),
    "MLP": ("mlp", "submodular", "sac", "ph"),
    "WUM": ("gnn", "additive", "sac", "ph"),
    "FI-1": ("gnn", "submodular", "fi-1", "ph"),
    "Equal": ("gnn", "submodular", "equal", "ph"),
    "Equal-Match": ("gnn", "submodular", "equal-match", "ph"),
    "Reference": ("gnn", "submodular", "reference", "ph"),
    "FI-CVX": ("gnn", "submodular", "fi-cvx", "ph"),
    "CostCov": ("gnn", "density", "reference", "ph"),
    "Reward-Add": ("gnn", "submodular", "reward-add", "ph"),
    "NoAdapt": ("gnn", "submodular", "sac", "none"),
    "EMA": ("gnn", "submodular", "sac", "ema"),
    "Periodic": ("gnn", "submodular", "sac", "periodic"),
}


def density_select(scene, q, imp, budget):
    """Cost-benefit coverage greedy adaptation, not a reproduction of Jia et al."""
    cpu, power = b._resource_demands(scene)
    h = np.clip(np.exp(-.55 * np.maximum(cpu + power - .35, 0)), .22, 1)
    chosen = []; usedc = np.zeros(len(scene.node_cpu)); usedp = usedc.copy()
    current = np.zeros(len(imp))
    while len(chosen) < budget:
        candidates = []
        for k in range(len(scene.flow_x)):
            n = scene.node_id[k]
            if k in chosen or not scene.node_active[n] or sum(scene.node_id[j] == n for j in chosen) >= 4:
                continue
            if usedc[n] + min(cpu[k], .85) > 1.12 or usedp[n] + min(power[k], .80) > 1.15:
                continue
            gain = sum(imp[w] * max(0, q[e] * h[k] - current[w]) for w, e in scene.flow_event_edges[k].items())
            candidates.append((gain / (.15 + cpu[k] + power[k]), -k, k))
        if not candidates or max(candidates)[0] <= 0:
            break
        k = max(candidates)[2]; chosen.append(k); n = scene.node_id[k]
        usedc[n] += min(cpu[k], .85); usedp[n] += min(power[k], .80)
        for w, e in scene.flow_event_edges[k].items():
            current[w] = max(current[w], q[e] * h[k])
    return chosen


def evaluate_snapshot(scene, chosen, q, imp, mode, models, infer_ms=0., select_ms=0.):
    pack_start = time.perf_counter()
    ctx = context(scene, chosen, q, imp, with_credit=mode in ["sac", "reward-add", "equal-match"])
    pack_ms = (time.perf_counter() - pack_start) * 1000
    outcome, alloc_ms, solver_ok, p, f = allocate(ctx, mode, models)
    alloc_ms += pack_ms
    overhead = infer_ms + select_ms + alloc_ms
    latency = np.asarray(outcome["latency"])
    complete_service = np.asarray(outcome["completed"], bool)
    complete = ((latency + overhead / 1000 <= ctx["deadline"][:len(chosen)]) &
                (outcome["sinr_db"] >= ctx["rel_db"][:len(chosen)]))
    raw = b.coverage_value(scene, chosen, scene.q_true, scene.imp_true, complete)
    den = b.coverage_value(scene, range(len(scene.flow_x)), scene.q_true, scene.imp_true)
    potential = b.coverage_value(scene, chosen, scene.q_true, scene.imp_true)
    component = b.coverage_value(scene, chosen, scene.q_true, scene.imp_true, complete_service)
    nc = int(complete.sum()); energy = float(np.sum(outcome["energy"]))
    nodes = ctx["node"][:len(chosen)].astype(int)
    max_power = max([p[nodes == n].sum() / scene.node_power[n] for n in np.unique(nodes)] or [0])
    max_cpu = max([f[nodes == n].sum() / scene.node_cpu[n] for n in np.unique(nodes)] or [0])
    assert max_power <= 1 + 1e-8 and max_cpu <= 1 + 1e-8
    assert 0 <= raw <= den + 1e-6
    return dict(value=raw / den if den > 0 else 0., raw_value=raw, reference_value=den,
                selected_value=potential / den if den > 0 else 0.,
                service_only_value=component / den if den > 0 else 0.,
                completion=nc / len(chosen) if chosen else 0.,
                sla_violation=1 - nc / len(chosen) if chosen else 0.,
                request_completion=nc / len(scene.flow_x), n_completed=nc,
                completed_mbit=float(scene.data_bits[np.asarray(chosen, int)][complete].sum() / 1e6),
                latency_p95_ms=float(np.percentile((latency[complete] * 1000 + overhead), 95)) if nc else np.nan,
                energy_total_j=energy, energy_per_completion=energy / nc if nc else np.nan,
                cpu_util=float(outcome["cpu_util"]), power_util=float(outcome["power_util"]),
                max_cpu_ratio=max_cpu, max_power_ratio=max_power,
                overhead_ms=overhead, infer_ms=infer_ms, select_ms=select_ms, alloc_ms=alloc_ms,
                solver_feasible=float(solver_ok),
                selected=json.dumps([int(k) for k in chosen]), n_selected=len(chosen), n_requests=len(scene.flow_x))


def trace(seed, method, cfg, spec, gnn, mlp, models):
    estimator, rule, mode, adaptation = METHODS[method]
    cal = b.OnlineCalibrator(adaptation); rng = np.random.default_rng(seed + 9)
    rows = []
    for slot in range(cfg.slots):
        scene = b.make_scene(seed, slot, cfg, spec)
        ts = time.perf_counter()
        qr, ir = b.predict_scene(gnn if estimator == "gnn" else mlp, scene)
        q, imp = cal.apply(scene, qr, ir)
        infer_ms = (time.perf_counter() - ts) * 1000
        ts = time.perf_counter()
        chosen = density_select(scene, q, imp, cfg.admission_budget) if rule == "density" else \
            b.select_flows(scene, q, imp, cfg.admission_budget, rule, rng)
        select_ms = (time.perf_counter() - ts) * 1000
        row = evaluate_snapshot(scene, chosen, q, imp, mode, models, infer_ms, select_ms)
        row.update(seed=seed, slot=slot, method=method, scenario=spec.name,
                   prediction_mae=.55 * np.abs(q - scene.q_true).mean() + .45 * np.abs(imp - scene.imp_true).mean(),
                   q_mae=float(np.abs(q - scene.q_true).mean()), importance_mae=float(np.abs(imp - scene.imp_true).mean()))
        adapt_start = time.perf_counter()
        cal.update(slot, scene, chosen, qr, ir)
        row["adapt_ms"] = (time.perf_counter() - adapt_start) * 1000
        row["control_ms"] = row["overhead_ms"]
        row["overhead_ms"] += row["adapt_ms"]
        rows.append(row)
    return pd.DataFrame(rows)


def ci(values):
    a = np.asarray(values, float); a = a[np.isfinite(a)]
    return float(student_t.ppf(.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.


def summarize(raw, keys=("scenario", "method")):
    excluded = {"seed", "slot"}; metrics = [c for c in raw.select_dtypes("number") if c not in set(keys) | excluded]
    per = raw.groupby(list(keys) + ["seed"], as_index=False)[metrics].mean()
    rows = []
    for name, group in per.groupby(list(keys)):
        if not isinstance(name, tuple): name = (name,)
        row = dict(zip(keys, name)); row["seeds"] = len(group)
        for c in metrics: row[c] = float(group[c].mean()); row[c + "_ci95"] = ci(group[c])
        # Ratio of aggregate energy to successful services, not mean slot ratios.
        source = raw
        for key, val in zip(keys, name): source = source[source[key] == val]
        ec = source.groupby("seed").apply(lambda g: g.energy_total_j.sum() / g.n_completed.sum()
                                         if g.n_completed.sum() else np.nan, include_groups=False)
        row["energy_per_completion"] = ec.mean(); row["energy_per_completion_ci95"] = ci(ec)
        rows.append(row)
    return pd.DataFrame(rows)


def paired(raw, left, right, post=None):
    if post is not None: raw = raw[raw.slot >= post]
    a = raw[raw.method == left]; d = raw[raw.method == right]
    metrics = ["value", "selected_value", "prediction_mae", "sla_violation", "raw_value"]
    rows = []
    for metric in metrics:
        diff = a.groupby("seed")[metric].mean() - d.groupby("seed")[metric].mean()
        rows.append(dict(left=left, right=right, metric=metric, delta=diff.mean(), ci95=ci(diff), seeds=len(diff)))
    joined = a.merge(d, on=["seed", "slot"], suffixes=("_a", "_b"))
    jac = []
    for r in joined.itertuples():
        aa, bb = set(json.loads(r.selected_a)), set(json.loads(r.selected_b))
        jac.append(1 - len(aa & bb) / max(len(aa | bb), 1))
    rows.append(dict(left=left, right=right, metric="selection_jaccard_distance", delta=np.mean(jac), ci95=np.nan, seeds=a.seed.nunique()))
    return pd.DataFrame(rows)


def thin_overlap(scene, retention, seed):
    """Controlled incidence thinning with sensing qualities/resources held fixed."""
    rng = np.random.default_rng(seed)
    keep = rng.random(len(scene.q_true)) < retention
    for w in range(len(scene.event_x)):
        ii = np.flatnonzero(scene.edge_index[1] == w)
        if len(ii):
            keep[ii[np.argmax(scene.q_true[ii])]] = True
    old_indices = np.flatnonzero(keep)
    incidence = scene.edge_index[:, keep]
    maps = [dict() for _ in scene.flow_event_edges]
    for e, (k, w) in enumerate(incidence.T): maps[int(k)][int(w)] = e
    return replace(scene, edge_index=incidence, edge_x=scene.edge_x[old_indices],
                   q_true=scene.q_true[old_indices], flow_event_edges=maps)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--train-steps", type=int, default=36000)
    ap.add_argument("--force-train", action="store_true"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(); OUT.mkdir(exist_ok=True); CKPT.mkdir(exist_ok=True)
    b.seed_everything(2026)
    gnn, mlp, metrics = b.train_estimators(b.SimConfig(), False, False)
    models = train_models(gnn, args.train_steps, args.force_train)
    cfg = b.SimConfig(slots=3 if args.smoke else 120)
    seeds = range(2 if args.smoke else 10)
    nominal = ["SERVO", "MLP", "WUM", "FI-1", "Equal", "Equal-Match", "Reference", "FI-CVX", "CostCov", "Reward-Add"]
    raw = []
    for method in nominal:
        print("Nominal", method, flush=True)
        for s in seeds: raw.append(trace(20000 + s, method, cfg, b.Scenario(name="nominal"), gnn, mlp, models))
        pd.concat(raw).to_csv(OUT / "raw_main.partial.csv", index=False)
    raw_main = pd.concat(raw, ignore_index=True); raw_main.to_csv(OUT / "raw_main.csv", index=False)
    summarize(raw_main).to_csv(OUT / "summary.csv", index=False)
    pair = [paired(raw_main, "SERVO", r) for r in ["MLP", "WUM", "Equal", "Equal-Match", "Reference", "Reward-Add", "CostCov"]]
    specs = [b.Scenario(name="nominal"), b.Scenario(name="burst", load_multiplier=1.35),
             b.Scenario(name="mobility", speed_multiplier=1.75), b.Scenario(name="heterogeneity", capacity_cv=.65),
             b.Scenario(name="failure", fail_node=True), b.Scenario(name="congestion", congestion_multiplier=1.8),
             b.Scenario(name="drift", concept_drift=True)]
    rseeds = range(2 if args.smoke else 6); rob = []
    for spec in specs:
        print("Stress", spec.name, flush=True)
        for method in ["SERVO", "NoAdapt", "CostCov"]:
            for s in rseeds: rob.append(trace(21000 + s, method, cfg, spec, gnn, mlp, models))
    robust = pd.concat(rob); robust.to_csv(OUT / "raw_robustness.csv", index=False)
    summarize(robust).to_csv(OUT / "robustness.csv", index=False)
    rec = []
    for method in ["SERVO", "NoAdapt", "EMA", "Periodic"]:
        print("Recovery", method, flush=True)
        for s in rseeds: rec.append(trace(22000 + s, method, cfg, b.Scenario(name="drift", concept_drift=True), gnn, mlp, models))
    rec = pd.concat(rec); rec.to_csv(OUT / "raw_recovery.csv", index=False)
    summarize(rec[rec.slot >= cfg.drift_slot]).to_csv(OUT / "recovery_post.csv", index=False)
    recovery_rows=[]
    if not args.smoke:
        for method, group in rec.groupby('method'):
            duration, success=b.recovery_time(group,cfg.drift_slot)
            recovery_rows.append(dict(method=method,recovery_slots=duration,recovery_success=success))
    pd.DataFrame(recovery_rows).to_csv(OUT/'recovery.csv',index=False)
    for other in ["NoAdapt", "EMA", "Periodic"]: pair.append(paired(rec, "SERVO", other, cfg.drift_slot))
    pd.concat(pair).to_csv(OUT / "paired.csv", index=False)
    scale = []
    for nv, ne, nn, budget in [(36,20,4,4),(72,36,4,8),(128,64,8,12),(224,96,12,16)]:
        print("Scale", nv, ne, nn, budget, flush=True)
        scfg = replace(cfg, n_vehicles=nv, n_events=ne, n_nodes=nn, admission_budget=budget, slots=3 if args.smoke else 35)
        for method in ["SERVO", "WUM", "CostCov"]:
            for s in range(2 if args.smoke else 5):
                dd = trace(23000+s, method, scfg, b.Scenario(name="scale"), gnn, mlp, models)
                dd["vehicles"]=nv; dd["events"]=ne; dd["nodes"]=nn; dd["budget"]=budget
                scale.append(dd)
    scale = pd.concat(scale); scale.to_csv(OUT / "raw_scale.csv", index=False)
    summarize(scale, ("vehicles","events","nodes","budget","method")).to_csv(OUT / "scalability.csv", index=False)
    curves=[]
    for seed in range(2 if args.smoke else 5):
        for slot in range(3 if args.smoke else 30):
            sc=b.make_scene(24000+seed,slot,cfg,b.Scenario());q,imp=b.predict_scene(gnn,sc)
            chosen=b.select_flows(sc,q,imp,8,"submodular",np.random.default_rng(seed))
            for frac in [.7,.8,.9,1.]:
                adjusted=replace(sc,node_cpu=sc.node_cpu*frac,node_power=sc.node_power*frac)
                for mode in ["sac","reference","equal"]:
                    row=evaluate_snapshot(adjusted,chosen,q,imp,mode,models)
                    row.update(seed=seed,slot=slot,method=mode,resource_fraction=frac);curves.append(row)
    curves=pd.DataFrame(curves);curves.to_csv(OUT/'raw_resource_curve.csv',index=False)
    summarize(curves,("resource_fraction","method")).to_csv(OUT/'resource_curve.csv',index=False)
    # Small exact cardinality coverage upper bound, independent of any allocator.
    oracle=[]
    for seed in range(2 if args.smoke else 10):
        for slot in range(2 if args.smoke else 10):
            small=replace(cfg,n_vehicles=12,n_events=8,n_nodes=3,admission_budget=4)
            sc=b.make_scene(26000+seed,slot,small,b.Scenario());q,imp=b.predict_scene(gnn,sc)
            pool=range(len(sc.flow_x));best=0.
            for size in range(1,min(4,len(sc.flow_x))+1):
                for subset in itertools.combinations(pool,size):
                    if max(np.bincount(sc.node_id[list(subset)],minlength=3))<=4:
                        best=max(best,b.coverage_value(sc,subset,sc.q_true,sc.imp_true))
            den=b.coverage_value(sc,pool,sc.q_true,sc.imp_true)
            for method in ["SERVO","WUM","CostCov"]:
                rule=METHODS[method][1]
                chosen=density_select(sc,q,imp,4) if rule=="density" else b.select_flows(sc,q,imp,4,rule,np.random.default_rng(seed))
                row=evaluate_snapshot(sc,chosen,q,imp,"reference",models)
                row.update(seed=seed,slot=slot,method=method,oracle_value=best/den if den else 0.,oracle_ratio=row['raw_value']/best if best else 0.)
                oracle.append(row)
    oracle=pd.DataFrame(oracle);oracle.to_csv(OUT/'raw_oracle.csv',index=False)
    summarize(oracle,("method",)).to_csv(OUT/'oracle.csv',index=False)
    overlap=[]
    for seed in range(2 if args.smoke else 5):
        for slot in range(2 if args.smoke else 30):
            original=b.make_scene(25000+seed,slot,cfg,b.Scenario())
            for retention in [0.,.25,.5,1.]:
                sc=thin_overlap(original,retention,25000+seed*100+slot)
                counts=np.bincount(sc.edge_index[1],minlength=len(sc.event_x))
                for method,rule in [('Coverage','submodular'),('Additive','additive')]:
                    chosen=b.select_flows(sc,sc.q_true,sc.imp_true,8,rule,np.random.default_rng(seed))
                    row=evaluate_snapshot(sc,chosen,sc.q_true,sc.imp_true,'equal',models)
                    row.update(seed=seed,slot=slot,method=method,retention=retention,
                               mean_redundancy=counts.mean(),max_redundancy=counts.max())
                    overlap.append(row)
    overlap=pd.DataFrame(overlap);overlap.to_csv(OUT/'raw_overlap.csv',index=False)
    summarize(overlap,('retention','method')).to_csv(OUT/'overlap.csv',index=False)
    manifest=dict(timestamp_utc=pd.Timestamp.utcnow().isoformat(),config=asdict(cfg),max_policy_budget=MAX_B,
                  training_seed=2026,train_steps=args.train_steps,smoke=args.smoke,gnn_metrics=metrics,
                  correction="SINR denominator uses noise/interference in W without dimensionless EPS",
                  timing="serial pre-execution control (including input packing) + service; full overhead additionally includes post-feedback calibration; independent snapshots, no queues/carry-over",
                  seeds=dict(nominal=[20000+s for s in seeds],stress=[21000+s for s in rseeds],recovery=[22000+s for s in rseeds]),
                  software=dict(numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,cvxpy=cp.__version__),
                  ci='two-sided Student-t over paired or independent trace-seed means; fixed training seed',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Completed', OUT, flush=True)


if __name__ == "__main__":
    main()

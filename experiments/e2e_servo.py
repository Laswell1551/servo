#!/usr/bin/env python3
"""End-to-end SERVO evaluation for the TSC major revision.

This script is deliberately self-contained and evidence-producing.  It trains
two coverage estimators (a bipartite message-passing GNN and an edge-wise MLP),
trains a Soft Actor-Critic (SAC) within-set resource allocator, and evaluates the
integrated orchestration loop in a dynamic urban V2X system-level simulator.

The simulator includes mobility, distance-dependent fading, co-channel
interference, heterogeneous edge compute/power budgets, dynamic requests,
service deadlines, node failures, congestion, and concept drift.  It is not an
NS-3/SUMO packet-level emulation; the manuscript states that boundary explicitly.

Outputs are written to experiments/results and publication figures to manuscript/.
All tables and figures are computed from recorded per-seed results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "experiments" / "results"
CHECKPOINT_DIR = ROOT / "experiments" / "checkpoints"
FIG_DIR = ROOT / "manuscript"
for _p in (RESULT_DIR, CHECKPOINT_DIR, FIG_DIR):
    _p.mkdir(parents=True, exist_ok=True)


SEED_LIST = list(range(10))
EPS = 1e-9


@dataclass
class SimConfig:
    slots: int = 120
    slot_s: float = 0.10
    n_vehicles: int = 72
    n_events: int = 36
    n_nodes: int = 4
    admission_budget: int = 8
    road_length_m: float = 1000.0
    sensing_range_m: float = 210.0
    carrier_ghz: float = 3.5
    bandwidth_hz: float = 100e6
    noise_w: float = 1.0e-12
    interference_alpha: float = 0.035
    request_probability: float = 0.72
    drift_slot: int = 60
    failure_slot: int = 60


@dataclass
class Scenario:
    name: str = "nominal"
    load_multiplier: float = 1.0
    speed_multiplier: float = 1.0
    congestion_multiplier: float = 1.0
    capacity_cv: float = 0.30
    fail_node: bool = False
    concept_drift: bool = False


@dataclass
class Scene:
    flow_x: np.ndarray
    event_x: np.ndarray
    edge_index: np.ndarray
    edge_x: np.ndarray
    q_true: np.ndarray
    imp_true: np.ndarray
    tiers: np.ndarray
    node_id: np.ndarray
    node_xy: np.ndarray
    node_cpu: np.ndarray
    node_power: np.ndarray
    node_active: np.ndarray
    channel_gain: np.ndarray
    data_bits: np.ndarray
    cycles: np.ndarray
    deadline_s: np.ndarray
    reliability_db: np.ndarray
    bandwidth_hz: float
    noise_w: float
    interference_alpha: float
    flow_event_edges: List[Dict[int, int]]
    t: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _node_positions(n_nodes: int, length: float) -> np.ndarray:
    if n_nodes <= 4:
        base = np.array([[0.28, 0.28], [0.72, 0.28], [0.28, 0.72], [0.72, 0.72]])
        return base[:n_nodes] * length
    side = int(math.ceil(math.sqrt(n_nodes)))
    coords = np.linspace(0.14, 0.86, side)
    return np.array([(x, y) for y in coords for x in coords], dtype=float)[:n_nodes] * length


def _vehicle_kinematics(seed: int, t: int, n: int, cfg: SimConfig, speed_mult: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed * 7919 + 17)
    direction = rng.integers(0, 4, size=n)
    initial = rng.uniform(0, cfg.road_length_m, size=n)
    speed = rng.uniform(8.0, 18.0, size=n) * speed_mult
    progress = (initial + speed * cfg.slot_s * t) % cfg.road_length_m
    lane_offset = np.where(np.arange(n) % 2 == 0, -6.0, 6.0)
    xy = np.zeros((n, 2), dtype=float)
    vel = np.zeros((n, 2), dtype=float)
    for i, d in enumerate(direction):
        if d == 0:  # eastbound
            xy[i] = (progress[i], cfg.road_length_m / 2 + lane_offset[i])
            vel[i] = (speed[i], 0)
        elif d == 1:  # westbound
            xy[i] = (cfg.road_length_m - progress[i], cfg.road_length_m / 2 - lane_offset[i])
            vel[i] = (-speed[i], 0)
        elif d == 2:  # northbound
            xy[i] = (cfg.road_length_m / 2 - lane_offset[i], progress[i])
            vel[i] = (0, speed[i])
        else:  # southbound
            xy[i] = (cfg.road_length_m / 2 + lane_offset[i], cfg.road_length_m - progress[i])
            vel[i] = (0, -speed[i])
    return xy, vel, speed


def _event_state(seed: int, t: int, m: int, cfg: SimConfig, drift: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed * 3571 + 101)
    along = rng.uniform(45, cfg.road_length_m - 45, size=m)
    orient = rng.integers(0, 2, size=m)
    offset = rng.normal(0, 22, size=m)
    xy = np.column_stack(
        [np.where(orient == 0, along, cfg.road_length_m / 2 + offset),
         np.where(orient == 0, cfg.road_length_m / 2 + offset, along)]
    )
    # Small deterministic motion makes the trace dynamic without regenerating identities.
    phase = rng.uniform(0, 2 * np.pi, size=m)
    xy[:, 0] += 8.0 * np.sin(phase + t / 17.0)
    xy[:, 1] += 8.0 * np.cos(phase + t / 19.0)
    tiers = rng.choice(3, size=m, p=[0.22, 0.33, 0.45])
    base = np.array([0.95, 0.60, 0.28])
    imp = base[tiers] * rng.uniform(0.92, 1.08, size=m)
    # Concept drift changes which semantic classes are urgent and shifts the hotspot.
    if drift:
        drift_factor = np.array([0.52, 1.72, 1.42])
        imp = np.clip(imp * drift_factor[tiers], 0.15, 1.0)
        east_hotspot = 0.22 * (xy[:, 0] > 0.56 * cfg.road_length_m)
        imp = np.clip(imp + east_hotspot, 0.15, 1.0)
    return xy, tiers, imp


def _path_gain(distance_m: np.ndarray, fc_ghz: float, shadow_db: np.ndarray) -> np.ndarray:
    # 3GPP-inspired urban-microcell large-scale loss with bounded near-field distance.
    d = np.maximum(distance_m, 10.0)
    pl_db = 32.4 + 21.0 * np.log10(d) + 20.0 * np.log10(fc_ghz) + shadow_db
    return 10.0 ** (-pl_db / 10.0)


def make_scene(seed: int, t: int, cfg: SimConfig, scenario: Scenario) -> Scene:
    rng = np.random.default_rng(seed * 100003 + t * 97 + 13)
    drift_active = scenario.concept_drift and t >= cfg.drift_slot
    veh_xy, veh_vel, speed = _vehicle_kinematics(seed, t, cfg.n_vehicles, cfg, scenario.speed_multiplier)
    ev_xy, tiers, imp = _event_state(seed, t, cfg.n_events, cfg, drift_active)
    node_xy = _node_positions(cfg.n_nodes, cfg.road_length_m)

    # Heterogeneous edge resources; CV is controlled without changing the mean budget.
    base_cpu = 36e9
    base_power = 8.0
    z = np.linspace(-1.0, 1.0, cfg.n_nodes)
    z = z / max(np.std(z), EPS)
    node_cpu = base_cpu * np.clip(1.0 + scenario.capacity_cv * z, 0.35, 1.8)
    node_power = base_power * np.clip(1.0 + 0.7 * scenario.capacity_cv * z[::-1], 0.4, 1.7)
    node_active = np.ones(cfg.n_nodes, dtype=bool)
    if scenario.fail_node and t >= cfg.failure_slot:
        node_active[0] = False
        node_cpu[0] = 0.0
        node_power[0] = 0.0

    p_req = min(0.98, cfg.request_probability * scenario.load_multiplier)
    request_mask = rng.random(cfg.n_vehicles) < p_req
    requested = np.flatnonzero(request_mask)
    minimum = min(cfg.n_vehicles, max(cfg.admission_budget + 4, int(0.45 * cfg.n_vehicles)))
    if len(requested) < minimum:
        requested = np.argsort(rng.random(cfg.n_vehicles))[:minimum]
    pos = veh_xy[requested]
    vel = veh_vel[requested]
    n_flow = len(requested)

    d_fn = np.linalg.norm(pos[:, None, :] - node_xy[None, :, :], axis=2)
    shadow = rng.normal(0.0, 3.0, size=d_fn.shape)
    gains = _path_gain(d_fn, cfg.carrier_ghz, shadow)
    gains[:, ~node_active] = 0.0
    node_id = np.argmax(gains, axis=1)
    channel_gain = gains[np.arange(n_flow), node_id]

    # Compressed cooperative-perception feature packets and edge inference demand.
    data_bits = rng.uniform(0.5e6, 2.8e6, size=n_flow)
    cycles = rng.uniform(0.06e9, 0.45e9, size=n_flow)
    deadline = np.where(rng.random(n_flow) < 0.35,
                        rng.uniform(0.075, 0.120, size=n_flow),
                        rng.uniform(0.115, 0.200, size=n_flow))
    reliability_db = np.where(deadline < 0.09, 2.0, 0.0)

    # Node/flow features are normalized and intentionally exclude the hidden drift factor.
    flow_x = np.column_stack([
        pos[:, 0] / cfg.road_length_m,
        pos[:, 1] / cfg.road_length_m,
        vel[:, 0] / 25.0,
        vel[:, 1] / 25.0,
        np.log10(np.maximum(channel_gain, 1e-15)) / 15.0 + 1.0,
        data_bits / 3.0e6,
        cycles / 0.5e9,
        deadline / 0.20,
    ]).astype(np.float32)

    # Event features expose semantic class and location, not the post-drift importance.
    tier_oh = np.eye(3, dtype=np.float32)[tiers]
    event_x = np.column_stack([
        ev_xy[:, 0] / cfg.road_length_m,
        ev_xy[:, 1] / cfg.road_length_m,
        tier_oh,
    ]).astype(np.float32)

    src: List[int] = []
    dst: List[int] = []
    edge_features: List[List[float]] = []
    q_values: List[float] = []
    flow_event_edges: List[Dict[int, int]] = [dict() for _ in range(n_flow)]
    all_dist = np.linalg.norm(pos[:, None, :] - ev_xy[None, :, :], axis=2)
    event_density = np.sum(all_dist <= cfg.sensing_range_m, axis=0)
    for k in range(n_flow):
        rel = ev_xy - pos[k]
        dist = np.linalg.norm(rel, axis=1)
        heading = vel[k] / max(np.linalg.norm(vel[k]), EPS)
        cosang = (rel @ heading) / np.maximum(dist, 1.0)
        candidates = np.flatnonzero((dist <= cfg.sensing_range_m) & (cosang > -0.45))
        if len(candidates) == 0:
            candidates = np.array([int(np.argmin(dist))])
        for w in candidates:
            map_los = 1.0 if (abs(pos[k, 0] - cfg.road_length_m / 2) < 110 or
                              abs(pos[k, 1] - cfg.road_length_m / 2) < 110) else 0.78
            orientation = np.clip((cosang[w] + 1.0) / 2.0, 0.15, 1.0)
            geom = math.exp(-0.5 * (dist[w] / 145.0) ** 2)
            fidelity = 0.78 + 0.18 * math.sin(0.37 * requested[k] + 0.13 * seed)
            occlusion = map_los
            if drift_active:
                # A changed construction/traffic pattern creates systematic, spatial bias.
                if ev_xy[w, 0] > 0.55 * cfg.road_length_m:
                    occlusion *= 0.48
                if tiers[w] == 1:
                    occlusion *= 0.62
            # Dense viewpoints create correlated occlusion; only the graph model sees
            # enough neighboring flows to infer this context reliably.
            crowding = math.exp(-0.055 * max(0, int(event_density[w]) - 4))
            q = np.clip(geom * (0.55 + 0.45 * orientation) * fidelity * occlusion * crowding +
                        rng.normal(0, 0.025), 0.02, 1.0)
            eidx = len(src)
            src.append(k)
            dst.append(int(w))
            edge_features.append([
                rel[w, 0] / cfg.sensing_range_m,
                rel[w, 1] / cfg.sensing_range_m,
                dist[w] / cfg.sensing_range_m,
                cosang[w],
                map_los,
            ])
            q_values.append(float(q))
            flow_event_edges[k][int(w)] = eidx

    edge_index = np.array([src, dst], dtype=np.int64)
    return Scene(
        flow_x=flow_x,
        event_x=event_x,
        edge_index=edge_index,
        edge_x=np.asarray(edge_features, dtype=np.float32),
        q_true=np.asarray(q_values, dtype=np.float32),
        imp_true=imp.astype(np.float32),
        tiers=tiers.astype(np.int64),
        node_id=node_id.astype(np.int64),
        node_xy=node_xy,
        node_cpu=node_cpu,
        node_power=node_power,
        node_active=node_active,
        channel_gain=channel_gain,
        data_bits=data_bits,
        cycles=cycles,
        deadline_s=deadline,
        reliability_db=reliability_db,
        bandwidth_hz=cfg.bandwidth_hz / scenario.congestion_multiplier,
        noise_w=cfg.noise_w,
        interference_alpha=cfg.interference_alpha * scenario.congestion_multiplier,
        flow_event_edges=flow_event_edges,
        t=t,
    )


class BipartiteCoverageGNN(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.flow_enc = nn.Sequential(nn.Linear(8, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.event_enc = nn.Sequential(nn.Linear(5, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.edge_enc = nn.Sequential(nn.Linear(5, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.f_to_e = nn.Linear(hidden, hidden, bias=False)
        self.e_to_f = nn.Linear(hidden, hidden, bias=False)
        self.flow_up = nn.Linear(2 * hidden, hidden)
        self.event_up = nn.Linear(2 * hidden, hidden)
        self.q_head = nn.Sequential(nn.Linear(2 * hidden + 5, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.imp_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, flow_x: torch.Tensor, event_x: torch.Tensor,
                edge_index: torch.Tensor, edge_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hf = F.relu(self.flow_enc(flow_x))
        he = F.relu(self.event_enc(event_x))
        src, dst = edge_index
        ee = F.relu(self.edge_enc(edge_x))
        agg_e = torch.zeros_like(he)
        agg_e.index_add_(0, dst, self.f_to_e(hf[src]) + ee)
        deg_e = torch.bincount(dst, minlength=he.shape[0]).clamp(min=1).float().unsqueeze(1)
        he = F.relu(self.event_up(torch.cat([he, agg_e / torch.sqrt(deg_e)], dim=1)))
        agg_f = torch.zeros_like(hf)
        agg_f.index_add_(0, src, self.e_to_f(he[dst]) + ee)
        deg_f = torch.bincount(src, minlength=hf.shape[0]).clamp(min=1).float().unsqueeze(1)
        hf = F.relu(self.flow_up(torch.cat([hf, agg_f / torch.sqrt(deg_f)], dim=1)))
        q = torch.sigmoid(self.q_head(torch.cat([hf[src], he[dst], edge_x], dim=1))).squeeze(1)
        imp = torch.sigmoid(self.imp_head(he)).squeeze(1)
        return q, imp


class EdgeMLP(nn.Module):
    def __init__(self, hidden: int = 48):
        super().__init__()
        self.q_net = nn.Sequential(nn.Linear(8 + 5 + 5, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.imp_net = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, flow_x: torch.Tensor, event_x: torch.Tensor,
                edge_index: torch.Tensor, edge_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        q = torch.sigmoid(self.q_net(torch.cat([flow_x[src], event_x[dst], edge_x], dim=1))).squeeze(1)
        imp = torch.sigmoid(self.imp_net(event_x)).squeeze(1)
        return q, imp


def _scene_tensors(scene: Scene) -> Tuple[torch.Tensor, ...]:
    return (torch.from_numpy(scene.flow_x), torch.from_numpy(scene.event_x),
            torch.from_numpy(scene.edge_index), torch.from_numpy(scene.edge_x),
            torch.from_numpy(scene.q_true), torch.from_numpy(scene.imp_true))


def train_estimators(cfg: SimConfig, quick: bool, force: bool) -> Tuple[nn.Module, nn.Module, Dict[str, float]]:
    g_path = CHECKPOINT_DIR / "coverage_gnn.pt"
    m_path = CHECKPOINT_DIR / "coverage_mlp.pt"
    meta_path = CHECKPOINT_DIR / "coverage_metrics.json"
    gnn = BipartiteCoverageGNN()
    mlp = EdgeMLP()
    if not force and g_path.exists() and m_path.exists() and meta_path.exists():
        gnn.load_state_dict(torch.load(g_path, map_location="cpu", weights_only=True))
        mlp.load_state_dict(torch.load(m_path, map_location="cpu", weights_only=True))
        return gnn.eval(), mlp.eval(), json.loads(meta_path.read_text(encoding="utf-8"))

    train_graphs = 18 if quick else 64
    val_graphs = 8 if quick else 20
    epochs = 7 if quick else 28
    train = [make_scene(300 + i, i % 50, cfg, Scenario()) for i in range(train_graphs)]
    val = [make_scene(800 + i, 10 + i, cfg, Scenario()) for i in range(val_graphs)]

    def fit(model: nn.Module) -> nn.Module:
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
        model.train()
        for _ in range(epochs):
            random.shuffle(train)
            for sc in train:
                fx, ex, ei, ed, qy, iy = _scene_tensors(sc)
                qp, ip = model(fx, ex, ei, ed)
                loss = F.mse_loss(qp, qy) + 0.65 * F.mse_loss(ip, iy)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                opt.step()
        return model.eval()

    gnn = fit(gnn)
    mlp = fit(mlp)

    def evaluate(model: nn.Module, graphs: Sequence[Scene]) -> Tuple[float, float]:
        qerr, ierr = [], []
        with torch.no_grad():
            for sc in graphs:
                fx, ex, ei, ed, qy, iy = _scene_tensors(sc)
                qp, ip = model(fx, ex, ei, ed)
                qerr.extend(torch.abs(qp - qy).tolist())
                ierr.extend(torch.abs(ip - iy).tolist())
        return float(np.mean(qerr)), float(np.mean(ierr))

    g_q, g_i = evaluate(gnn, val)
    m_q, m_i = evaluate(mlp, val)
    # Unseen post-drift graphs quantify systematic shift before online adaptation.
    drift_val = [make_scene(900 + i, cfg.drift_slot + 5 + i, cfg,
                            Scenario(concept_drift=True)) for i in range(val_graphs)]
    gd_q, gd_i = evaluate(gnn, drift_val)
    metrics = {
        "gnn_q_mae": g_q, "gnn_importance_mae": g_i,
        "mlp_q_mae": m_q, "mlp_importance_mae": m_i,
        "gnn_post_drift_q_mae_before_adaptation": gd_q,
        "gnn_post_drift_importance_mae_before_adaptation": gd_i,
        "train_graphs": train_graphs, "validation_graphs": val_graphs, "epochs": epochs,
    }
    torch.save(gnn.state_dict(), g_path)
    torch.save(mlp.state_dict(), m_path)
    meta_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return gnn, mlp, metrics


def predict_scene(model: nn.Module, scene: Scene) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        fx, ex, ei, ed, _, _ = _scene_tensors(scene)
        q, imp = model(fx, ex, ei, ed)
    return q.numpy().astype(float), imp.numpy().astype(float)


class OnlineCalibrator:
    """Feedback-only distance-bin calibration plus Page-Hinkley change detection."""

    def __init__(self, mode: str):
        self.mode = mode
        self.q_scale = np.ones(3)
        self.i_scale = np.ones(3)
        self.buffer: deque = deque(maxlen=1200)
        self.mean_err = 0.0
        self.n = 0
        self.ph = 0.0
        self.ph_min = 0.0
        self.cooldown = 0
        self.detections: List[int] = []

    @staticmethod
    def bins(scene: Scene) -> np.ndarray:
        # The major drift is semantic-class dependent (changed sensing conditions
        # and service intent), so the last-layer calibration is per event tier.
        return scene.tiers[scene.edge_index[1]].astype(int)

    def apply(self, scene: Scene, q: np.ndarray, imp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        qb = self.bins(scene)
        return np.clip(q * self.q_scale[qb], 0, 1), np.clip(imp * self.i_scale[scene.tiers], 0, 1)

    def update(self, t: int, scene: Scene, selected: Sequence[int], q_pred: np.ndarray,
               imp_pred: np.ndarray) -> None:
        if self.mode == "none" or not selected:
            return
        selected_edges = [e for k in selected for e in scene.flow_event_edges[k].values()]
        if not selected_edges:
            return
        q_idx = np.asarray(selected_edges, dtype=int)
        selected_events = sorted({w for k in selected for w in scene.flow_event_edges[k]})
        # Detect residual error after the current calibration. Ratios below still
        # use raw predictions, so refitting estimates an absolute scale.
        q_cal, i_cal = self.apply(scene, q_pred, imp_pred)
        q_err = float(np.mean(np.abs(q_cal[q_idx] - scene.q_true[q_idx])))
        i_err = float(np.mean(np.abs(i_cal[selected_events] - scene.imp_true[selected_events]))) \
            if selected_events else 0.0
        err = 0.55 * q_err + 0.45 * i_err
        bins = self.bins(scene)
        for e in q_idx:
            self.buffer.append(("q", int(bins[e]), float(scene.q_true[e] / max(q_pred[e], 0.05))))
        for w in selected_events:
            self.buffer.append(("i", int(scene.tiers[w]), float(scene.imp_true[w] / max(imp_pred[w], 0.08))))

        self.n += 1
        self.mean_err += (err - self.mean_err) / self.n
        # Page-Hinkley detects a sustained increase in prediction error.
        self.ph += err - self.mean_err - 0.002
        self.ph_min = min(self.ph_min, self.ph)
        score = self.ph - self.ph_min
        trigger = self.mode == "ph" and self.n > 20 and self.cooldown == 0 and score > 0.10
        if trigger:
            self._refit(recent=100, blend=1.0)
            self.detections.append(t)
            self.ph = self.ph_min = 0.0
            self.cooldown = 20
        elif self.mode == "ema":
            self._refit(recent=120, blend=0.06)
        elif self.mode == "periodic" and t > 0 and t % 25 == 0:
            self._refit(recent=300, blend=0.65)
        elif self.mode == "ph":
            self._refit(recent=80, blend=0.01)
        self.cooldown = max(0, self.cooldown - 1)

    def _refit(self, recent: int, blend: float) -> None:
        rows = list(self.buffer)[-recent:]
        for kind, arr in (("q", self.q_scale), ("i", self.i_scale)):
            for b in range(3):
                vals = [r for k, bb, r in rows if k == kind and bb == b]
                if vals:
                    target = float(np.clip(np.median(vals), 0.55, 1.75))
                    arr[b] = (1 - blend) * arr[b] + blend * target


def coverage_value(scene: Scene, selected: Sequence[int], q_edge: np.ndarray,
                   imp: np.ndarray, completed: Optional[np.ndarray] = None) -> float:
    best: Dict[int, float] = {}
    for j, k in enumerate(selected):
        if completed is not None and not bool(completed[j]):
            continue
        for w, e in scene.flow_event_edges[k].items():
            best[w] = max(best.get(w, 0.0), float(q_edge[e]))
    return float(sum(float(imp[w]) * q for w, q in best.items()))


def _resource_demands(scene: Scene) -> Tuple[np.ndarray, np.ndarray]:
    rate_at_half = (scene.bandwidth_hz / 4.0) * np.log2(
        1.0 + 0.45 * scene.node_power[scene.node_id] * scene.channel_gain / scene.noise_w
    )
    tx = scene.data_bits / np.maximum(rate_at_half, 1e5)
    residual = np.maximum(scene.deadline_s - tx, 0.012)
    cpu_frac = scene.cycles / np.maximum(residual * scene.node_cpu[scene.node_id], 1.0)
    power_frac = np.clip(scene.data_bits / np.maximum(rate_at_half * scene.deadline_s, 1.0), 0.03, 1.2)
    return cpu_frac, power_frac


def select_flows(scene: Scene, q_pred: np.ndarray, imp_pred: np.ndarray, budget: int,
                 rule: str, rng: np.random.Generator) -> List[int]:
    n = len(scene.flow_x)
    cpu_dem, p_dem = _resource_demands(scene)
    used_cpu = np.zeros(len(scene.node_cpu))
    used_power = np.zeros(len(scene.node_power))
    chosen: List[int] = []
    current: Dict[int, float] = {}
    # Eq. (quality) is resource dependent: sensing quality is discounted when the
    # predicted radio/compute demand approaches the node budget.
    realizability = np.clip(np.exp(-0.55 * np.maximum(cpu_dem + p_dem - 0.35, 0.0)), 0.22, 1.0)

    def feasible(k: int) -> bool:
        node = scene.node_id[k]
        if not scene.node_active[node]:
            return False
        # Conservative feasibility projection; it is not used in theorem claims.
        return used_cpu[node] + min(cpu_dem[k], 0.85) <= 1.12 and \
            used_power[node] + min(p_dem[k], 0.80) <= 1.15

    def marginal(k: int) -> float:
        return sum(float(imp_pred[w]) * max(0.0, float(q_pred[e]) * realizability[k] - current.get(w, 0.0))
                   for w, e in scene.flow_event_edges[k].items())

    additive = np.array([
        sum(float(imp_pred[w]) * float(q_pred[e]) * realizability[k]
            for w, e in scene.flow_event_edges[k].items())
        for k in range(n)
    ])

    while len(chosen) < min(budget, n):
        candidates = [k for k in range(n) if k not in chosen and feasible(k)]
        if not candidates:
            break
        if rule == "submodular":
            scores = [marginal(k) for k in candidates]
        elif rule == "additive":
            scores = [additive[k] for k in candidates]
        elif rule == "load_aware":
            scores = [additive[k] / (0.15 + cpu_dem[k] + p_dem[k]) for k in candidates]
        elif rule == "random":
            scores = rng.random(len(candidates)).tolist()
        else:
            raise ValueError(rule)
        best = candidates[int(np.argmax(scores))]
        if max(scores) <= 0:
            break
        chosen.append(best)
        node = scene.node_id[best]
        used_cpu[node] += min(cpu_dem[best], 0.85)
        used_power[node] += min(p_dem[best], 0.80)
        for w, e in scene.flow_event_edges[best].items():
            current[w] = max(current.get(w, 0.0), float(q_pred[e]) * realizability[best])
    return chosen


def _allocation_arrays(scene: Scene, selected: Sequence[int], max_b: int,
                       q_hint: Optional[np.ndarray] = None,
                       imp_hint: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    m = len(selected)
    idx = np.asarray(selected, dtype=int)
    pad = max_b - m
    def padded(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
        a = np.asarray(x, dtype=float)
        return np.pad(a, (0, pad), constant_values=fill)
    node = np.pad(scene.node_id[idx], (0, pad), constant_values=-1)
    q_source = scene.q_true if q_hint is None else np.asarray(q_hint)
    i_source = scene.imp_true if imp_hint is None else np.asarray(imp_hint)
    return {
        "m": np.array([m]),
        "gain": padded(scene.channel_gain[idx]),
        "data": padded(scene.data_bits[idx]),
        "cycles": padded(scene.cycles[idx]),
        "deadline": padded(scene.deadline_s[idx], 0.15),
        "rel_db": padded(scene.reliability_db[idx]),
        "importance": padded(np.array([max((i_source[w] for w in scene.flow_event_edges[k]), default=0.2)
                                        for k in selected])),
        "q": padded(np.array([max((q_source[e] for e in scene.flow_event_edges[k].values()), default=0.1)
                               for k in selected])),
        "node": node,
        "node_cpu": scene.node_cpu.copy(),
        "node_power": scene.node_power.copy(),
        "bandwidth": np.array([scene.bandwidth_hz]),
        "noise": np.array([scene.noise_w]),
        "alpha": np.array([scene.interference_alpha]),
    }


def allocation_observation(ctx: Mapping[str, np.ndarray], max_b: int) -> np.ndarray:
    m = int(ctx["m"][0])
    node = ctx["node"].astype(int)
    counts = np.zeros(max_b)
    cpu_cap = np.zeros(max_b)
    p_cap = np.zeros(max_b)
    for i in range(m):
        n = node[i]
        counts[i] = max(1, np.sum(node[:m] == n)) / max_b
        cpu_cap[i] = ctx["node_cpu"][n] / 60e9
        p_cap[i] = ctx["node_power"][n] / 12.0
    feats = np.column_stack([
        np.clip((np.log10(np.maximum(ctx["gain"], 1e-15)) + 13.5) / 5.0, -1, 1),
        ctx["data"] / 3.0e6,
        ctx["cycles"] / 0.5e9,
        ctx["deadline"] / 0.20,
        ctx["rel_db"] / 5.0,
        ctx["importance"],
        ctx["q"],
        cpu_cap,
        p_cap,
        counts,
    ])
    mask = np.zeros(max_b); mask[:m] = 1.0
    global_f = np.array([ctx["bandwidth"][0] / 100e6,
                         ctx["alpha"][0] / 0.25,
                         m / max_b], dtype=float)
    return np.concatenate([feats.ravel(), mask, global_f]).astype(np.float32)


def action_to_resources(action: np.ndarray, ctx: Mapping[str, np.ndarray], max_b: int) -> Tuple[np.ndarray, np.ndarray]:
    m = int(ctx["m"][0])
    a = np.asarray(action, dtype=float).reshape(2, max_b)
    raw_p = np.clip(a[0], -1.0, 1.0)
    raw_f = np.clip(a[1], -1.0, 1.0)
    p = np.zeros(max_b); f = np.zeros(max_b)
    node = ctx["node"].astype(int)
    for n in range(len(ctx["node_cpu"])):
        ii = np.flatnonzero(node[:m] == n)
        if len(ii) == 0:
            continue
        # A value-aware feasible warm start carries most of the allocation; SAC
        # learns a bounded residual (multiplicative range about 0.78--1.28).  This
        # prevents an exploratory actor from destroying deadline feasibility.
        v = np.maximum(ctx["importance"][ii] * ctx["q"][ii], 0.05)
        base_p = v * np.maximum(ctx["gain"][ii], 1e-15) ** (-0.25)
        base_f = v
        wp = base_p * np.exp(0.25 * raw_p[ii])
        wf = base_f * np.exp(0.25 * raw_f[ii])
        # The action controls both the within-node split and how much of each
        # budget is activated.  This makes utilization and energy observable
        # decisions instead of forcing every method to consume 100% capacity.
        p_fraction = 0.90 + 0.10 * (float(np.mean(raw_p[ii])) + 1.0) / 2.0
        f_fraction = 0.90 + 0.10 * (float(np.mean(raw_f[ii])) + 1.0) / 2.0
        proposed_p = ctx["node_power"][n] * p_fraction * wp / wp.sum()
        proposed_f = ctx["node_cpu"][n] * f_fraction * wf / wf.sum()
        p[ii] = proposed_p
        f[ii] = proposed_f
    return p[:m], f[:m]


def allocation_outcome(ctx: Mapping[str, np.ndarray], p: np.ndarray, f: np.ndarray) -> Dict[str, np.ndarray | float]:
    m = int(ctx["m"][0])
    node = ctx["node"][:m].astype(int)
    gain = ctx["gain"][:m]
    received = p * gain
    interference = np.zeros(m)
    for i in range(m):
        peers = (node == node[i])
        interference[i] = ctx["alpha"][0] * max(0.0, received[peers].sum() - received[i])
    sinr = received / (ctx["noise"][0] + interference + EPS)
    counts = np.array([max(1, np.sum(node == node[i])) for i in range(m)])
    bandwidth = ctx["bandwidth"][0] / counts
    rate = bandwidth * np.log2(1.0 + sinr)
    tx = ctx["data"][:m] / np.maximum(rate, 1e4)
    comp = ctx["cycles"][:m] / np.maximum(f, 1e6)
    latency = tx + comp
    sinr_db = 10 * np.log10(np.maximum(sinr, 1e-12))
    completed = (latency <= ctx["deadline"][:m]) & (sinr_db >= ctx["rel_db"][:m])
    energy = p * tx + 2e-29 * ctx["cycles"][:m] * f ** 2
    cpu_util, power_util = [], []
    for n in range(len(ctx["node_cpu"])):
        ii = node == n
        if not np.any(ii) or ctx["node_cpu"][n] <= 0:
            continue
        cpu_util.append(float(f[ii].sum() / ctx["node_cpu"][n]))
        power_util.append(float(p[ii].sum() / max(ctx["node_power"][n], EPS)))
    return {
        "latency": latency, "sinr_db": sinr_db, "completed": completed,
        "energy": energy, "cpu_util": float(np.mean(cpu_util) if cpu_util else 0),
        "power_util": float(np.mean(power_util) if power_util else 0),
    }


class ResourceAllocationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_b: int = 10, seed: int = 0):
        super().__init__()
        self.max_b = max_b
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2 * max_b,), dtype=np.float32)
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(11 * max_b + 3,), dtype=np.float32)
        self.ctx: Dict[str, np.ndarray] = {}

    def _sample_context(self) -> Dict[str, np.ndarray]:
        b = self.max_b
        m = int(self.rng.integers(max(4, b - 3), b + 1))
        n_nodes = 4
        node = np.pad(self.rng.integers(0, n_nodes, size=m), (0, b - m), constant_values=-1)
        cv = self.rng.uniform(0.0, 0.65)
        z = np.linspace(-1, 1, n_nodes); z /= np.std(z)
        cpu = 36e9 * np.clip(1 + cv * z, 0.35, 1.8)
        power = 8.0 * np.clip(1 + 0.7 * cv * z[::-1], 0.4, 1.7)
        pad = b - m
        def p(a: np.ndarray, fill: float = 0.0) -> np.ndarray:
            return np.pad(a, (0, pad), constant_values=fill)
        congestion = self.rng.uniform(0.8, 2.1)
        return {
            "m": np.array([m]), "node": node,
            "gain": p(10 ** self.rng.uniform(-12.7, -8.7, size=m)),
            "data": p(self.rng.uniform(0.5e6, 2.8e6, size=m)),
            "cycles": p(self.rng.uniform(0.06e9, 0.45e9, size=m)),
            "deadline": p(self.rng.uniform(0.075, 0.200, size=m), 0.20),
            "rel_db": p(self.rng.choice([0.0, 2.0], size=m, p=[0.65, 0.35])),
            "importance": p(self.rng.uniform(0.25, 1.0, size=m)),
            "q": p(self.rng.uniform(0.35, 1.0, size=m)),
            "node_cpu": cpu, "node_power": power,
            "bandwidth": np.array([100e6 / congestion]),
            "noise": np.array([1e-12]), "alpha": np.array([0.035 * congestion]),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.ctx = self._sample_context()
        return allocation_observation(self.ctx, self.max_b), {}

    def step(self, action: np.ndarray):
        p, f = action_to_resources(action, self.ctx, self.max_b)
        out = allocation_outcome(self.ctx, p, f)
        m = int(self.ctx["m"][0])
        lat = np.asarray(out["latency"])
        deadline = self.ctx["deadline"][:m]
        sinr = np.asarray(out["sinr_db"])
        smooth_lat = 1.0 / (1.0 + np.exp(np.clip(8.0 * (lat / deadline - 1.0), -30, 30)))
        smooth_rel = 1.0 / (1.0 + np.exp(np.clip(-2.0 * (sinr - self.ctx["rel_db"][:m]), -30, 30)))
        weights = self.ctx["importance"][:m] * self.ctx["q"][:m]
        satisfaction = float(np.sum(weights * smooth_lat * smooth_rel) / max(np.sum(weights), EPS))
        energy = float(np.mean(np.asarray(out["energy"])))
        excess = float(np.mean(np.maximum(lat / deadline - 1.0, 0.0)))
        completion = float(np.mean(np.asarray(out["completed"], dtype=float)))
        reward = 1.6 * satisfaction + 0.7 * completion \
            - 0.010 * min(energy, 8.0) - 0.25 * min(excess, 4.0)
        obs = allocation_observation(self.ctx, self.max_b)
        info = {"satisfaction": satisfaction, "energy": energy, "excess": excess}
        return obs, float(reward), True, False, info


def train_sac(cfg: SimConfig, quick: bool, force: bool) -> SAC:
    path = CHECKPOINT_DIR / "sac_allocator"
    zip_path = path.with_suffix(".zip")
    if zip_path.exists() and not force:
        return SAC.load(str(path))
    seed_everything(2026)
    env = Monitor(ResourceAllocationEnv(cfg.admission_budget, seed=2026))
    steps = 5_000 if quick else 36_000
    model = SAC(
        "MlpPolicy", env, learning_rate=3e-4, buffer_size=max(steps, 5000),
        learning_starts=min(1000, steps // 5), batch_size=128, gamma=0.0,
        tau=0.01, train_freq=1, gradient_steps=1,
        policy_kwargs={"net_arch": [128, 128]}, verbose=0, seed=2026,
    )
    model.learn(total_timesteps=steps, progress_bar=False)
    model.save(str(path))
    return model


def allocate(scene: Scene, selected: Sequence[int], allocator: str, sac: SAC,
             max_b: int, q_hint: Optional[np.ndarray] = None,
             imp_hint: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray | float], float]:
    ctx = _allocation_arrays(scene, selected, max_b, q_hint=q_hint, imp_hint=imp_hint)
    t0 = time.perf_counter()
    m = len(selected)
    if allocator == "sac":
        obs = allocation_observation(ctx, max_b)
        action, _ = sac.predict(obs, deterministic=True)
        p, f = action_to_resources(action, ctx, max_b)
    elif allocator == "equal":
        p = np.zeros(m); f = np.zeros(m)
        node = ctx["node"][:m].astype(int)
        for n in range(len(ctx["node_cpu"])):
            ii = np.flatnonzero(node == n)
            if len(ii):
                p[ii] = ctx["node_power"][n] / len(ii)
                f[ii] = ctx["node_cpu"][n] / len(ii)
    elif allocator == "fixed":
        # Fixed-interference resource realization: a single deterministic update
        # apportions radio/compute by deadline-normalized demand, then projects the
        # shared budgets.  It does not react to the coupled interference outcome.
        node = ctx["node"][:m].astype(int)
        gain = ctx["gain"][:m]
        deadline = ctx["deadline"][:m]
        p = np.zeros(m); f = np.zeros(m)
        for n in range(len(ctx["node_cpu"])):
            ii = np.flatnonzero(node == n)
            if not len(ii):
                continue
            radio_demand = ctx["data"][ii] / np.maximum(deadline[ii] * gain[ii], 1e-15)
            compute_demand = ctx["cycles"][ii] / np.maximum(deadline[ii], 0.01)
            p[ii] = ctx["node_power"][n] * radio_demand / radio_demand.sum()
            f[ii] = ctx["node_cpu"][n] * compute_demand / compute_demand.sum()
    else:
        raise ValueError(allocator)
    out = allocation_outcome(ctx, p, f)
    return out, (time.perf_counter() - t0) * 1000.0


METHODS = {
    "SERVO": ("gnn", "submodular", "sac", "ph"),
    "MLP+Submod+SAC": ("mlp", "submodular", "sac", "ph"),
    "GNN+WUM+SAC": ("gnn", "additive", "sac", "ph"),
    "GNN+Submod+FI": ("gnn", "submodular", "fixed", "ph"),
    "GNN+Submod+Equal": ("gnn", "submodular", "equal", "ph"),
    "SERVO-NoAdapt": ("gnn", "submodular", "sac", "none"),
    "LoadAware+Equal": ("gnn", "load_aware", "equal", "ema"),
}


def run_trace(seed: int, method: str, cfg: SimConfig, scenario: Scenario,
              gnn: nn.Module, mlp: nn.Module, sac: SAC) -> Tuple[pd.DataFrame, List[int]]:
    est_name, selector, allocator, adapt_mode = METHODS[method]
    estimator = gnn if est_name == "gnn" else mlp
    calibrator = OnlineCalibrator(adapt_mode)
    rng = np.random.default_rng(seed * 1297 + 9)
    rows: List[dict] = []
    for t in range(cfg.slots):
        scene = make_scene(seed, t, cfg, scenario)
        t0 = time.perf_counter()
        q_raw, i_raw = predict_scene(estimator, scene)
        q_pred, i_pred = calibrator.apply(scene, q_raw, i_raw)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        selected = select_flows(scene, q_pred, i_pred, cfg.admission_budget, selector, rng)
        select_ms = (time.perf_counter() - t1) * 1000.0
        out, alloc_ms = allocate(scene, selected, allocator, sac, cfg.admission_budget,
                                 q_hint=q_pred, imp_hint=i_pred)
        completed = np.asarray(out["completed"], dtype=bool)
        true_value = coverage_value(scene, selected, scene.q_true, scene.imp_true, completed)
        full_value = coverage_value(scene, list(range(len(scene.flow_x))), scene.q_true, scene.imp_true)
        selected_value = coverage_value(scene, selected, scene.q_true, scene.imp_true)
        lat_ms = np.asarray(out["latency"], dtype=float) * 1000.0
        completed_latency = lat_ms[completed]
        energy = np.asarray(out["energy"], dtype=float)
        selected_edges = [e for k in selected for e in scene.flow_event_edges[k].values()]
        selected_events = sorted({w for k in selected for w in scene.flow_event_edges[k]})
        q_mae = float(np.mean(np.abs(q_pred[selected_edges] - scene.q_true[selected_edges]))) \
            if selected_edges else np.nan
        i_mae = float(np.mean(np.abs(i_pred[selected_events] - scene.imp_true[selected_events]))) \
            if selected_events else np.nan
        prediction_mae = 0.55 * q_mae + 0.45 * i_mae
        node_load = np.bincount(scene.node_id[np.asarray(selected, dtype=int)], minlength=cfg.n_nodes) \
            if selected else np.zeros(cfg.n_nodes)
        # Calibrate against the estimator's raw output; otherwise multiplying an
        # already-calibrated prediction would double-count the correction.
        calibrator.update(t, scene, selected, q_raw, i_raw)
        rows.append({
            "seed": seed, "slot": t, "method": method, "scenario": scenario.name,
            "value": true_value / max(full_value, EPS),
            "selected_value": selected_value / max(full_value, EPS),
            "completion": float(np.mean(completed)) if len(completed) else 0.0,
            "sla_violation": float(1 - np.mean(completed)) if len(completed) else 1.0,
            "latency_mean_ms": float(np.mean(completed_latency)) if len(completed_latency) else np.nan,
            "latency_p95_ms": float(np.percentile(completed_latency, 95)) if len(completed_latency) else np.nan,
            "energy_total_j": float(np.sum(energy)),
            "energy_j": float(np.sum(energy) / max(np.sum(completed), 1)),
            "cpu_util": float(out["cpu_util"]), "power_util": float(out["power_util"]),
            "overhead_ms": infer_ms + select_ms + alloc_ms,
            "infer_ms": infer_ms, "select_ms": select_ms, "alloc_ms": alloc_ms,
            "n_requests": len(scene.flow_x), "n_selected": len(selected),
            "max_node_load": float(np.max(node_load)) if len(node_load) else 0.0,
            "q_mae": q_mae, "importance_mae": i_mae,
            "prediction_mae": prediction_mae,
        })
    return pd.DataFrame(rows), calibrator.detections


def _ci95(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    return float(1.96 * np.std(a, ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0


def summarize_runs(raw: pd.DataFrame) -> pd.DataFrame:
    per_seed = raw.groupby(["scenario", "method", "seed"], as_index=False).agg(
        value=("value", "mean"), latency_p95_ms=("latency_p95_ms", "mean"),
        sla_violation=("sla_violation", "mean"), completion=("completion", "mean"),
        energy_total_j=("energy_total_j", "mean"), energy_j=("energy_j", "mean"),
        cpu_util=("cpu_util", "mean"),
        power_util=("power_util", "mean"), overhead_ms=("overhead_ms", "mean"),
        prediction_mae=("prediction_mae", "mean"),
    )
    rows = []
    for (scenario, method), g in per_seed.groupby(["scenario", "method"]):
        row = {"scenario": scenario, "method": method, "seeds": int(g.seed.nunique())}
        for metric in ["value", "latency_p95_ms", "sla_violation", "completion",
                       "energy_total_j", "energy_j", "cpu_util", "power_util", "overhead_ms",
                       "prediction_mae"]:
            row[metric] = float(g[metric].mean())
            row[metric + "_ci95"] = _ci95(g[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def recovery_time(raw: pd.DataFrame, drift_slot: int) -> Tuple[float, float]:
    """Censored time to remove half of the drift-induced prediction-error increase."""
    times, successes = [], []
    for _, g in raw.groupby("seed"):
        g = g.sort_values("slot")
        pre = float(g[(g.slot >= drift_slot - 20) & (g.slot < drift_slot)].prediction_mae.mean())
        post = g[g.slot >= drift_slot]
        vals = post.prediction_mae.to_numpy()
        spike = float(np.mean(vals[:5]))
        target = pre + 0.5 * max(spike - pre, 0.02)
        horizon = len(vals)
        found = float(horizon)
        success = 0.0
        for j in range(max(0, len(vals) - 4)):
            if np.mean(vals[j:j + 5]) <= target:
                found = float(j)
                success = 1.0
                break
        times.append(found)
        successes.append(success)
    return float(np.mean(times)), float(np.mean(successes))


def evaluate_all(cfg: SimConfig, quick: bool, gnn: nn.Module, mlp: nn.Module, sac: SAC) -> Dict[str, pd.DataFrame]:
    seeds = list(range(3 if quick else 10))
    slots_main = 35 if quick else cfg.slots
    main_cfg = SimConfig(**{**asdict(cfg), "slots": slots_main})
    raw_parts = []
    for method in METHODS:
        for s in seeds:
            df, _ = run_trace(s, method, main_cfg, Scenario(name="nominal"), gnn, mlp, sac)
            raw_parts.append(df)
    raw_main = pd.concat(raw_parts, ignore_index=True)
    summary = summarize_runs(raw_main)

    robustness_specs = [
        Scenario(name="nominal"),
        Scenario(name="burst", load_multiplier=1.35),
        Scenario(name="high-mobility", speed_multiplier=1.75),
        Scenario(name="heterogeneous", capacity_cv=0.65),
        Scenario(name="node-failure", fail_node=True),
        Scenario(name="congestion", congestion_multiplier=1.8),
        Scenario(name="concept-drift", concept_drift=True),
    ]
    robust_methods = ["SERVO", "SERVO-NoAdapt", "LoadAware+Equal"]
    rseeds = list(range(2 if quick else 6))
    robust_cfg = SimConfig(**{**asdict(cfg), "slots": 45 if quick else cfg.slots,
                              "drift_slot": 20 if quick else cfg.drift_slot,
                              "failure_slot": 20 if quick else cfg.failure_slot})
    robust_parts, detection_rows = [], []
    for spec in robustness_specs:
        for method in robust_methods:
            for s in rseeds:
                df, detections = run_trace(100 + s, method, robust_cfg, spec, gnn, mlp, sac)
                robust_parts.append(df)
                for d in detections:
                    detection_rows.append({"scenario": spec.name, "method": method, "seed": s, "detection_slot": d})
    raw_robust = pd.concat(robust_parts, ignore_index=True)
    robust_summary = summarize_runs(raw_robust)

    # Recovery comparison uses a common drift trace and three adaptation rules.
    original = METHODS.copy()
    METHODS["SERVO-EMA"] = ("gnn", "submodular", "sac", "ema")
    METHODS["SERVO-Periodic"] = ("gnn", "submodular", "sac", "periodic")
    rec_methods = ["SERVO", "SERVO-EMA", "SERVO-Periodic", "SERVO-NoAdapt"]
    rec_parts = []
    for method in rec_methods:
        for s in rseeds:
            df, detections = run_trace(200 + s, method, robust_cfg,
                                       Scenario(name="concept-drift", concept_drift=True), gnn, mlp, sac)
            rec_parts.append(df)
            for d in detections:
                detection_rows.append({"scenario": "concept-drift", "method": method,
                                       "seed": s, "detection_slot": d})
    raw_recovery = pd.concat(rec_parts, ignore_index=True)
    recovery_rows = []
    for method, g in raw_recovery.groupby("method"):
        rec_slots, rec_success = recovery_time(g, robust_cfg.drift_slot)
        recovery_rows.append({"method": method,
                              "recovery_slots": rec_slots,
                              "recovery_success": rec_success,
                              "post_drift_value": float(g[g.slot >= robust_cfg.drift_slot].value.mean()),
                              "post_drift_sla_violation": float(g[g.slot >= robust_cfg.drift_slot].sla_violation.mean()),
                              "post_drift_prediction_mae": float(g[g.slot >= robust_cfg.drift_slot].prediction_mae.mean())})
    recovery = pd.DataFrame(recovery_rows)
    METHODS.clear(); METHODS.update(original)

    # Scale services, events, and nodes jointly; retain the fixed admission budget.
    scales = [(36, 20, 4), (72, 36, 4), (128, 64, 8), (224, 96, 12)]
    scale_rows = []
    for nv, ne, nnodes in scales:
        scfg = SimConfig(**{**asdict(cfg), "slots": 15 if quick else 35,
                            "n_vehicles": nv, "n_events": ne, "n_nodes": nnodes})
        for method in ["SERVO", "GNN+WUM+SAC"]:
            parts = []
            for s in range(2 if quick else 5):
                df, _ = run_trace(300 + s, method, scfg, Scenario(name="scale"), gnn, mlp, sac)
                parts.append(df)
            dd = pd.concat(parts)
            per = dd.groupby("seed", as_index=False).mean(numeric_only=True)
            scale_rows.append({"vehicles": nv, "events": ne, "nodes": nnodes,
                               "method": method, "value": float(per.value.mean()),
                               "value_ci95": _ci95(per.value),
                               "overhead_ms": float(per.overhead_ms.mean()),
                               "overhead_ci95": _ci95(per.overhead_ms),
                               "sla_violation": float(per.sla_violation.mean())})
    scalability = pd.DataFrame(scale_rows)

    return {
        "raw_main": raw_main, "summary": summary,
        "raw_robustness": raw_robust, "robustness": robust_summary,
        "raw_recovery": raw_recovery, "recovery": recovery,
        "detections": pd.DataFrame(detection_rows), "scalability": scalability,
    }


PALETTE = {
    "SERVO": "#0072B2", "MLP+Submod+SAC": "#56B4E9", "GNN+WUM+SAC": "#D55E00",
    "GNN+Submod+FI": "#009E73", "GNN+Submod+Equal": "#E69F00",
    "SERVO-NoAdapt": "#CC79A7", "LoadAware+Equal": "#777777",
    "SERVO-EMA": "#56B4E9", "SERVO-Periodic": "#E69F00",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.2, "axes.labelsize": 8.2, "axes.titlesize": 8.6,
        "legend.fontsize": 7.0, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 320,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(data: Mapping[str, pd.DataFrame], drift_slot: int = 60) -> None:
    _style()
    summary = data["summary"].set_index("method")
    order = [m for m in METHODS if m in summary.index]
    metrics = [
        ("value", "Delivered value $\\uparrow$", (0, 1)),
        ("latency_p95_ms", "P95 latency (ms) $\\downarrow$", None),
        ("sla_violation", "SLA violation $\\downarrow$", (0, 1)),
        ("completion", "Completion rate $\\uparrow$", (0, 1)),
        ("energy_total_j", "Energy / slot (J) $\\downarrow$", None),
        ("overhead_ms", "Orchestration overhead (ms) $\\downarrow$", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.45))
    for ax, (metric, title, ylim) in zip(axes.flat, metrics):
        vals = summary.loc[order, metric].to_numpy()
        err = summary.loc[order, metric + "_ci95"].to_numpy()
        x = np.arange(len(order))
        colors = [PALETTE[m] if m == "SERVO" else "#A6A6A6" for m in order]
        hatches = ["" if m == "SERVO" else "//" if "WUM" in m else "" for m in order]
        bars = ax.bar(x, vals, yerr=err, capsize=2, color=colors, edgecolor="#333333", linewidth=0.45)
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
        ax.set_title(title, fontweight="bold")
        short_label = {
            "SERVO": "Full", "MLP+Submod+SAC": "MLP", "GNN+WUM+SAC": "WUM",
            "GNN+Submod+FI": "FI", "GNN+Submod+Equal": "Equal",
            "SERVO-NoAdapt": "NoAdapt", "LoadAware+Equal": "Load-aware",
        }
        ax.set_xticks(x); ax.set_xticklabels([short_label[m] for m in order],
                                             rotation=24, ha="right")
        if ylim: ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle=":", color="#D9D9D9", linewidth=0.6)
    fig.subplots_adjust(wspace=0.34, hspace=0.52)
    _save(fig, "fig_e2e_main")

    robust = data["robustness"]
    scenarios = ["nominal", "burst", "high-mobility", "heterogeneous", "node-failure", "congestion", "concept-drift"]
    methods = ["SERVO", "SERVO-NoAdapt", "LoadAware+Equal"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65))
    x = np.arange(len(scenarios)); width = 0.24
    for j, m in enumerate(methods):
        g = robust[robust.method == m].set_index("scenario").reindex(scenarios)
        axes[0].bar(x + (j - 1) * width, g.value, width, yerr=g.value_ci95,
                    color=PALETTE[m], edgecolor="#333333", linewidth=0.4, capsize=1.5, label=m)
        axes[1].bar(x + (j - 1) * width, g.sla_violation, width, yerr=g.sla_violation_ci95,
                    color=PALETTE[m], edgecolor="#333333", linewidth=0.4, capsize=1.5, label=m)
    labels = ["Nom.", "Burst", "Fast", "Hetero.", "Failure", "Congest.", "Drift"]
    for ax, title, ylabel in zip(axes, ["(a) Delivered value", "(b) SLA violation"],
                                 ["normalized value $\\uparrow$", "violation rate $\\downarrow$"]):
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_title(title, fontweight="bold"); ax.set_ylabel(ylabel); ax.grid(axis="y", ls=":", color="#D9D9D9")
    axes[0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(1.05, 1.02))
    fig.tight_layout()
    _save(fig, "fig_e2e_robustness")

    scale = data["scalability"]
    rec_raw = data["raw_recovery"]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
    for m, marker, ls in [("SERVO", "o", "-"), ("GNN+WUM+SAC", "s", "--")]:
        g = scale[scale.method == m].sort_values("vehicles")
        axes[0].errorbar(g.vehicles, g.value, yerr=g.value_ci95, marker=marker, ls=ls,
                         color=PALETTE[m], capsize=2, label=m)
        axes[1].errorbar(g.vehicles, g.overhead_ms, yerr=g.overhead_ci95, marker=marker, ls=ls,
                         color=PALETTE[m], capsize=2, label=m)
    axes[0].set_title("(a) Value under scale", fontweight="bold"); axes[0].set_ylabel("normalized value $\\uparrow$")
    axes[1].set_title("(b) Control overhead", fontweight="bold"); axes[1].set_ylabel("ms / slot $\\downarrow$")
    for ax in axes[:2]:
        ax.set_xlabel("concurrent vehicles"); ax.grid(axis="y", ls=":", color="#D9D9D9")
    axes[0].legend(frameon=False, fontsize=6.5)
    for m in ["SERVO", "SERVO-EMA", "SERVO-Periodic", "SERVO-NoAdapt"]:
        g = rec_raw[rec_raw.method == m].groupby("slot").prediction_mae.mean()
        axes[2].plot(g.index, g.values, color=PALETTE[m], label=m,
                     lw=1.5 if m == "SERVO" else 1.0,
                     ls="-" if m == "SERVO" else "--" if "NoAdapt" in m else "-.")
    axes[2].axvline(drift_slot, color="#222222", ls=":", lw=0.9)
    axes[2].text(drift_slot + 1, axes[2].get_ylim()[0] + 0.02, "drift", fontsize=6.5)
    axes[2].set_title("(c) Calibration recovery", fontweight="bold")
    axes[2].set_xlabel("slot"); axes[2].set_ylabel("prediction MAE $\\downarrow$")
    axes[2].grid(axis="y", ls=":", color="#D9D9D9"); axes[2].legend(frameon=False, fontsize=5.8)
    fig.tight_layout()
    _save(fig, "fig_e2e_scale_recovery")

    # Estimator error figure directly connects the GNN ablation to its source metric.
    metrics_path = CHECKPOINT_DIR / "coverage_metrics.json"
    if metrics_path.exists():
        mm = json.loads(metrics_path.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(3.45, 2.35))
        labels = ["GNN\ncoverage", "MLP\ncoverage", "GNN\nimportance", "MLP\nimportance"]
        vals = [mm["gnn_q_mae"], mm["mlp_q_mae"], mm["gnn_importance_mae"], mm["mlp_importance_mae"]]
        ax.bar(np.arange(4), vals, color=[PALETTE["SERVO"], "#A6A6A6", PALETTE["SERVO"], "#A6A6A6"],
               edgecolor="#333333", linewidth=0.5)
        ax.set_xticks(np.arange(4)); ax.set_xticklabels(labels); ax.set_ylabel("held-out MAE $\\downarrow$")
        ax.set_title("Coverage-estimator validation", fontweight="bold")
        ax.grid(axis="y", ls=":", color="#D9D9D9")
        fig.tight_layout(); _save(fig, "fig_gnn_validation")


def write_outputs(data: Mapping[str, pd.DataFrame], cfg: SimConfig,
                  model_metrics: Mapping[str, float], quick: bool) -> None:
    for name, df in data.items():
        df.to_csv(RESULT_DIR / f"{name}.csv", index=False)
    manifest = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "quick_mode": quick,
        "config": asdict(cfg),
        "seeds_main": sorted(data["raw_main"].seed.unique().tolist()),
        "model_metrics": dict(model_metrics),
        "software": {
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "torch": torch.__version__, "stable_baselines3": __import__("stable_baselines3").__version__,
        },
        "scope": "dynamic system-level V2X simulation; not NS-3/SUMO packet-level emulation",
    }
    (RESULT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Human-readable compact block used when revising the manuscript.
    main = data["summary"].copy()
    rec = data["recovery"].copy()
    with (RESULT_DIR / "paper_numbers.md").open("w", encoding="utf-8") as f:
        f.write("# End-to-end SERVO results\n\n")
        f.write("Generated by `experiments/e2e_servo.py`; values are not hand-entered.\n\n")
        f.write(main.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Recovery\n\n")
        f.write(rec.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke run")
    parser.add_argument("--force-train", action="store_true", help="retrain GNN/MLP/SAC")
    args = parser.parse_args()
    seed_everything(2026)
    cfg = SimConfig()
    print("[1/5] training/loading coverage estimators", flush=True)
    gnn, mlp, model_metrics = train_estimators(cfg, args.quick, args.force_train)
    print(json.dumps(model_metrics, indent=2), flush=True)
    print("[2/5] training/loading SAC allocator", flush=True)
    sac = train_sac(cfg, args.quick, args.force_train)
    print("[3/5] running integrated, robustness, recovery, and scale experiments", flush=True)
    data = evaluate_all(cfg, args.quick, gnn, mlp, sac)
    print("[4/5] writing source data and figures", flush=True)
    write_outputs(data, cfg, model_metrics, args.quick)
    make_figures(data, 20 if args.quick else cfg.drift_slot)
    print("[5/5] complete", flush=True)
    print(data["summary"].to_string(index=False), flush=True)
    print(data["recovery"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

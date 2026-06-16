# SERVO — Submodular Coverage-Aware Orchestration (selection-layer simulator)

Reproducible Monte-Carlo simulation code for the paper

> **SERVO: Submodular Coverage-Aware Orchestration of Edge Services in the 6G
> Sensing–Communication–Computing Continuum**, J. Lin *et al.*,
> IEEE Transactions on Services Computing (under review, 2026).

This repository releases the **coverage-aware selection layer**: the monotone-submodular
coverage value model and every selection method compared in the paper, implemented in
**pure NumPy on a single CPU core** with **fixed random seeds**, so every table number and
data figure can be reproduced exactly — no GPU, no proprietary data.

---

## What this code reproduces

Running the scripts regenerates, from scratch, all selection-layer results in the paper:

| Result | Script | Headline numbers (correlated regime, default settings) |
|---|---|---|
| Comparison vs. budget, redundancy, ablation, drift, approx. ratio | `code/servo_sim.py` | SERVO `0.880`, WUM `0.759`, MAB-CP `0.742`; r̂ = `0.987 ± 0.026` (200 instances, all > 1−1/e) |
| 95% CIs, importance-independent regime, paired significance | `code/servo_extra.py` | SERVO−WUM = `+0.121 ± 0.017` (paired, 20 seeds) |
| REINFORCE deep-RL selector (DRL-Sel) | `code/servo_drl.py` | `0.824` (correlated) / `0.804` (independent) |
| 10-method comparison incl. VoI + CTS, all figures | `code/servo_v2figs.py` | SERVO `0.880`, CTS `0.877`, VoI `0.872`, DRL `0.824` |

All ten selection methods are faithfully implemented: **SERVO** (lazy/CELF greedy submodular
maximization), **VoI** (noisy-OR value-of-information greedy), **WUM** / **WUM-RD** (additive
and redundancy-discounted weighted utility), **OSM** (sieve-streaming submodular),
**MAB-CP** (UCB bandit), **CTS** (combinatorial Thompson sampling), **DRL-Sel** (a REINFORCE
neural policy-gradient selector), **PF/Prio**, **Random**, and the brute-force **OPT** at
small scale.

## Scope (what is and is **not** in this release)

**Included and reproduced here:** the synthetic coverage-aware *selection layer* — the submodular
value oracle and all ten selectors above, the drift / self-evolution tracking, and the
approximation-ratio study against the exact optimum.

**Not included** (presented in the paper as design and theory, with empirical validation
deferred to the ongoing campaign):

- packet-/traffic-level network realism (NS-3 / SUMO) and hardware-in-the-loop;
- the soft actor–critic (SAC) within-set sub-allocator experiments;
- the coverage GNN trained and validated on **real** cooperative-perception traces;
- the queue-aware Lyapunov-DRL selection baseline.

Problem instances are **synthetic** (drawn from the importance-tier / Poisson coverage model
of the paper, Table III), not real-world datasets.

## Requirements

```
python >= 3.9
numpy
scipy
matplotlib
```

```bash
pip install -r requirements.txt
```

CPU only; a full run needs no GPU. First run of `servo_v2figs.py` trains the DRL policies
(~a few minutes on one core) and caches them, so later runs are fast.

## Quickstart

```bash
cd code

# core results (E1–E6): prints numbers, writes servo_results.json, tables, and figures
python servo_sim.py

# revision experiments: 95% CIs, importance-independent regime, paired significance
python servo_extra.py

# the trained DRL-Sel baseline on its own
python servo_drl.py

# the full 10-method comparison (adds VoI + CTS) and the publication figures
python servo_v2figs.py

# dense parameter / scalability / redundancy / approximation figures
python servo_figs.py
```

Each script writes its outputs (figures as `.pdf`+`.png`, and `*_numbers.json`) next to
itself in `code/`. Results are deterministic: the same seeds reproduce the paper's numbers
bit-for-bit.

## Repository layout

```
code/
  servo_sim.py      core simulator: value oracle + all classical selectors + experiments E1–E6
  servo_extra.py    reviewer-requested CIs, importance-independent regime, paired significance
  servo_drl.py      REINFORCE deep-RL selection baseline (DRL-Sel), NumPy-only
  servo_v2figs.py   VoI + CTS baselines, 10-method comparison, main result figures
  servo_figs.py     dense parameter/scalability/redundancy/approx-ratio figures
  servo_restyle.py  editorial re-render of all data figures (uses servo_style.py)
  servo_style.py    figure styling (palette, labels) — imported by servo_restyle.py
  fig_teaser.py     Fig. 1 concept schematic
  servo_hero_sys.py system-model schematic
data/
  servo_results.json, servo_extra_results.json, servo_v2_numbers.json, servo_figs_numbers.json
  table_comprehensive.csv, table_ablation.csv
  SERVO_simulation_results.md   human-readable results summary
```

## Citation

```bibtex
@article{lin2026servo,
  author  = {Lin, Jiaqi and Yan, Shi and Zhang, Shenhu and Wang, Yong and
             Sun, Yifu and Xu, Chengwei and Peng, Mugen},
  title   = {{SERVO}: Submodular Coverage-Aware Orchestration of Edge Services in the
             {6G} Sensing--Communication--Computing Continuum},
  journal = {IEEE Transactions on Services Computing},
  note    = {under review},
  year    = {2026}
}
```

## License

Released under the MIT License — see [LICENSE](LICENSE).

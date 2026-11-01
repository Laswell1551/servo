# Corrected SERVO revision artifact — 1 November 2026

The current paper uses `revision_consistency.py` and `results_consistency/`.
The previous `e2e_servo.py` command-line pipeline, `results/`, and its legacy
SAC checkpoint are superseded. That module is still imported for shared scene,
estimator, allocation, and calibration functions. The original `code/` and
`data/` track concerns fixed-quality selection only.

## Correction and scope

The previous physical calculation added a dimensionless `1e-9` guard to a SINR
denominator whose noise power was `1e-12` W. The corrected calculation uses
`max(noise + interference, 1e-30)` in watts. Both allocator reward variants
were retrained for 36,000 transitions, and all system-level results were rerun.
Do not mix the previous results with the corrected dataset.

Admission count, node count, power, and compute budgets are hard constraints.
Service success is checked after actual interference and measured serial
control-plus-service latency. Budget normalization does not certify an SLA.
Reference sensing quality and the normalization denominator are independent of
controller actions. The main results concern independent network snapshots;
they do not establish continuous queueing or cross-epoch reservations.

## Commands

From the repository root, install `requirements.txt`, then:

```bash
python experiments/verify_consistency.py
python experiments/diagnose_consistency.py
python experiments/present_consistency.py
```

These verify the released results, reconstruct 3,840 logged event outcomes with
their recorded control times, and generate tables and figures. No training is
needed with the released checkpoints. Figure generation writes to `manuscript/`.

To run all traces again, using the released models:

```bash
python experiments/revision_consistency.py --train-steps 36000
python experiments/verify_consistency.py
python experiments/diagnose_consistency.py
python experiments/present_consistency.py
```

This overwrites `results_consistency/`. Preserve the released directory if
comparing runs. Add `--force-train` to retrain the two SAC policies. The fixed
GNN/MLP checkpoints remain in `checkpoints/`; current SAC policies are in
`checkpoints_consistency/`. A `--smoke` run is only a software check and cannot
be used to generate the paper. Timing and timing-dependent success may differ
between machines or under different host load.

## Experimental controls

- Ten nominal methods: full SERVO, MLP, additive selection, FI-1, full Equal,
  resource-matched Equal, Reference, FI-CVX, CostCov, and additive-reward SAC.
- Equal-match reproduces actual actor power/compute totals per node and snapshot,
  then splits them equally. Its timing includes the actor used to obtain totals.
- FI-1 is a single-update heuristic. FI-CVX actually solves the frozen-interference
  energy problem using CVXPY 1.6.7 and CLARABEL; infeasibility/failure triggers an
  explicitly logged Reference fallback. Service-only and serial outcomes differ.
- CostCov adapts marginal coverage per demand cost from budgeted maximum coverage.
  It is not a reproduction of Jia et al.'s single-peer AVUCB policy.
- Admission budgets grow 4/8/12/16 with vehicles/events/nodes. Resource sweeps
  hold the admitted set fixed. The small exact oracle bounds reference coverage,
  not the resource-coupled delivered-value optimum.
- Confidence intervals use trace-seed means with paired comparisons on common
  traces. Training seed 2026 is fixed; intervals exclude training-seed variation.

`run_manifest.json` records the executed script hash and run settings.
`release_manifest.json` records code, model and data hashes plus every study's
seed range. `verification.json` records scientific and raw-table checks.

The corrected evidence identifies coverage-aware selection as the main value
gain. It does not establish an extra average delivered-value gain from SAC over
Reference or resource-matched Equal. GNN and PH improve some prediction metrics,
but their paired delivered-value intervals include zero on these traces.

## References for the added controls

- S. Khuller, A. Moss, and J. S. Naor, ?The budgeted maximum coverage problem,? Information Processing Letters 70(1), 39?45, 1999. DOI: 10.1016/S0020-0190(99)00031-9.
- Y. Jia et al., ?Online V2X scheduling for raw-level cooperative perception,? IEEE ICC 2022, 309?314. DOI: 10.1109/ICC45855.2022.9838439. The comparator here is CostCov, not AVUCB reproduction.
- L. Chen, H. Hassani, and A. Karbasi, ?Online continuous submodular maximization,? AISTATS 2018, 1896?1905. https://proceedings.mlr.press/v84/chen18f.html. Meta-Frank-Wolfe Algorithm 1/Theorem 1 differs from projected OGA Algorithm 2/Theorem 2.

Released under the MIT License.

# SERVO end-to-end major-revision experiments

Run the full evidence pipeline from the repository root:

```bash
pip install -r requirements.txt
python experiments/e2e_servo.py --force-train
```

A smoke run is available with `--quick`. The script trains the coverage GNN,
the MLP estimator ablation, and the SAC resource allocator; replays dynamic V2X
traces; and writes all source tables to `experiments/results/`. Publication PDF
and PNG figures are written to `manuscript/`.

The environment is a reproducible system-level simulator with mobility,
distance-dependent fading, coupled interference, heterogeneous edge resources,
dynamic requests, failures, congestion, and drift. It is intentionally described
as such in the manuscript and is not presented as NS-3/SUMO packet emulation.

Key traceability files:

- `results/run_manifest.json`: configuration, seeds, software versions, and scope.
- `results/raw_main.csv`: per-slot integrated-framework observations.
- `results/summary.csv`: per-seed mean and 95% CI table.
- `results/raw_robustness.csv`: per-slot stress-test observations.
- `results/recovery.csv`: drift recovery comparison.
- `results/scalability.csv`: joint node/service/event scaling results.
- `results/paper_numbers.md`: compact manuscript-facing number table.
- `results/implementation_audit.md`: superseded-run diagnosis and corrective action.

After a full run, verify row counts, SLA identities, feasibility, finite ranges, and
summary reproducibility with:

```bash
python experiments/verify_results.py
```

The committed results were produced by a full run (`quick_mode: false`).
Released checkpoints permit a non-retraining run with
`python experiments/e2e_servo.py`; use `--force-train` to regenerate them.

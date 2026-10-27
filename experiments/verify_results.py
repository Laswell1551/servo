"""Consistency checks for the final SERVO revision evidence package."""
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "experiments" / "results"


def main() -> None:
    manifest = json.loads((R / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["quick_mode"] is False, "results are from a smoke run, not the full run"
    cfg = manifest["config"]
    main = pd.read_csv(R / "raw_main.csv")
    robust = pd.read_csv(R / "raw_robustness.csv")
    recovery = pd.read_csv(R / "raw_recovery.csv")
    summary = pd.read_csv(R / "summary.csv")

    assert len(main) == 7 * 10 * cfg["slots"]
    assert len(robust) == 3 * 7 * 6 * cfg["slots"]
    assert len(recovery) == 4 * 6 * cfg["slots"]
    assert summary.groupby("method").seeds.first().eq(10).all()
    assert np.allclose(main.completion + main.sla_violation, 1.0, atol=1e-10)
    assert main.cpu_util.max() <= 1.0 + 1e-9
    assert main.power_util.max() <= 1.0 + 1e-9
    for col in ["value", "completion", "sla_violation", "energy_total_j",
                "overhead_ms", "prediction_mae"]:
        assert np.isfinite(main[col]).all(), f"non-finite {col}"
    assert ((main.value >= 0) & (main.value <= 1 + 1e-9)).all()

    # Seed-level recomputation must reproduce the published summary means.
    per_seed = main.groupby(["method", "seed"], as_index=False).value.mean()
    means = per_seed.groupby("method").value.mean()
    reported = summary.set_index("method").value
    assert np.allclose(means.loc[reported.index], reported, atol=1e-12)

    print("PASS: full-run manifest, row counts, SLA identity, budget feasibility, "
          "finite ranges, and summary means")


if __name__ == "__main__":
    main()

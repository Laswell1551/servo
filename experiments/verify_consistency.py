"""Scientific regression checks and raw-result audit for the corrected revision."""
from pathlib import Path
import itertools
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import e2e_servo as b
from experiments import revision_consistency as r


def scientific_checks():
    ctx = dict(m=np.array([1]), node=np.array([0]), gain=np.array([1e-9]),
               data=np.array([1e6]), cycles=np.array([1e8]), deadline=np.array([.15]),
               rel_db=np.array([0.]), node_cpu=np.array([36e9]), node_power=np.array([8.]),
               bandwidth=np.array([100e6]), noise=np.array([1e-12]), alpha=np.array([0.]),
               importance=np.array([1.]), q=np.array([1.]))
    o = b.allocation_outcome(ctx, np.array([1.]), np.array([2e9]))
    assert abs(float(o['sinr_db'][0]) - 30.) < 1e-10, 'Noise-power regression'
    difficult = {**ctx, 'data': np.array([100e6]), 'deadline': np.array([.075])}
    o = b.allocation_outcome(difficult, np.array([8.]), np.array([36e9]))
    assert o['cpu_util'] <= 1 and o['power_util'] <= 1 and not o['completed'][0], \
        'Budget feasibility must not be equated with SLA success'
    p, f, feasible = r.convex_resources(ctx)
    assert feasible
    o = b.allocation_outcome(ctx, p, f)
    assert o['completed'].all()
    tau = np.linspace(.001, .149, 20000)
    power = 1e-3 * (2 ** (.01 / tau) - 1)
    cpu = 1e8 / (.15 - tau)
    valid = (power <= 8) & (cpu <= 36e9) & (power * 1e-9 / 1e-12 >= 1)
    energies = power * tau + 2e-29 * 1e8 * cpu ** 2
    optimum = float(np.min(energies[valid]))
    assert abs(float(o['energy'].sum()) - optimum) / optimum < 2e-3
    # A redundant flow improving below the event maximum must not increase utility.
    covctx = {'m': np.array([2]), 'truth_matrix': np.array([[.8], [.3]]), 'truth_imp': np.array([1.])}
    first = r.soft_coverage(covctx, np.ones(2))
    covctx['truth_matrix'][1, 0] = .5
    assert r.soft_coverage(covctx, np.ones(2)) == first
    for rho in range(2, 7):
        eps = .001
        events = [0] * rho + list(range(1, rho + 1))
        weights = [1 + eps] + [1.] * rho
        val = lambda s: sum(weights[e] for e in {events[k] for k in s})
        exact = max(val(s) for s in itertools.combinations(range(2 * rho), rho))
        assert abs(exact - (rho + eps)) < 1e-12
        assert abs(val(range(rho)) - (1 + eps)) < 1e-12
    sc = b.make_scene(20000, 0, b.SimConfig(), b.Scenario())
    for budget in [1, 3, 8, 16]:
        chosen = b.select_flows(sc, sc.q_true, sc.imp_true, budget, 'submodular', np.random.default_rng(0))
        assert len(chosen) <= budget
        assert np.bincount(sc.node_id[chosen], minlength=4).max() <= 4
    return dict(noise_units=True, budget_vs_sla_distinction=True,
                convex_solution_matches_scalar_grid=True, event_max_reward=True,
                separation_construction=True, admission_budget_and_node_capacity=True)


def result_checks():
    folder = r.OUT
    manifest = json.loads((folder / 'run_manifest.json').read_text(encoding='utf-8'))
    assert not manifest['smoke'], 'Smoke outputs are not paper evidence'
    raw = pd.read_csv(folder / 'raw_main.csv')
    assert len(raw) == 10 * 10 * 120
    assert raw.seed.nunique() == 10 and raw.method.nunique() == 10
    for filename in ['raw_main.csv', 'raw_robustness.csv', 'raw_recovery.csv', 'raw_scale.csv',
                     'raw_resource_curve.csv', 'raw_overlap.csv', 'raw_oracle.csv']:
        df = pd.read_csv(folder / filename)
        assert np.isfinite(df[['value', 'raw_value', 'reference_value', 'energy_total_j', 'overhead_ms']]).all().all()
        assert df.value.between(-1e-10, 1 + 1e-8).all()
        assert np.allclose(df.value, df.raw_value / df.reference_value)
        assert np.allclose(df.completion + df.sla_violation, 1)
        assert df.max_cpu_ratio.max() <= 1 + 1e-8 and df.max_power_ratio.max() <= 1 + 1e-8
        assert (df.n_completed <= df.n_selected).all()
        assert np.allclose(df.request_completion, df.n_completed / df.n_requests)
        if 'budget' in df: assert (df.n_selected <= df.budget).all()
    paired = raw[raw.method == 'SERVO'].merge(raw[raw.method == 'Equal-Match'],
                                            on=['seed', 'slot'], suffixes=('_s', '_e'))
    assert (paired.selected_s == paired.selected_e).all()
    for field in ['cpu_util', 'power_util', 'reference_value', 'selected_value']:
        assert np.allclose(paired[field + '_s'], paired[field + '_e'], atol=1e-12)
    rec = pd.read_csv(folder / 'raw_recovery.csv')
    assert len(rec) == 4 * 6 * 120
    robust = pd.read_csv(folder / 'raw_robustness.csv')
    assert len(robust) == 7 * 3 * 6 * 120
    # Recompute all summary point estimates and seed-level Student-t intervals.
    wanted = r.summarize(raw).sort_values('method').reset_index(drop=True)
    recorded = pd.read_csv(folder / 'summary.csv').sort_values('method').reset_index(drop=True)
    numeric = wanted.select_dtypes('number').columns
    assert np.allclose(wanted[numeric], recorded[numeric], equal_nan=True)
    return dict(main_rows=len(raw), recovery_rows=len(rec), robustness_rows=len(robust),
                matched_resource_control_exact=True, normalized_reference_common=True,
                summary_reproducible=True, all_hard_budgets_respected=True)


if __name__ == '__main__':
    report = {'scientific': scientific_checks()}
    if '--scientific-only' not in sys.argv: report['results'] = result_checks()
    print(json.dumps(report, indent=2))
    if '--scientific-only' not in sys.argv:
        (r.OUT / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

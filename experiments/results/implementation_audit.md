# Experiment implementation audit

## Superseded full run: forced-capacity allocator

The first full post-revision run completed successfully but exposed an implementation
problem in `action_to_resources`: every allocator normalized its allocation to exactly
100% of each active node's power and CPU budget. Consequently, `cpu_util` and
`power_util` were identically 1.0 and the SAC policy could redistribute resources but
could not trade resource activation against energy. This made the utilization and
energy evidence non-diagnostic.

Representative superseded nominal results were: SERVO value 0.6508, completion
0.7343, energy 3.0286 J/completion; equal allocation value 0.6485, completion 0.7326,
energy 2.9919 J/completion. These values are **not used in the manuscript**.

Corrective action: the bounded SAC action now controls both (i) the within-node
power/compute split and (ii) an activation fraction in [0.90, 1.0], followed by exact
budget normalization. The training reward's energy term was increased from 0.008 to
0.010. We also strengthened the graph-structured crowding effect from
0.025 to 0.055 so that the GNN ablation tests the stated relational signal rather than
a nearly local target. A new forced-training full run replaces all generated result
tables, checkpoints, and figures.

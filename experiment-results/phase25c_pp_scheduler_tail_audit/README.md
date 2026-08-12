# Phase 25C: PP scheduler teacher GPU tail audit

Status: **PASS**. The Phase 25B scheduler-faithful teacher matches all
3/3 measured GPU cells and all
12/12 profile-phase comparisons exactly.

The audit uses two complete fixed-draining windows: a 48-request BurstGPT window
with a 6,216-token prompt and a 930-request Mooncake conversation window with
8,192-token prompts. The diagonal cells PP2/MB1, PP4/MB4, and PP8/MB16 cover
cross-chunk continuation, different lane counts, and small/large microbatch
limits without running an expensive full Cartesian matrix.

For every cell, GPU execution integrity, sender-boundary identity, total calls,
logical bytes, 12-bin calls/bytes, and the exact payload histogram pass. Compact
GPU histograms and logs are retained; model weights, caches, raw profiler traces,
and PID files are excluded.

This supports promoting the Phase 25B formula across the audited BurstGPT and
Mooncake tails. It does not establish online-arrival semantics or replace all
nine combinations on every tail window. The next step is to recompute
H32/H64/H128/Hfull convergence under the scheduler-faithful PP teacher.

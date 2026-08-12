# Phase 25A: full-window fixed-draining structural teacher

Status: **PROVISIONAL_READY_FOR_GPU_AUDIT**. All labels remain provisional until the
full-window GPU sentinel audit passes; they are not GPU ground truth.

## Contract

- Offline teacher input: every capped request length in original order, with
  arrival timestamps removed from scheduling.
- Predictor input remains compact profile + model structure + fixed TP/PP +
  fixed policy + phase.
- Output: exact and 12-bin calls/logical-bytes histograms per 1000 requests.
- PP additionally stores one explicit label per sender boundary.

## Provisional assets

- TP labels: 1,296.
- PP phase labels: 432.
- PP boundary labels: 1,584.
- Full-window requests represented: 18,285.
- Phase 24 Qwen Hfull regression: 864/864
  exact histograms and 864/864 scalar rows match.

`gpu_audit/sentinel_profiles.csv` records deterministic source/tail coverage;
`gpu_audit/plans/` contains TP trace-replay plans and complete PP draining
request lists. GPU calls, bytes, exact histograms, 12-bin conservation, and PP
sender boundaries must pass before promotion to a formal teacher dataset.

Raw traces, model weights, caches, and PIDs are excluded.

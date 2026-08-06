# Phase 14F: op/backend-proxy continuous cost curve

This analysis measures 105 exact `raw_op x payload x TP` supports, convolves the
curve with the Phase 14C PatternDemand histograms, and evaluates held-out workload
and leave-one-model-out prediction. Observed inference backend signatures are not
predictive features; they are retained only for post-hoc audit.

Selected workload-CV result (`phase14f_op_backend_scaled`):

- MAPE: 26.086%
- P95 APE: 67.283%
- Decode MAPE: 35.141%
- all convergence gates passed: False

See `curve_summary.csv`, `predictions.csv`, `metrics.csv`, and `summary.json` for
the complete evidence and statistical scopes.

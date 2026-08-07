# Phase 20: pure-PP PatternDemand predictor (Qwen3-8B)

## Scope

This stage converts the archived Phase-19 profiler output into a prediction task:

`traffic/workload profile + PP execution policy + PP size + model structure -> per-boundary 12-bin message histogram`.

The truth label is the histogram emitted by SGLang's histogram-only PP profiler on a representative forward boundary. All `9` cells passed cross-boundary equality checks, and all `234` phase samples were identical across three repetitions. Pipeline-wide logical demand is derived by multiplying the representative-boundary calls and bytes by `pp_size - 1`.

## Data

- Model: Qwen3-8B (`hidden_size=4096`, BF16, two proxy tensors: `hidden_states` and `residual`).
- Configuration samples: `117`; phase-separated samples: `234`.
- PP sizes: 2/4/8; microbatch caps: 1/4/16; workloads: 13; repeats: 3.
- Labels: exact payload histogram is preserved in the source; the predictor target uses 12 logarithmic payload bins.

## Models and controls

- `total_bytes_oracle`: knows the true calls/bytes but collapses them to one average-size bin. This is an intentionally strong representation-loss control, not a deployable predictor.
- `three_bin_oracle`: knows the true three coarse bins and maps them to 12 bins. This isolates information lost by three hard buckets.
- `structured_h0`: derives forward-message demand from token survival, the 4096-token prefill chunk limit, the microbatch cap, hidden size, dtype and the two PP proxy tensors.
- `direct_dnn`: predicts the 12-bin target directly.
- `h0_dnn_residual`: the DNN only learns the bounded residual in encoded histogram space on top of structured H0.

## Strict holdout headline

Mean errors below are computed only on held-out groups (not random rows):

| holdout | method | calls APE | bytes APE | calls distribution L1 | calls EMD |
|---|---|---:|---:|---:|---:|
| workload | structured H0 | 2.56% | 0.00% | 0.0085 | 0.0004 |
| workload | direct DNN | 119.61% | 190582.91% | 0.8934 | 0.0774 |
| workload | H0 + DNN residual | 7.94% | 1.24% | 0.0110 | 0.0008 |
| strategy | H0 + DNN residual | 5.37% | 0.56% | 0.0094 | 0.0005 |
| PP size | H0 + DNN residual | 5.40% | 0.49% | 0.0094 | 0.0005 |

See `metrics.csv` for phase-specific and P95 values. `holdout_predictions.csv.gz` retains every held-out prediction for auditing.

## Interpretation boundary

This closes the first pure-PP predictor on one model under controlled simultaneous arrivals/draining batches. It does **not** yet establish cross-model PP generalization or real online arrival/burst prediction. Qwen3-8B is the only model here, so the structural model fields are constant. A second PP-capable model and trace-derived arrival windows are required before those claims.

This stage predicts logical demand, not latency. PP P2P `op x payload x topology/backend -> latency` curves remain the next independent input to the communication-time equation.

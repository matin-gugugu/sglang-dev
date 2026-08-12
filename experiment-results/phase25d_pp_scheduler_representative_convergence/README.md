# Phase 25D: scheduler-faithful PP representative convergence

Status: **PASS**. This reruns H32/H64/H128/Hfull and compact32
with the Phase 25B SGLang lane scheduler, replacing the static PP grouping used
by Phase 24. Hfull reproduces all 432/432
Phase 25B teacher rows exactly.

| Sample | calls MAPE | calls WAPE | bytes MAPE | hist TV | norm EMD | cost MAPE | P95 calls APE | all gates |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| H32 | 71.99% | 25.21% | 5.82% | 0.4209 | 0.0511 | 17.13% | 294.37% | FAIL |
| H64 | 33.50% | 12.93% | 2.78% | 0.3318 | 0.0347 | 8.40% | 150.86% | FAIL |
| H128 | 18.65% | 7.93% | 2.01% | 0.2624 | 0.0241 | 6.06% | 59.39% | FAIL |

The experiment covers 24 BurstGPT/Mooncake windows, PP2/4/8, MB1/4/16,
Prefill/Decode/total, and normalization per 1,000 requests. It stores exact
payload histograms, per-case and aggregate metrics, compact32 decomposition,
the old-vs-new teacher comparison, figure, logs, DONE, and manifest.

Phase 25B has GPU evidence from all nine configurations on the 42-request smoke;
Phase 25C adds exact BurstGPT and Mooncake long-prompt tail evidence on the three
diagonal configurations. This validates the teacher contract in those audited
regions. It does not turn H32/H64/H128 into GPU-measured labels or cover online
arrival-aware scheduling.

Complete request lists remain offline label-generation inputs. The next model
still consumes only the compact history profile, model structure, fixed PP
configuration, policy, and phase.

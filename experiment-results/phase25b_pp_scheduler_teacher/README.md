# Phase 25B: scheduler-faithful PP full-window teacher

Status: **PASS** for the scoped fixed-draining scheduler contract. The source-derived
simulator matches all 9/9 saved GPU
smoke cells exactly, including calls, logical bytes, payload histograms, phase
labels, active batch size, and active tokens.

## What changed

The Phase 25A PP formula used static prefill/decode groups. SGLang instead keeps
one running batch per PP loop lane, continues a global chunked request across
lanes, and attempts prefill before filtering finished decode requests. A freed
slot therefore causes one smaller decode forward and is refilled on the next
visit. Page-rounded chunk budget accounting is also required to reproduce the
exact prefill payloads.

## Assets and checks

- Complete windows: 24 profiles and
  18,285 requests, original order, fixed-draining.
- PP phase labels: 432.
- Explicit sender-boundary labels: 1584.
- CPU configurations checked: 216;
  token-mass conservation and request completion pass in all cases.
- GPU smoke: PP2/4/8 x MB1/4/16 on the 42-request BurstGPT sentinel; all 9 cells exact.

## Size of the Phase 25A correction

Treating the new scheduler-faithful label as reference, the old static formula
has overall calls WAPE 121.59%, mean histogram TV
0.5557, normalized log-payload EMD
0.0941, and reference-cost MAPE
44.58%. Logical bytes remain conserved.

- MB1: calls WAPE 31.93%; it is exact on the
  no-cross-chunk smoke, but not generally on full windows with long prompts.
- MB4: calls WAPE 208.36%.
- MB16: calls WAPE 603.21%.

## Scientific boundary

These labels are valid for the recorded fixed-draining contract. The 9/9 exact
result validates all PP-size/microbatch combinations on one heterogeneous full
window, not every traffic distribution. Online arrivals, preemption, radix
cache, mixed chunking, async PP depth, speculative decoding, and other policies
require separate teachers or audits. The final predictor still consumes only
the compact history profile, model structure, fixed PP configuration, policy,
and phase; full request lists are offline label-generation inputs only.

Raw profiler traces, weights, caches, and PID files are not included.

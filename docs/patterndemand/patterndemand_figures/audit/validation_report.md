# Validation report

- Branch: `experiment/pattern-demand-v0.5.15-clean`
- Commit: `ffb413ffec69fd2f87bc958ed73f618696457baa`
- Result: **PASS**

## Checks

- PASS — PNG count is 11
- PASS — SVG count is 11
- PASS — all PNG files are nonempty
- PASS — all SVG files are nonempty
- PASS — accuracy table has 12 aligned cells
- PASS — model composite table has 18 cells
- PASS — sample selection has 18 rows
- PASS — no sample selected using prediction error
- PASS — Phase39 has 12 unique curves
- PASS — Phase51 has 18 unique curves

## Confidence and caveats

- High confidence that figures reproduce the copied frozen tables and curve JSON: all joins and coverage checks passed.
- Histogram examples are illustrative, not best-case claims; they are selected from Hfull shape quantiles without prediction-error access.
- The TP/PP composite in figure 02 is derived from Phase34D fields, while the PD composite is the Phase50 official value. Interpret ratios within each cell, not as a shared absolute metric.
- Physical-curve replica envelopes show measured variability and should not be read as statistical confidence intervals.

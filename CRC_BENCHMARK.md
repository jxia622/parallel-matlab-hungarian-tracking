# CRC tracking benchmark — 18 September 2026

The MATLAB CPU candidate preserved every track and sparse adjacency entry on
the tested inputs. It delivered a modest improvement, not a GPU-scale speedup.

| Input | Original (4 workers) | Optimized (4 workers) | Speedup |
|---|---:|---:|---:|
| 8 frames, 6,143 points | 1.773 s | 1.718 s | 1.032x |
| 340 frames, 292,606 points | 141.481 s | 128.282 s | 1.103x |

The short test reports medians from three timed runs after warmup; its small
difference is within the observed run-to-run variation. The full-block test
is a single pass per method, original first; the 9.33% time reduction is
preliminary, not a repeatability claim. Tracking time excludes file loading,
saving, pool startup, and equality checks. MATLAB and worker startup were
substantial overheads, so keeping a pool alive matters for repeated use.

## Correctness

- 216 serial and 216 parallel comparisons passed in MATLAB R2025a, covering
  both linking methods, gap limits 1/2/3, empty frames, ties, boundaries, and
  row/column cell layouts.
- All real-data timing runs matched the original `tracks`, `adjacency_tracks`,
  and sparse `A` exactly with `isequaln`, including order and NaN positions.
- The assignment and gap-closing code and solver dependencies are unchanged.
- The tested reference is one of two byte-identical SimpleTracker copies found
  alongside the supplied workflow. The three-script folder did not contain
  its own dependency, so the deployed copy still needs confirmation.
- These tests do not establish equivalence of any upstream point-detection or
  localization stage.

## Execution

Pitt CRC smp node smp-n227, MATLAB R2025a, four process workers plus client,
five allocated CPU cores, 24 GB requested RAM, no GPU. Completed job: 24121422.
The earlier job 24121420 completed and saved all 432 correctness tests, then
was cancelled to shorten the initial timing workload. No user jobs were
cancelled. The completed job shut down its MATLAB pool.

Input: one private 340-frame real-data tracking block. It is not included in
the repository and is not a complete performance study across datasets.
Hashes, raw timings, MATLAB reports, and logs are in crc_results.

The remote validation package was run from a private CRC work directory.

## Interpretation and next step

The existing tracker already parallelizes adjacent-frame assignment. The
candidate saves track-building work but does not change the costly Hungarian
solver. Profile assignment versus graph reconstruction on representative
blocks before choosing the next implementation. A Python/CUDA port is possible;
its solver must preserve distance arithmetic, threshold decisions, tie-breaking,
and point/track order to satisfy the same-results requirement. Replacing it
with a different optimal-assignment library does not automatically guarantee
identical tracks when multiple assignments have the same cost.

The batch runner's block-level parallel mode has not been benchmarked or
runtime-validated in this run; the measured mode is parallel frame pairs.
SVD, localization, Kalman filtering, and map rendering are outside these timings.

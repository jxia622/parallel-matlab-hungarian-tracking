# GPU tracking benchmark — 18 September 2026

CUDA v2 tracked the 340-frame example in **50.32 seconds**, versus
141.48 seconds for the original MATLAB tracker: **2.81x faster**,
or **64.4% less tracking time**. All tested outputs matched exactly.

| Implementation | Hardware | Full-block time |
|---|---|---:|
| Original MATLAB | 4 CPU workers | 141.481 s |
| Optimized MATLAB | 4 CPU workers | 128.282 s |
| CUDA v2 | A100-SXM4-80GB | 50.320 s |

The final CUDA timing repeats were 50.315 and 50.324 seconds.
An earlier run took 49.996 seconds. The standalone file runner took 49.398
seconds of tracking time. Relative to the optimized CPU version, CUDA v2 is
2.55x faster. CPU figures are previous single-run measurements on
CRC smp-n227; GPU measurements used CRC gpu-n31. This is a measured comparison
on one real-data block, not a hardware-normalized or full-dataset speed claim.

## What is timed

CUDA timing includes CPU input packing, host-to-device transfers, distance
calculation, the complete assignment solver on the GPU, device-to-host result
transfer, and CPU reconstruction of both track outputs. It excludes disk I/O,
reference loading/equality checks, and CUDA compilation/loading. CPU timing
likewise excluded disk I/O, pool startup, and equality checks.

CUDA compilation and extension loading took about 55 seconds the first time;
cached extension loading took 1.2 seconds. CPU reconstruction took about
0.23 seconds. No float32 conversion or alternative assignment library is used.

## Exact comparisons

The input block contains 292,606 points, 102,026 tracks, and 190,580 links.
Every full-block run matched MATLAB's sparse adjacency, frame-relative tracks
(including NaNs), global-index tracks, and ordering exactly. There were zero
adjacency differences. CUDA also passed 106 linker edge cases at batch sizes
1 and 32 (212 comparisons per validation run), including empty frames, ties,
duplicates, and distance boundaries. Local graph/serialization checks passed
102 independent cases.

The output MAT file was reloaded locally and matched all values, cell shapes,
and MATLAB numeric classes. Direct MATLAB `isequaln` verification also passed for SR_Localizations, tracks,
adjacency_tracks, and A (job 24125249, successful exit).
The initial export checker incorrectly compared SciPy's raw storage dtypes:
MATLAB can store a double-valued cell using compact integer storage. Reading
with `mat_dtype=True` checks MATLAB's declared type and fixes this test issue;
no tracking values or exported numeric types were changed.

## How to run

Code is in `gpu/`. The CRC copy is:
`/ihome/kkim/xiac/bal-tracking-validation-20260918/tracking_acceleration`

From that directory, supply absolute paths:

```bash
sbatch gpu/track_file.sbatch /absolute/input_localizations.mat /absolute/new_tracks.mat
```

Add `--reference /absolute/matlab_reference.mat` to require exact equality
before saving. Input must contain SR_Localizations. Output contains the same
localizations, MATLAB-compatible tracks and adjacency_tracks cells, sparse A,
and timing/provenance metadata. Existing outputs are never overwritten.

## Scope and limits

- Supports the supplied settings: 2D finite double localizations, Hungarian
  linking, distance 8, and MaxGapClosing=1 (which disables gap closing in this
  SimpleTracker version). The Python API accepts other finite distance limits,
  but these have not been validated here. Other gap settings are rejected.
- Preserves the original block boundaries and input point order. Do not split
  a block into independent temporal chunks to speed it up.
- The input file format is MATLAB v7, not v7.3/HDF5.
- The tracker starts from the exact same saved localizations as MATLAB. It
  does not yet integrate with or validate the separate CUDA localization
  pipeline, whose float32 coordinates may differ from MATLAB's input.
- The tested reference is the supplied drive's Jack/Zahra SimpleTracker
  dependency; Bal's three-script folder did not include its own dependency.
- The GPU reduction used to set the Munkres bigM penalty can sum in a different
  order from MATLAB. The validated block/edge cases match exactly; inputs
  extremely close to a power-of-ten penalty boundary need additional checks.
  Exact identity on all possible inputs is not claimed.

## Development record

The first CUDA prototype was correct but slower (227–405 seconds on an
A100-SXM4-40GB, depending on batch size). Its source is preserved in
`gpu/archive_v1`. V2 combines repeated matrix scans, skips covered rows and
columns, preserves the original zero-queue ordering, and uses larger batches.

Oracle job: 24121482. First V2 job: 3979947 (passed). Final GPU timing/export
job: 3980164 (all tracking comparisons passed; original dtype checker then
failed as explained above). Native MATLAB export check: 24125249 (passed, successful exit).
Raw logs, hashes, JSON reports, and the verified candidate MAT file are in
`crc_results`. Originals and the localization project remain unchanged.

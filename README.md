# Parallel MATLAB-compatible Hungarian tracking

This repository parallelizes frame-to-frame Hungarian point tracking while
preserving the output conventions and decision order of a reference MATLAB
SimpleTracker implementation. It provides a conservative MATLAB CPU path and
a CUDA implementation for finite double-precision 2D point sequences.

The file interface accepts any MATLAB cell-array variable containing one
N-by-2 point array per frame. `SR_Localizations` remains the default for
backward compatibility with the research pipeline that motivated the work.

On the validated 340-frame block (292,606 points), CUDA v2 completed
tracking in 50.32 seconds on an A100-SXM4-80GB. The original four-worker MATLAB
tracker took 141.48 seconds: a 2.81x speedup. All 102,026 tracks and 190,580
links matched MATLAB exactly, including order, NaN positions, sparse adjacency,
cell shapes, and MATLAB numeric classes.

| Implementation | Full-block time | Relative to original |
|---|---:|---:|
| Original MATLAB, 4 CPU workers | 141.481 s | 1.00x |
| Optimized MATLAB, 4 CPU workers | 128.282 s | 1.10x |
| CUDA v2, A100-SXM4-80GB | 50.320 s | 2.81x |

Detailed scope and raw methodology are in [GPU_BENCHMARK.md](GPU_BENCHMARK.md)
and [CRC_BENCHMARK.md](CRC_BENCHMARK.md).

## Repository layout

- `gpu/`: CUDA Munkres solver, Python tracker, validation, and Slurm jobs.
- `simpletracker_fast.m`: conservative MATLAB CPU optimization.
- `tests/`: MATLAB equivalence tests and local structural checks.
- `baseline/`: private research snapshot used as the behavioral reference.
- `crc_results/*.json`: compact benchmark and validation records. Research
  data, generated MAT files, compiled extensions, and raw logs are excluded.

## Run the GPU tracker on a Slurm cluster

The input must be a MATLAB v7 file containing one cell array with a finite
double-precision N-by-2 coordinate array per frame. For the default variable
name, submit one complete temporal sequence from the repository root:

```bash
sbatch gpu/track_file.sbatch \
  /absolute/path/input_points.mat \
  /absolute/path/new_tracks.mat
```

The included job file is cluster-neutral. If the CUDA-enabled PyTorch
environment is not already active for batch jobs, set its module and virtual
environment before submission:

```bash
export TRACKING_PYTORCH_MODULE=python/pytorch_251_311_cu124
export TRACKING_VENV=/absolute/path/to/cuda-venv
sbatch gpu/track_file.sbatch /absolute/input.mat /absolute/output.mat
```

Cluster-specific scheduling options can be supplied to `sbatch`. For example,
the validated Pitt CRC configuration used:

```bash
sbatch --clusters=gpu --partition=a100_nvlink --account=kkim \
  gpu/track_file.sbatch /absolute/input.mat /absolute/output.mat
```

To compare against a MATLAB oracle before saving:

```bash
sbatch gpu/track_file.sbatch \
  /absolute/input.mat /absolute/new_output.mat \
  --reference /absolute/matlab_reference.mat
```

For another MATLAB variable name, add `--points-variable`:

```bash
sbatch gpu/track_file.sbatch input.mat output.mat \
  --points-variable detections
```

The distance cutoff is also configurable:

```bash
sbatch gpu/track_file.sbatch input.mat output.mat \
  --points-variable detections --max-linking-distance 12
```

The output preserves the input point variable name and adds `tracks`,
`adjacency_tracks`, sparse `A`, and JSON provenance/timing metadata. Existing
output files are never overwritten. The first CUDA extension build takes
roughly 55 seconds; cached
loading took about 1.2 seconds in validation.

## Supported behavior

The validated production configuration is:

- finite double-precision 2D coordinates;
- Hungarian linking with maximum distance 8;
- `MaxGapClosing = 1`, which performs no gap-closing iterations in this
  SimpleTracker implementation;
- complete original temporal blocks with unchanged point order.

Other gap settings are rejected. The Python API accepts other finite distance
limits, but they have not been validated here. MATLAB v7.3/HDF5 inputs are not
currently supported. Do not divide a temporal block into independent pieces,
because that can break tracks at the boundaries.

## Python environment

The tested CRC environment used Python 3.11, PyTorch 2.5.1 with CUDA 12.4,
NumPy, SciPy, and Ninja. A minimal dependency list is provided in
`requirements.txt`; install a CUDA-enabled PyTorch build appropriate for the
target cluster.

## Validate

`gpu/export_oracle.sbatch` exports MATLAB reference outputs. After it succeeds,
`gpu/run_gpu.sbatch` builds the CUDA extension, checks 106 edge cases, runs
repeated full-block comparisons, exercises the file runner, and reloads the
saved MAT file. `gpu/verify_export.sbatch` performs the final native MATLAB
`isequaln` check.

For the CPU fallback, MATLAB R2025a with Parallel Computing Toolbox passed all
432 serial/parallel equivalence comparisons:

```matlab
addpath(pwd, fullfile(pwd,'tests'), fullfile(pwd,'baseline'))
test_tracking_equivalence(false)
parpool('local',4)
test_tracking_equivalence(true)
```

## Reproducibility and limits

The current result is an exact match on the supplied real block and the tested
edge cases, not a proof for every possible floating-point input. The GPU
reduction used for the Munkres penalty can sum finite costs in a different
order from MATLAB; inputs extremely close to a power-of-ten penalty boundary
need additional validation.

The tracker begins with the exact same saved points as MATLAB. It does not
establish equivalence with upstream detectors that produce different coordinate
precision. Validate that boundary independently before combining stages.

The original research files remain unchanged. See [NOTICE.md](NOTICE.md) before
changing this repository's visibility or redistributing the baseline sources.

## Reference implementation findings

The supplied workflow processes blocks serially with linking distance 8 and
`MaxGapClosing = 1`. Its SimpleTracker dependency was stored separately from
the three pipeline scripts. Two available dependency copies matched byte for
byte; the reference snapshot and SHA-256 hashes are in `baseline/`.

That dependency already uses parfor for adjacent-frame assignment and track
extraction. Each track allocates a buffer sized to *all points in the block*,
and rebuilding frame-relative IDs scans frame counts repeatedly. These are
source-level performance opportunities, not measured bottlenecks yet.

simpletracker_fast retains the original Hungarian/nearest-neighbor solvers,
floating-point distance calculations, ordered gap closing, and link assembly.
It replaces graph traversal and frame lookup with smaller buffers and cached
indices. Roots and points retain their original order. MaxGapClosing 1 still
executes no gap-closing iterations in this implementation. No parameter or
solver changes are made to improve speed.

run_tracking_batch offers parallel blocks or parallel frame pairs, with one
active parallel level. Use complete existing blocks: dividing a block into
independent temporal chunks could break tracks at boundaries. Parallel blocks
may use substantial RAM; choose worker count after measuring one whole block.

## MATLAB CPU batch runner

Requires MATLAB plus Parallel Computing Toolbox. Run it on a compute node,
not a shared login node.

After transferring this folder and a representative MAT file containing a
point-cell variable to the compute node:

```matlab
cd('/path/to/tracking_acceleration')
addpath(pwd, fullfile(pwd,'tests'), fullfile(pwd,'baseline'))
test_tracking_equivalence(false)
parpool('local',4) % adjust to allocated CPUs and RAM
test_tracking_equivalence(true)
report = benchmark_tracking('/path/to/block1.mat',4,3,true,true, ...
    'detections',12,2);
save('benchmark_report.mat','report')
```

Tests compare every track, adjacency-track cell, and sparse adjacency entry
with isequaln; they cover empty frames, duplicates/ties, distance boundaries,
row/column cell arrays, both solvers, and gap limits 1, 2, and 3. The serial
reference differs from the snapshot only in function name and parfor→for.
Benchmarking compares the candidate against the *already-parallel original*,
separates pool startup, and checks exact equality on every run. By default it
warms all methods, rotates run order, and reports median times. Optional
arguments can omit serial timing and warmup for a single-pass full-block check. Loading and saving are excluded from those kernel timings.
Also measure full batch wall time for production throughput.

```matlab
files = {'/path/to/block1.mat','/path/to/block2.mat'};
t = tic;
report = run_tracking_batch(files,'/path/to/new_verified_output', ...
    'blocks',true,'detections',12,2);
wallSeconds = toc(t);
```

The batch requires a new output directory. It verifies each block before
saving. Verification reruns the original serial algorithm and is deliberately
expensive; use false only after validation on representative data. Saved files
include the named point variable, tracks, adjacency_tracks, and
timing/provenance info.
Input order determines output numbering; supply files in acquisition order.
Both parameters are configurable; distance 8 and gap limit 1 remain the
defaults that match the validated reference workflow.

## Validation record

The supplied tracking script also contains an extra `]` after `clear
SR_Localizations`; the snapshot is retained as supplied, not executed. The
batch runner replaces its machine-specific paths with explicit inputs.

MATLAB R2025a on Pitt CRC passed all 432 serial/parallel synthetic comparisons
against the original solver and serial reference. The 8-frame, 6,143-point
real-data excerpt also matched exactly on every benchmark run. Warm median
times with four workers were 1.773 s original and 1.718 s optimized; this 3.2%
difference is too small to establish a robust speed gain from three repetitions.
The full 340-frame block (292,606 points) matched exactly: original 141.481 s,
optimized 128.282 s, a 1.103x speedup (9.33% less time). This was one run per
method, original first, and needs repeated measurements before generalizing.
See crc_results for machine-readable reports and CRC_BENCHMARK.md for details.
local_validation.json describes separate Python/static checks only.

The submitted job uses five allocated CPU cores (four MATLAB workers plus the
client), 24 GB RAM, and no GPU. Pool startup was measured separately. The
first job was stopped after saving all 432 test results to shorten the timing
workload; the second job reuses those results and runs the revised benchmark.

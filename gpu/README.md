# CUDA MATLAB-compatible Hungarian tracker

The prototype translates the supplied Yi Cao Munkres solver to CUDA, retaining
its initial column-major greedy stars, ordered zero queue, ordered augmenting
paths, subtraction grouping, finite-distance mask, and dummy costs. Independent
frame pairs run in separate GPU thread blocks. Matrix scans run cooperatively
within each block. Track reconstruction runs in NumPy on the CPU.

The supported contract is finite float64 2D point sequences, Hungarian linking,
a finite nonnegative distance limit (8 in the reference test), and
MaxGapClosing=1. In this MATLAB version, gap limit 1 disables gap closing.
Other gap settings are rejected. Source-frame point order is never changed.
GPU source/target assignment is zero-based internally; saved tracks use MATLAB
one-based indices, original root order, and NaN for absent frames.

The kernel uses double precision and disables fused multiply-add contraction
with `--fmad=false`. It does not use a replacement assignment library. Exact
compatibility must be assessed by the exported MATLAB oracles; algorithmic
similarity alone is not a guarantee. In particular, the finite-cost sum used
to set bigM uses a GPU reduction, whose addition order can differ from MATLAB.
Boundary cases near powers of ten need explicit coverage before broader use.

## Build and validate on CRC

The validation job uses a CUDA-enabled PyTorch environment, an A100 GPU, and
the PyTorch CUDA extension toolchain.

From the remote tracking_acceleration directory:

```bash
sbatch gpu/export_oracle.sbatch
# After the CPU oracle job succeeds:
sbatch gpu/run_gpu.sbatch
```

Compile/context startup is recorded separately. Recorded tracking times include
CPU input packing, GPU transfers, the full CUDA assignment solver, result
transfer, and CPU construction of both track outputs. Disk I/O, oracle loading,
and equality checks are excluded, as in the earlier MATLAB benchmark.
The first prototype tested three batch sizes (32, 64, 128), each twice.
The revised kernel runs all 339 frame pairs together. Full-block checks took
49.996, 50.315, and 50.324 seconds and matched every MATLAB output exactly.
The reusable runner defaults to 384 pairs, reduced automatically when GPU
workspace memory would exceed its budget.

## Input precision compatibility

The benchmark starts from exactly the same double-precision point arrays as
MATLAB. An upstream pipeline producing float32 coordinates may have
already changed those coordinates; converting them to float64 does not undo
that difference. Validate that boundary separately before claiming end-to-end
identity for an upstream point detector plus this tracking pipeline.

## Implementation references

- Original MATLAB code: ../baseline/munkres.m (Yi Cao, 2011) and
  ../baseline/hungarianlinker.m (Jean-Yves Tinevez).
- PyTorch extension build API: https://docs.pytorch.org/docs/main/cpp_extension.html
- CUDA compiler floating-point flags:
  https://docs.nvidia.com/cuda/archive/12.5.0/cuda-compiler-driver-nvcc/index.html

## Run on a MATLAB point-cell file

```bash
python gpu/run_gpu_tracking.py input_points.mat output_tracks.mat \
  --batch-size 384 --max-linking-distance 8 \
  --reference crc_results/gpu_oracle.mat
```

The default input variable is `SR_Localizations`. Use
`--points-variable detections` (or another MATLAB variable name) for a general
point-cell input. The distance cutoff is configurable with
`--max-linking-distance`. The reference argument is optional and must refer to
the same input frames and parameters.
It requires exact equality before writing. Output preserves the named point
cell array and adds `tracks`, `adjacency_tracks`, sparse `A`, and provenance JSON.
Existing output files are never overwritten. Saving happens after timing.
Only MATLAB v7 (not v7.3/HDF5) input files are currently supported.

## Validation status

CUDA v2 passed 106 linker cases at each of two batch sizes and repeated full
340-frame comparisons. All 102,026 tracks and 190,580 adjacency links matched
MATLAB exactly. The runner's saved MAT file also passed native MATLAB isequaln checks for all
four output variables; cell shapes and MATLAB numeric types match.
See ../GPU_BENCHMARK.md for measurements and limits.

CUDA v1 was correct but slower than MATLAB; it is retained only in archive_v1.
Use the current gpu_tracker.py and munkres_cuda.cu together.

## Submit a production input file

From the repository root on a Slurm GPU cluster:

```bash
sbatch gpu/track_file.sbatch /absolute/input.mat /absolute/new_output.mat
```

Add `--reference /absolute/matlab_reference.mat` to enforce exact equality
before saving. Set `TRACKING_PYTORCH_MODULE` and `TRACKING_VENV` when the batch
environment does not already provide CUDA-enabled PyTorch. Pass the target
cluster's account and partition options to `sbatch`. Logs go under
`crc_results`. Keep each temporal sequence intact; do not split it at
arbitrary frame boundaries.

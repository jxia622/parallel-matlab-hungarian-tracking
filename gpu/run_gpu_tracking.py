"""Run the CUDA candidate on an existing MATLAB SR_Localizations file.

Example (on an allocated GPU node):
  python gpu/run_gpu_tracking.py INPUT.mat OUTPUT.mat --reference MATLAB.mat
No existing output is overwritten. MATLAB v7.3 inputs must be exported as v7.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy import sparse
from scipy.io import loadmat, savemat


def to_cells(vectors):
    cells = np.empty((len(vectors), 1), dtype=object)
    for i, vector in enumerate(vectors):
        cells[i, 0] = np.asarray(vector, dtype=np.float64).reshape(-1, 1)
    return cells


def validate_result(result, points, reference):
    expected_points = reference['SR_Localizations'].ravel(order='F')
    if len(expected_points) != len(points) or not all(
            np.array_equal(a, b) for a, b in zip(points, expected_points)):
        raise ValueError('Reference localizations differ from the input')
    expected_tracks = reference['tracks'].ravel(order='F')
    expected_adj = reference['adjacency_tracks'].ravel(order='F')
    if len(expected_tracks) != len(result['tracks']) or not all(
            np.array_equal(a, b.ravel(), equal_nan=True)
            for a, b in zip(result['tracks'], expected_tracks)):
        raise AssertionError('Frame-relative tracks differ from MATLAB')
    if len(expected_adj) != len(result['adjacency_tracks']) or not all(
            np.array_equal(a, b.ravel())
            for a, b in zip(result['adjacency_tracks'], expected_adj)):
        raise AssertionError('Global-index tracks differ from MATLAB')
    if 'A' in reference and (result['A'] != reference['A']).nnz:
        raise AssertionError('Sparse adjacency differs from MATLAB')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference', type=Path, help='MATLAB oracle containing both track outputs')
    parser.add_argument('--batch-size', type=int, default=384)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    data = loadmat(args.input, variable_names=['SR_Localizations'], mat_dtype=True)
    if 'SR_Localizations' not in data or data['SR_Localizations'].dtype != object:
        raise ValueError('Input must contain an SR_Localizations cell array')
    points = list(data['SR_Localizations'].ravel(order='F'))
    if not points:
        raise ValueError('At least one frame is required')
    for point in points:
        if point.ndim != 2 or point.shape[1] != 2 or point.dtype != np.float64 or not np.isfinite(point).all():
            raise ValueError('Each frame must contain finite double-precision N-by-2 coordinates')

    # Deferred import makes --help and export utilities usable without CUDA.
    import torch
    from gpu_tracker import build, track
    start = time.perf_counter()
    build()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = track(points, batch_size=args.batch_size)
    torch.cuda.synchronize()
    tracking_seconds = time.perf_counter() - start
    total = sum(map(len, points))
    result['A'] = sparse.csc_matrix((np.ones(len(result['source_ids'])),
        (result['source_ids'], result['target_ids'])), shape=(total, total))
    if args.reference:
        validate_result(result, points, loadmat(args.reference, mat_dtype=True))
    info = {
        'input': str(args.input.resolve()),
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
        'batch_size': args.batch_size, 'max_linking_distance': 8, 'max_gap_closing': 1,
        'frames': len(points), 'points': total,
        'setup_seconds': setup_seconds, 'tracking_seconds': tracking_seconds,
        'link_seconds': result['link_seconds'], 'assembly_seconds': result['assembly_seconds'],
        'verified_against_matlab': bool(args.reference),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'SR_Localizations': data['SR_Localizations'],
        'tracks': to_cells(result['tracks']),
        'adjacency_tracks': to_cells(result['adjacency_tracks']),
        'A': result['A'], 'gpu_tracking_info_json': json.dumps(info),
    }
    # Write beside the final destination, then publish with an exclusive link.
    # This is atomic and will also refuse a file created after the initial check.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix='.mat', delete=False) as tmp:
            temporary = Path(tmp.name)
        savemat(temporary, payload, do_compression=True, appendmat=False)
        os.link(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps(info, indent=2))
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()

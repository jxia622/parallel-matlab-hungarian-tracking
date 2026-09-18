"""Check the actual NumPy graph function without importing CUDA/PyTorch.
This validates assembly/serialization, not the CUDA assignment solver.
"""
import ast
from pathlib import Path
import tempfile
import numpy as np
from scipy.io import loadmat, savemat
from scipy import sparse
from run_gpu_tracking import to_cells, validate_result

root = Path(__file__).resolve().parent
module = ast.parse((root/'gpu_tracker.py').read_text())
function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'assemble')
namespace = {'np': np}
exec(compile(ast.Module(body=[function], type_ignores=[]), 'gpu_tracker.py:assemble', 'exec'), namespace)
assemble = namespace['assemble']
rng = np.random.default_rng(20260918)
for trial in range(102):
    counts = rng.integers(0, 12, size=rng.integers(1, 30))
    if trial == 0: counts = np.zeros(3, dtype=int)
    if trial == 1: counts = np.array([4])
    points = [rng.normal(size=(count, 2)) for count in counts]
    links = []
    offset = np.r_[0, np.cumsum(counts)]
    next_point = {}
    incoming = set()
    for frame in range(len(counts)-1):
        indices = np.full(counts[frame], -1, dtype=int)
        number = int(rng.integers(0, min(counts[frame:frame+2])+1))
        sources = rng.permutation(counts[frame])[:number]
        targets = rng.permutation(counts[frame+1])[:number]
        indices[sources] = targets
        links.append(indices)
        for source, target in zip(sources, targets):
            a, b = int(offset[frame]+source), int(offset[frame+1]+target)
            next_point[a] = b
            incoming.add(b)
    result = assemble(points, links)
    ref_adj = []
    ref_tracks = []
    for point in range(sum(counts)):
        if point in incoming: continue
        chain = []
        track = np.full(len(counts), np.nan)
        while True:
            chain.append(point+1)
            frame = int(np.searchsorted(offset[1:], point, side='right'))
            track[frame] = point-offset[frame]+1
            if point not in next_point: break
            point = next_point[point]
        ref_adj.append(np.array(chain))
        ref_tracks.append(track)
    assert len(result['adjacency_tracks']) == len(ref_adj)
    assert all(np.array_equal(a,b) for a,b in zip(result['adjacency_tracks'],ref_adj))
    assert np.array_equal(result['tracks'],np.array(ref_tracks).reshape(len(ref_tracks),len(counts)),equal_nan=True)
    result['A'] = sparse.csc_matrix((np.ones(len(next_point)), (list(next_point), list(next_point.values()))),shape=(sum(counts),sum(counts)))
    cells = np.empty((1,len(points)),dtype=object)
    for i, point in enumerate(points): cells[0,i] = point
    payload = {'SR_Localizations':cells,'tracks':to_cells(ref_tracks),'adjacency_tracks':to_cells(ref_adj),'A':result['A']}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'roundtrip.mat'
        savemat(path,payload)
        validate_result(result,points,loadmat(path))
print('Passed 102 independent graph and MATLAB serialization cases.')

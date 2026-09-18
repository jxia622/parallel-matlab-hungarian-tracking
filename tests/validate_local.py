"""Limited structural checks; does not execute MATLAB or assignment solvers."""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
from tree_sitter import Language, Parser
import tree_sitter_matlab

root = Path(__file__).resolve().parents[1]
parser = Parser(Language(tree_sitter_matlab.language()))
files = list(root.glob('*.m')) + list((root / 'tests').glob('*.m'))
for path in files:
    assert not parser.parse(path.read_bytes()).root_node.has_error, path
original = (root/'baseline/simpletracker.m').read_text()
fast = (root/'simpletracker_fast.m').read_text()
start, end = '    %% Frame to frame linking', '    %% Parse adjacency matrix to build tracks'
expected = original[original.index(start):original.index(end)].replace(
    'parfor i = 1 : n_slices-1', 'parfor (i = 1 : n_slices-1, worker_limit)')
assert expected == fast[fast.index(start):fast.index(end)]
serial = (root/'tests/simpletracker_serial_reference.m').read_text()
assert serial == original.replace('= simpletracker(points, varargin)',
    '= simpletracker_serial_reference(points, varargin)',1).replace('    parfor i =','    for i =')
fixture = Path('/Volumes/kim lab/MB SRU Example Mice Kidney/EPCR1_2/bloc1_track.mat')
d = loadmat(fixture)
points = d['SR_Localizations'].ravel()
tracks = [t.ravel().astype(int) for t in d['adjacency_tracks'].ravel()]
counts = np.array([len(p) for p in points])
n = counts.sum()
successor = np.zeros(n+1,dtype=int)
incoming = np.zeros(n+1,dtype=bool)
for track in tracks:
    successor[track[:-1]] = track[1:]
    incoming[track[1:]] = True
roots = np.flatnonzero(~incoming[1:])+1
assert len(roots) == len(tracks)
frame_ids = np.repeat(np.arange(1,len(counts)+1),counts)
offsets = np.r_[0,np.cumsum(counts)]
local_ids = np.arange(1,n+1) - np.repeat(offsets[:-1],counts)
for root_id, expected_track in zip(roots,tracks):
    actual = []
    target = root_id
    while target:
        actual.append(target)
        target = successor[target]
    assert np.array_equal(actual,expected_track)
    assert len(actual) <= len(counts)
    # Independent old subtract-frame-count loop for every saved point.
    for global_id in actual:
        tmp, frame = int(global_id), 0
        while tmp > 0:
            tmp -= int(counts[frame])
            frame += 1
        assert frame_ids[global_id-1] == frame
        assert local_ids[global_id-1] == tmp+counts[frame-1]
report = dict(matlabExecuted=False, syntaxFilesChecked=len(files),
    assignmentAndGapCodeUnchanged=True, serialReferenceOnlyRenamesAndChangesLoops=True,
    fixture=str(fixture), frames=len(points), points=int(n), savedTracks=len(tracks),
    pythonGraphReconstructionExact=True,
    limitation='Python graph check uses saved tracks to reconstruct edges; it does not validate MATLAB assignment, parfor execution, or runtime speedup.')
(root/'local_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

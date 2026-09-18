"""Read the written MAT file back and require exact cell shapes and values."""
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat

path=Path(sys.argv[1]);root=Path(__file__).resolve().parents[1]
actual=loadmat(path,mat_dtype=True);expected=loadmat(root/'crc_results/gpu_oracle.mat',mat_dtype=True)
for name in ['SR_Localizations','tracks','adjacency_tracks']:
    assert actual[name].shape==expected[name].shape,(name,'cell array shape')
    for a,b in zip(actual[name].ravel(order='F'),expected[name].ravel(order='F')):
        assert a.shape==b.shape,(name,'cell content shape')
        assert a.dtype==b.dtype,(name,'dtype')
        assert np.array_equal(a,b,equal_nan=True),(name,'values')
assert (actual['A']!=expected['A']).nnz==0
print('MAT_EXPORT_ROUNDTRIP_EXACT',path,flush=True)

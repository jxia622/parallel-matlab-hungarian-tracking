from pathlib import Path
import json,time,platform
import numpy as np
import torch
from scipy.io import loadmat
from scipy import sparse
from gpu_tracker import build,link_pairs,track

root=Path(__file__).resolve().parents[1];out=root/'crc_results'
print('GPU',torch.cuda.get_device_name(),flush=True)
t=time.perf_counter();build();torch.cuda.synchronize();compile_seconds=time.perf_counter()-t
print('BUILD_READY',compile_seconds,flush=True)
e=loadmat(out/'gpu_edge_oracle.mat')
pairs=[tuple(np.asarray(x,dtype=np.float64) for x in cell.ravel()) for cell in e['fixtures'].ravel()]
expected=[x.ravel().astype(np.int64) for x in e['expected'].ravel()]
for batch_size in [1,32]:
    actual=link_pairs(pairs,batch_size=batch_size)
    for i,(a,b) in enumerate(zip(actual,expected)):
        # MATLAB uses 1-based target IDs, -1 for unmatched; CUDA uses 0-based.
        a=np.where(a>=0,a+1,-1)
        if not np.array_equal(a,b):
            print('EDGE_MISMATCH',i,'actual',a,'expected',b,flush=True)
            raise AssertionError(f'Edge case {i} mismatch')
print('EDGE_TESTS_PASSED',len(pairs),flush=True)
d=loadmat(out/'gpu_oracle.mat')
points=[np.asarray(x,dtype=np.float64) for x in d['SR_Localizations'].ravel()]
expected_adj=[x.ravel().astype(np.int64) for x in d['adjacency_tracks'].ravel()]
expected_tracks=np.stack([x.ravel() for x in d['tracks'].ravel()])
expected_A=d['A'].tocsr()
# Warm up without including compilation/context startup in the recorded run.
track(points[:8],batch_size=8)
report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'python':platform.python_version(),
    'compile_and_load_seconds':compile_seconds,'edge_comparisons':len(pairs)*2,
    'frames':len(points),'points':sum(map(len,points)),'runs':[]}
for batch_size in [384]:
    for repeat in range(2):
        torch.cuda.synchronize();t=time.perf_counter()
        actual=track(points,batch_size=batch_size)
        torch.cuda.synchronize();elapsed=time.perf_counter()-t
        A=sparse.csr_matrix((np.ones(len(actual['source_ids'])),(actual['source_ids'],actual['target_ids'])),shape=expected_A.shape)
        differences=(A!=expected_A).nnz
        exact_tracks=np.array_equal(actual['tracks'],expected_tracks,equal_nan=True)
        exact_adj=len(actual['adjacency_tracks'])==len(expected_adj) and all(np.array_equal(a,b) for a,b in zip(actual['adjacency_tracks'],expected_adj))
        row={'batch_size':batch_size,'repeat':repeat,'seconds':elapsed,'link_seconds':actual['link_seconds'],
            'assembly_seconds':actual['assembly_seconds'],'adjacency_differences':int(differences),
            'tracks_exact':bool(exact_tracks),'adjacency_tracks_exact':bool(exact_adj)}
        report['runs'].append(row)
        print(json.dumps(row),flush=True)
        (out/'gpu_benchmark_final.json').write_text(json.dumps(report,indent=2)+'\n')
        if differences or not exact_tracks or not exact_adj:
            sparse.save_npz(out/'gpu_candidate_A.npz',A)
            raise AssertionError('Full tracking output mismatch')
print('GPU_VALIDATION_PASSED',flush=True)

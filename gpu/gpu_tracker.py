"""Experimental exact-order CUDA tracker for the supplied 2D MATLAB workflow.
Only MaxGapClosing=1 is supported: that value disables gap closing in baseline.
No silent approximation, precision conversion, or fallback to a different solver.
"""
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load

_EXTENSION = None

def build():
    global _EXTENSION
    if _EXTENSION is None:
        root=Path(__file__).resolve().parent
        _EXTENSION=load(name='bal_munkres_cuda_v2',sources=[str(root/'binding.cpp'),str(root/'munkres_cuda.cu')],
            extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false','-lineinfo'],verbose=True)
    return _EXTENSION

def link_pairs(pairs, max_distance=8., batch_size=384):
    if not isinstance(batch_size,int) or batch_size<1:
        raise ValueError('batch_size must be a positive integer')
    if not np.isfinite(max_distance) or max_distance<0:
        raise ValueError('finite nonnegative distance required')
    prepared=[]
    for a,b in pairs:
        a=np.asarray(a); b=np.asarray(b)
        for x in (a,b):
            if x.ndim!=2 or x.shape[1]!=2 or x.dtype!=np.float64 or not np.isfinite(x).all():
                raise ValueError('Each frame must be a finite float64 array of shape [N,2]')
        prepared.append((a,b))
    extension=build()
    result=[]
    if prepared:
        max_points=max(1,max(max(len(a),len(b)) for a,b in prepared))
        if max_points>46340:
            raise ValueError('This kernel supports at most 46,340 points per frame')
        free,_=torch.cuda.mem_get_info()
        reusable=free+torch.cuda.memory_reserved()-torch.cuda.memory_allocated()
        # Main workspace: dense double costs; also potentials, indexes and input.
        bytes_per_pair=8*max_points*max_points+128*max_points+4096
        memory_batch=int(0.65*reusable)//bytes_per_pair
        if memory_batch<1:
            raise MemoryError('One frame pair exceeds the GPU memory budget')
        batch_size=min(batch_size,memory_batch)
    for begin in range(0,len(prepared),batch_size):
        batch=prepared[begin:begin+batch_size]
        cap=max(1,max(max(len(a),len(b)) for a,b in batch))
        sources=np.zeros((len(batch),cap,2),dtype=np.float64)
        targets=np.zeros_like(sources)
        counts=np.zeros((len(batch),2),dtype=np.int32)
        for i,(a,b) in enumerate(batch):
            sources[i,:len(a)]=a;targets[i,:len(b)]=b;counts[i]=len(a),len(b)
        out,status=extension.solve(torch.from_numpy(sources).cuda(),torch.from_numpy(targets).cuda(),torch.from_numpy(counts).cuda(),float(max_distance))
        failures=status.cpu().numpy()
        if failures.any():
            raise RuntimeError(f'CUDA solver failed on pairs {begin+np.flatnonzero(failures)}: {failures[failures!=0]}')
        out=out.cpu().numpy()
        result.extend(out[i,:len(a)].copy() for i,(a,b) in enumerate(batch))
    return result

def assemble(points,links):
    counts=np.array([len(x) for x in points],dtype=np.int64)
    offsets=np.r_[0,np.cumsum(counts)]
    total=int(offsets[-1]);frames=len(points)
    successor=np.full(total,-1,dtype=np.int64)
    incoming=np.zeros(total,dtype=bool)
    srcs=[];tgts=[]
    for f,indices in enumerate(links):
        local=np.flatnonzero(indices>=0)
        src=offsets[f]+local;tgt=offsets[f+1]+indices[local]
        successor[src]=tgt;incoming[tgt]=True
        srcs.append(src);tgts.append(tgt)
    roots=np.flatnonzero(~incoming)
    track_id=np.full(total,-1,dtype=np.int64);track_id[roots]=np.arange(len(roots))
    # Same ascending root order and chronological point order as simpletracker.
    tracks=np.full((len(roots),frames),np.nan,dtype=np.float64)
    for f,count in enumerate(counts):
        ids=np.arange(offsets[f],offsets[f+1])
        tid=track_id[ids]
        if (tid<0).any():raise AssertionError('Invalid forward graph')
        tracks[tid,f]=np.arange(1,count+1)
        active=successor[ids]>=0
        track_id[successor[ids[active]]]=tid[active]
    order=np.argsort(track_id,kind='stable')
    lengths=np.bincount(track_id,minlength=len(roots)) if total else np.zeros(0,dtype=np.int64)
    cuts=np.r_[0,np.cumsum(lengths)]
    adjacency=[order[cuts[i]:cuts[i+1]]+1 for i in range(len(roots))]
    return {'tracks':tracks,'adjacency_tracks':adjacency,
            'source_ids':np.concatenate(srcs) if srcs else np.zeros(0,dtype=np.int64),
            'target_ids':np.concatenate(tgts) if tgts else np.zeros(0,dtype=np.int64)}

def track(points,max_distance=8.,max_gap_closing=1,batch_size=384):
    if max_gap_closing!=1:raise ValueError('Only the baseline MaxGapClosing=1 behavior is supported')
    if not points:raise ValueError('At least one frame required')
    for point in points:
        x=np.asarray(point)
        if x.ndim!=2 or x.shape[1]!=2 or x.dtype!=np.float64 or not np.isfinite(x).all():
            raise ValueError('Each frame must be a finite float64 array of shape [N,2]')
    start=time.perf_counter()
    links=link_pairs(list(zip(points[:-1],points[1:])),max_distance,batch_size)
    link_seconds=time.perf_counter()-start
    start=time.perf_counter()
    result=assemble(points,links)
    result['link_seconds']=link_seconds
    result['assembly_seconds']=time.perf_counter()-start
    return result

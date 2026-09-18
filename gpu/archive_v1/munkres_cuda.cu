// Deterministic CUDA translation of Yi Cao's Munkres v2.3 (2011), as
// snapshotted in ../baseline/munkres.m. Preserve zero traversal/append order.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <climits>

constexpr int THREADS = 256;
__device__ double reduce_min(double x, double* buf) {
    int t=threadIdx.x; buf[t]=x; __syncthreads();
    for(int s=THREADS/2;s;s>>=1){if(t<s)buf[t]=fmin(buf[t],buf[t+s]); __syncthreads();}
    double v=buf[0]; __syncthreads(); return v;
}
__device__ double reduce_max(double x, double* buf) {
    int t=threadIdx.x; buf[t]=x; __syncthreads();
    for(int s=THREADS/2;s;s>>=1){if(t<s)buf[t]=fmax(buf[t],buf[t+s]); __syncthreads();}
    double v=buf[0]; __syncthreads(); return v;
}
__device__ double reduce_sum(double x, double* buf) {
    int t=threadIdx.x; buf[t]=x; __syncthreads();
    for(int s=THREADS/2;s;s>>=1){if(t<s)buf[t]+=buf[t+s]; __syncthreads();}
    double v=buf[0]; __syncthreads(); return v;
}
__device__ long long reduce_key(long long x, long long* buf) {
    int t=threadIdx.x; buf[t]=x; __syncthreads();
    for(int s=THREADS/2;s;s>>=1){if(t<s && buf[t+s]<buf[t])buf[t]=buf[t+s]; __syncthreads();}
    long long v=buf[0]; __syncthreads(); return v;
}
__device__ double dist2(const double* src,const double* tgt,int r,int c) {
    double a=tgt[2*c]-src[2*r], b=tgt[2*c+1]-src[2*r+1];
    return a*a+b*b;
}
__device__ bool queued(double val,int r,int c,const double* R,const double* C,
    const double* oldR,const double* oldC,bool residual,double vmin) {
    return residual ? val-(oldR[r]+oldC[c])==vmin : val==R[r]+C[c];
}

__global__ void solve_kernel(const double* sources,const double* targets,const int* counts,
    double* matrices,double* potentials,int* workspace,int* output,int* status,
    int cap,double maxdist2) {
    int b=blockIdx.x,t=threadIdx.x;
    const double* src=sources+(long long)b*cap*2;
    const double* tgt=targets+(long long)b*cap*2;
    int ns=counts[2*b],nt=counts[2*b+1];
    double* D=matrices+(long long)b*cap*cap;
    double* R=potentials+(long long)b*4*cap;
    double* C=R+cap; double* oldR=C+cap; double* oldC=oldR+cap;
    int* rows=workspace+(long long)b*10*cap;
    int* cols=rows+cap; int* star=cols+cap; int* inv=star+cap;
    int* prime=inv+cap; int* cr=prime+cap; int* cc=cr+cap;
    int* cand=cc+cap; int* epoch=cand+cap; int* spare=epoch+cap;
    int* out=output+(long long)b*cap;
    __shared__ double buf[THREADS];
    __shared__ long long keys[THREADS];
    __shared__ int nr,nc,n,chosenR,chosenC,step,tick,residualMode,iterations;
    __shared__ double bigM,maxv,minval;
    for(int i=t;i<cap;i+=THREADS){out[i]=-1;rows[i]=0;cols[i]=0;}
    if(t==0){nr=0;nc=0;status[b]=0;iterations=0;}
    __syncthreads();
    if(!ns || !nt)return;
    // Separate multiplication and addition: compilation disables FMA.
    double sum=0,maxcost=0;
    for(int k=t;k<ns*nt;k+=THREADS){
        int r=k/nt,c=k%nt;
        double v=dist2(src,tgt,r,c);
        if(v<=maxdist2){atomicExch(rows+r,1);atomicExch(cols+c,1);sum+=v;maxcost=fmax(maxcost,v);}
    }
    double total=reduce_sum(sum,buf);
    double maximum=reduce_max(maxcost,buf);
    if(t==0){
        for(int r=0;r<ns;r++)if(rows[r])rows[nr++]=r;
        for(int c=0;c<nt;c++)if(cols[c])cols[nc++]=c;
        n=nr>nc?nr:nc;
        bigM=total>0?pow(10.0,ceil(log10(total))+1.0):0.0;
        maxv=10.0*maximum;
    }
    __syncthreads();
    if(n==0)return;
    for(int k=t;k<n*n;k+=THREADS){
        int r=k/n,c=k%n;
        double v=maxv;
        if(r<nr && c<nc){v=dist2(src,tgt,rows[r],cols[c]);if(v>maxdist2)v=bigM;}
        D[k]=v;
    }
    __syncthreads();
    for(int r=t;r<n;r+=THREADS){
        double v=INFINITY;for(int c=0;c<n;c++)v=fmin(v,D[r*n+c]);
        R[r]=v;star[r]=-1;inv[r]=-1;
    }
    __syncthreads();
    for(int c=t;c<n;c+=THREADS){
        double v=INFINITY;for(int r=0;r<n;r++)v=fmin(v,D[r*n+c]-R[r]);C[c]=v;
    }
    __syncthreads();
    // MATLAB find(...,1) visits rows inside columns, greedily starring zeros.
    for(int c=0;c<n;c++){
        long long row=LLONG_MAX;
        for(int r=t;r<n;r+=THREADS)if(star[r]<0 && D[r*n+c]==R[r]+C[c]){row=r;break;}
        long long selected=reduce_key(row,keys);
        if(t==0 && selected!=LLONG_MAX){star[selected]=c;inv[c]=selected;}
        __syncthreads();
    }
    while(true){
        if(t==0){step=0;for(int r=0;r<n;r++)if(star[r]<0){step=1;break;}}
        __syncthreads();
        if(step==0)break;
        for(int i=t;i<n;i+=THREADS){cr[i]=0;cc[i]=inv[i]>=0;prime[i]=-1;epoch[i]=0;}
        __syncthreads();
        for(int c=t;c<n;c+=THREADS){
            cand[c]=-1;
            if(!cc[c])for(int r=0;r<n;r++)if(D[r*n+c]==R[r]+C[c]){cand[c]=r;break;}
        }
        if(t==0){tick=0;residualMode=0;minval=0;}
        __syncthreads();
        while(true){
            if(t==0){iterations++;if(iterations>10000000)status[b]=1;}
            __syncthreads();
            if(status[b])return;
            long long best=LLONG_MAX;
            for(int c=t;c<n;c+=THREADS)if(cand[c]>=0){
                long long key=((long long)epoch[c]*n+c)*n+cand[c];
                if(key<best)best=key;
            }
            long long selected=reduce_key(best,keys);
            if(selected!=LLONG_MAX){
                if(t==0){
                    chosenR=selected%n;chosenC=(selected/n)%n;
                    prime[chosenR]=chosenC;
                    step=star[chosenR]<0?5:4;
                    if(step==4){cr[chosenR]=1;cc[star[chosenR]]=0;tick++;}
                }
                __syncthreads();
                if(step==5)break;
                int newly=star[chosenR];
                for(int c=t;c<n;c+=THREADS){
                    if(c==newly){
                        epoch[c]=tick;cand[c]=-1;
                        for(int r=0;r<n;r++)if(!cr[r] && D[r*n+c]==R[r]+C[c]){cand[c]=r;break;}
                    }else if(cand[c]==chosenR){
                        cand[c]=-1;
                        for(int r=chosenR+1;r<n;r++)if(!cr[r] && queued(D[r*n+c],r,c,R,C,oldR,oldC,residualMode && epoch[c]==0,minval)){cand[c]=r;break;}
                    }
                }
                __syncthreads();
            }else{
                // Preserve outerplus's subtraction grouping: D - (R + C).
                double v=INFINITY;
                for(int k=t;k<n*n;k+=THREADS){int r=k/n,c=k%n;if(!cr[r] && !cc[c])v=fmin(v,D[k]-(R[r]+C[c]));}
                double minimum=reduce_min(v,buf);
                if(t==0){minval=minimum;tick=0;residualMode=1;if(!isfinite(minimum))status[b]=2;}
                __syncthreads();
                if(status[b])return;
                for(int i=t;i<n;i+=THREADS){oldR[i]=R[i];oldC[i]=C[i];epoch[i]=0;}
                __syncthreads();
                // Save candidate rows BEFORE updating potentials, as in MATLAB.
                for(int c=t;c<n;c+=THREADS){
                    cand[c]=-1;
                    if(!cc[c])for(int r=0;r<n;r++)if(!cr[r] && D[r*n+c]-(R[r]+C[c])==minimum){cand[c]=r;break;}
                }
                __syncthreads();
                for(int i=t;i<n;i+=THREADS){if(!cc[i])C[i]+=minimum;if(cr[i])R[i]-=minimum;}
                __syncthreads();
            }
        }
        if(t==0){
            int r=chosenR,c=chosenC;
            while(true){
                int prior=inv[c];
                star[r]=c;inv[c]=r;
                if(prior<0)break;
                star[prior]=-1;r=prior;c=prime[r];
            }
        }
        __syncthreads();
    }
    for(int r=t;r<nr;r+=THREADS){
        int c=star[r];
        if(c>=0 && c<nc && dist2(src,tgt,rows[r],cols[c])<=maxdist2)out[rows[r]]=cols[c];
    }
}

std::vector<torch::Tensor> solve_cuda(torch::Tensor sources,torch::Tensor targets,torch::Tensor counts,double distance){
    TORCH_CHECK(sources.is_cuda() && targets.is_cuda() && counts.is_cuda(),"CUDA tensors required");
    TORCH_CHECK(sources.scalar_type()==torch::kFloat64 && targets.scalar_type()==torch::kFloat64,"float64 required");
    TORCH_CHECK(counts.scalar_type()==torch::kInt32,"int32 counts required");
    TORCH_CHECK(sources.is_contiguous() && targets.is_contiguous() && counts.is_contiguous(),"contiguous tensors required");
    TORCH_CHECK(sources.dim()==3 && sources.size(2)==2 && sources.sizes()==targets.sizes(),"[batch,capacity,2] required");
    TORCH_CHECK(counts.dim()==2 && counts.size(0)==sources.size(0) && counts.size(1)==2,"counts shape invalid");
    c10::cuda::CUDAGuard guard(sources.device());
    int batch=sources.size(0),cap=sources.size(1);
    auto opt=sources.options();auto ints=counts.options();
    auto matrix=torch::empty({batch,cap,cap},opt);
    auto potentials=torch::empty({batch,4,cap},opt);
    auto workspace=torch::empty({batch,10,cap},ints);
    auto output=torch::empty({batch,cap},ints);
    auto status=torch::zeros({batch},ints);
    if(batch && cap){
        solve_kernel<<<batch,THREADS,0,c10::cuda::getCurrentCUDAStream()>>>(sources.data_ptr<double>(),targets.data_ptr<double>(),counts.data_ptr<int>(),matrix.data_ptr<double>(),potentials.data_ptr<double>(),workspace.data_ptr<int>(),output.data_ptr<int>(),status.data_ptr<int>(),cap,distance*distance);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {output,status};
}

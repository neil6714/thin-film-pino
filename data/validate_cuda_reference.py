"""Compare one matched trajectory from the reference and fast CUDA simulators."""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from physics.simulator import ThinFilmDepositionSimulator
from physics.batched_transient_cuda import BatchedTransientALDCudaSimulator

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='results/stage4_validation'); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    from data.generate_transient_dataset import sample_parameters
    rng=np.random.default_rng(args.seed); p=sample_parameters(rng)
    kw=dict(diffusivity=p[0],k_ads=p[1],k_des=p[2],k_rxn=p[3],k_growth=p[4],pulse_time=p[5],purge_time=p[6],reaction_time=p[7],num_cycles=10)
    ref=ThinFilmDepositionSimulator(**kw); _, ref_theta, ref_h=ref.run()
    try:
        import torch
        fast=BatchedTransientALDCudaSimulator(np.asarray([p]),nx=80,ny=60); steps=fast.pulse_steps(); purge=fast.fixed_steps(fast.purge_time); reaction=fast.fixed_steps(fast.reaction_time)
        for _ in range(10):
            for s in range(1,int(steps.max())+1): fast.transport_step(steps>=s,True)
            for _ in range(purge): fast.transport_step(torch.ones(1,dtype=torch.bool,device='cuda'),False)
            fast.C.zero_()
            for _ in range(reaction): fast.reaction_step()
            for _ in range(purge): fast.transport_step(torch.ones(1,dtype=torch.bool,device='cuda'),False)
            fast.C.zero_()
        _, theta, h=fast.snapshot(); theta=theta[0]; h=h[0]
    except Exception as exc: raise RuntimeError('CUDA comparison requires a working CUDA runtime') from exc
    ref_theta=zoom(ref_theta,(60/120,80/160),order=1); ref_h=zoom(ref_h,(60/120,80/160),order=1); err_t=np.abs(ref_theta-theta); err_h=np.abs(ref_h-h)
    metrics={'surface_coverage_mae':float(err_t.mean()),'surface_coverage_rmse':float(np.sqrt((err_t**2).mean())),'film_thickness_mae':float(err_h.mean()),'film_thickness_rmse':float(np.sqrt((err_h**2).mean())),'reference_grid':[120,160],'cuda_grid':[60,80],'parameters':p.tolist()}
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); (out/'reference_vs_cuda.json').write_text(json.dumps(metrics,indent=2))
    fig,ax=plt.subplots(2,3,figsize=(12,7));
    for a,v,t in zip(ax[0],[ref_h,h,err_h],['Reference thickness','CUDA thickness','Absolute error']): a.imshow(v); a.set_title(t); a.axis('off')
    for a,v,t in zip(ax[1],[ref_theta,theta,err_t],['Reference coverage','CUDA coverage','Absolute error']): a.imshow(v); a.set_title(t); a.axis('off')
    fig.tight_layout(); fig.savefig(out/'reference_vs_cuda.png',dpi=200); plt.close(fig); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()

"""Evaluate an existing Stage 4C checkpoint with corrected global metrics."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from training.train_temporal_operator import TemporalFourierOperator

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output-dir',default='results/stage4c_temporal_operator_corrected'); args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('Stage 4C evaluation requires CUDA.')
    d=np.load(args.input); device=torch.device('cuda'); ck=torch.load(args.checkpoint,map_location=device,weights_only=False); cfg=ck.get('args',{})
    model=TemporalFourierOperator(width=cfg.get('width',32),modes=cfg.get('modes',12)).to(device); model.load_state_dict(ck['model']); model.eval()
    coords=torch.stack(torch.meshgrid(torch.as_tensor(d['y'],device=device),torch.as_tensor(d['x'],device=device),indexing='ij'),0).float()*2-1
    idx=d['test_indices']; p=torch.as_tensor(d['parameters'][idx],device=device); t=torch.as_tensor(d['time'][idx],device=device)
    target=torch.as_tensor(np.stack([d['concentration'][idx],d['surface_coverage'][idx],d['film_thickness'][idx]],2),device=device)
    mask=torch.as_tensor(np.stack([~d['solid_mask'],d['surface_mask'],d['surface_mask']]),device=device).float()[None,None]
    with torch.no_grad(): pred=model(p,t,coords)
    diff=pred-target; valid=mask.expand_as(diff); denom=valid.sum().clamp_min(1); scale=torch.as_tensor(d['field_std'],device=device)[None,None,:,None,None]; physical=diff*scale
    metrics={}
    for c,name in enumerate(('concentration','surface_coverage','film_thickness')):
        v=valid[:,:,c]; e=physical[:,:,c]; n=v.sum().clamp_min(1); y=(target[:,:,c]*scale[:,:,c]).masked_select(v.bool()); err=(e*v).masked_select(v.bool())
        metrics[name]={'MAE_physical':float(err.abs().sum()/n),'RMSE_physical':float(torch.sqrt(err.square().sum()/n)),'MAE_normalized':float((diff[:,:,c].abs()*v).sum()/n),'RMSE_normalized':float(torch.sqrt((diff[:,:,c].square()*v).sum()/n)),'R2_physical':float(1-(err.square().sum()/((y-y.mean()).square().sum().clamp_min(1e-12))))}
    metrics['overall']={'MSE_normalized':float((diff.square()*valid).sum()/denom),'MAE_normalized':float((diff.abs()*valid).sum()/denom),'test_trajectories':int(len(idx)),'grid':[int(d['concentration'].shape[-2]),int(d['concentration'].shape[-1])],'checkpoint':args.checkpoint}
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2)); print(f'Saved corrected metrics to {out}')
if __name__=='__main__': main()

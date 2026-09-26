"""Read a committed fine-grid prefix to diagnose precision while it runs.

No state transfer, no interpolation and no claim of full-time acceptance.
"""
import json
from pathlib import Path
import numpy as np
from q2_model import sha256
from q2_full_analysis import BASE, ROOT, load_main,load_node,compare,convergence,serial
from q2_grid_extrapolation import extrapolate


def cut(d,times):
    ids=np.searchsorted(d['time_s'],times)
    assert np.array_equal(d['time_s'][ids],times)
    return dict(time_s=times,radius_m=d['radius_m'],T_C=d['T_C'][ids],C_kg_kg=d['C_kg_kg'][ids])


def main():
    folder=BASE/'runs/space-16384'
    meta=json.loads((folder/'run.json').read_text(encoding='utf-8'))
    assert meta['signature']['driver_sha256']==sha256(ROOT/'src/q2_full_run.py')
    assert meta['signature']['settings']['cells']==16384
    chunks=[]
    for seg in meta['segments']:
        path=folder/seg['file'];assert sha256(path)==seg['sha256']
        with np.load(path) as q:chunks.append({k:q[k].copy() for k in ['time_s','radius_m','T_C','C_kg_kg']})
    t=np.r_[0.,np.concatenate([x['time_s'] for x in chunks])]
    fine=dict(time_s=t,radius_m=chunks[0]['radius_m'],
        T_C=np.concatenate([np.full((1,21),28.),*[x['T_C'] for x in chunks]]),
        C_kg_kg=np.concatenate([np.full((1,21),2.55),*[x['C_kg_kg'] for x in chunks]]))
    c=cut(load_main('space-4096'),t);m=cut(load_main('primary-8192'),t)
    na=cut(load_node('full-N8192-tight-s05'),t);nb=cut(load_node('full-N16384-tight-s05'),t)
    r8=extrapolate(c,m);r16=extrapolate(m,fine);nr16=extrapolate(na,nb)
    details=[]
    for ti,ri,field in [(3885.,.003,'T_C'),(3664.,0.,'T_C'),(9723.,.017,'C_kg_kg')]:
        if ti>t[-1]:continue
        it=int(np.flatnonzero(t==ti)[0]);ir=int(np.argmin(abs(fine['radius_m']-ri)))
        value=float(r16[field][it,ir]);boundary=(np.floor(value*1e4)+.5)/1e4
        details.append(dict(time_s=ti,radius_m=ri,field=field,main_extrapolate=value,
            previous_extrapolate=float(r8[field][it,ir]),independent_extrapolate=float(nr16[field][it,ir]),
            signed_half_distance=float(value-boundary)))
    result=dict(status='PREFIX_DIAGNOSTIC_NOT_FULL_ACCEPTANCE',through_s=float(t[-1]),
        source_run_status=meta['status'],main_convergence=convergence(c,m,fine),
        successive_extrapolates=compare(r8,r16),independent_extrapolate=compare(r16,nr16),
        sensitive_points=details)
    dest=BASE/'prefix-reviews';dest.mkdir(exist_ok=True)
    (dest/f'{int(t[-1]):05d}.json').write_text(json.dumps(result,indent=2,default=serial),encoding='utf-8')
    print(json.dumps(result,indent=2,default=serial))


if __name__=='__main__':main()

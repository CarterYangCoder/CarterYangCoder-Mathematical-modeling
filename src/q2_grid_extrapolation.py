"""Uniform, conditional spatial Richardson candidate for all Q2 point outputs.

This is a numerical postprocessor, not an original integrator, local digit fix,
new physical model, conserved FV trajectory, or resume state. It cannot issue
an acceptance gate. All underlying solves and counterexamples remain intact.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from q2_full_analysis import BASE, ROOT, load_main, same_grid, compare, serial
from q2_model import sha256


def extrapolate(coarse,fine):
    same_grid(coarse,fine)
    answer={key:fine[key].copy() for key in ['time_s','radius_m']}
    for field in ['T_C','C_kg_kg']:
        # Stable algebra, and exactly preserves matching t=0 initial values.
        answer[field]=fine[field]+(fine[field]-coarse[field])/3
    return answer


def candidate(coarse_label,fine_label,label):
    coarse=load_main(coarse_label);fine=load_main(fine_label)
    cs=coarse['_meta']['signature']['settings'];fs=fine['_meta']['signature']['settings']
    if fs['cells']!=2*cs['cells']:raise ValueError('Require spatial step ratio exactly 2')
    for k in cs:
        if k not in ('cells','audit') and cs[k]!=fs[k]:
            raise ValueError(f'Unmatched temporal settings in space comparison: {k}')
    output=extrapolate(coarse,fine)
    assert np.all(output['T_C'][0]==28) and np.all(output['C_kg_kg'][0]==2.55)
    assert np.all(np.isfinite(output['T_C'])) and np.all(np.isfinite(output['C_kg_kg']))
    assert np.all(output['C_kg_kg']>0) and np.all(output['T_C']+273.15>0)
    meta=dict(question='Q2',status='UNACCEPTED_UNIFORM_SPATIAL_EXTRAPOLATION_CANDIDATE',
        candidate_label=label,formula='u_R = u_f + (u_f-u_c)/3, applied uniformly to both fields at every stored physical coordinate',
        scheme='unchanged direct-Fick interior, separated-potential independent surface, monolithic sparse Radau; then spatial postprocessing',
        extrapolation_order_assumption=2,fourth_order_claim=False,
        source_grid_cells=[cs['cells'],fs['cells']],
        sources={str(ROOT/x['_source']):sha256(ROOT/x['_source']) for x in (coarse,fine)},
        postprocessor_sha256=sha256(__file__),no_per_point_branch=True,
        no_clipping=True,no_intermediate_decimal_rounding=True,physical_model_unchanged=True,
        differences_from_fine=compare(fine,output),
        limitations=['Requires measured asymptotic spatial behavior and independent checks over the actual output range.',
            'Nonlinear Robin and integral ledgers belong to the source FV trajectories; the extrapolated point array does not exactly inherit their algebraic identities.',
            'No assumption of an even-power expansion beyond the observed leading second order; in particular do not divide the extrapolate change by 15.',
            'Time errors of both source levels propagate with weights 4/3 and -1/3.',
            'Not a checkpoint; never reconstruct a PDE initial state from these 21 point values.',
            'The non-submitted 0.001 s pressure failures remain recorded; no uniform t-to-zero precision claim.'])
    dest=BASE/'candidates'/label;dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'solution.npz',**output)
    meta['solution_sha256']=sha256(dest/'solution.npz')
    (dest/'candidate.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False,default=serial),encoding='utf-8')
    return meta


def main():
    p=argparse.ArgumentParser();p.add_argument('--coarse',required=True);p.add_argument('--fine',required=True);p.add_argument('--label',required=True)
    a=p.parse_args();print(json.dumps(candidate(a.coarse,a.fine,a.label),indent=2,ensure_ascii=False,default=serial))


if __name__=='__main__':main()

"""Quantify nonlinear commutation defects of uniform spatial extrapolation.

These are algebraic consistency checks, not an independent PDE solution or a
claim that a Richardson point array is an exactly conserved FV trajectory.
"""
import argparse
import json
import numpy as np
from q2_full_analysis import BASE,ROOT,maximum,serial
from q2_grid_extrapolation import extrapolate
from q2_model import sha256
from openpyxl import load_workbook


def thermal(C):
    return (650+128*C)*(1450+2736*C/(1+C)), .21+.38*C/(1+C)


def diffusion(T,C):
    return .0024*np.exp(-.45/C)*np.exp(-3850/(T+273.15))


def audit(label):
    folder=BASE/'candidates'/label
    meta=json.loads((folder/'candidate.json').read_text(encoding='utf-8'))
    assert sha256(folder/'solution.npz')==meta['solution_sha256']
    assert sha256(ROOT/'src/q2_grid_extrapolation.py')==meta['postprocessor_sha256']
    sources=[]
    for path,digest in meta['sources'].items():
        assert sha256(path)==digest
        with np.load(path) as q:sources.append({k:q[k].copy() for k in q.files})
    a,b=sources
    r=extrapolate(a,b)
    with np.load(folder/'solution.npz') as q:
        for field in ['time_s','radius_m','T_C','C_kg_kg']:
            assert np.array_equal(q[field],r[field])
    wr,kr=thermal(r['C_kg_kg']);wa,ka=thermal(a['C_kg_kg']);wb,kb=thermal(b['C_kg_kg'])
    er=wr*(r['T_C']-28)
    ef=wb*(b['T_C']-28);ec=wa*(a['T_C']-28)
    eext=ef+(ef-ec)/3
    w0=float(thermal(np.array(2.55))[0])
    storage_defect=abs(er-eext)
    book=load_workbook(ROOT/'A题/附件/附件1.xlsx',read_only=True,data_only=True)
    env=np.array(list(book[book.sheetnames[0]].iter_rows(min_row=2,values_only=True)),dtype=float)
    book.close();t=r['time_s']
    ta=np.interp(t,env[:,0],env[:,1]);ce=np.interp(t,env[:,0],env[:,2])
    # Implied gradients of the source discretizations, before extrapolation.
    gaT=-25*(a['T_C'][:,-1]-ta)/ka[:,-1]
    gbT=-25*(b['T_C'][:,-1]-ta)/kb[:,-1]
    da=diffusion(a['T_C'][:,-1],a['C_kg_kg'][:,-1])
    db=diffusion(b['T_C'][:,-1],b['C_kg_kg'][:,-1])
    dr=diffusion(r['T_C'][:,-1],r['C_kg_kg'][:,-1])
    gaC=-8e-7*(a['C_kg_kg'][:,-1]-ce)/da
    gbC=-8e-7*(b['C_kg_kg'][:,-1]-ce)/db
    flux_T_defect=abs(-kr[:,-1]*(gbT+(gbT-gaT)/3)-25*(r['T_C'][:,-1]-ta))
    flux_C_defect=abs(-dr*(gbC+(gbC-gaC)/3)-8e-7*(r['C_kg_kg'][:,-1]-ce))
    valid=t>=1
    at=lambda arr:dict(maximum=float(np.max(arr[valid])),time_s=float(t[np.flatnonzero(valid)[np.argmax(arr[valid])]]))
    result=dict(label=label,status='ALGEBRAIC_POSTPROCESSING_CHECK_COMPLETE_NOT_PDE_ACCEPTANCE',
        uniform_formula_exactly_reproduced=True,
        pointwise_heat_storage_commutation_J_m3=maximum(storage_defect,r),
        heat_storage_commutation_divided_by_initial_W_K=maximum(storage_defect/w0,r),
        surface_heat_flux_commutation_W_m2=at(flux_T_defect),
        surface_moisture_flux_commutation_m_s=at(flux_C_defect),
        initial_values_exact=True,
        source_integral_ledgers='Only source trajectories have an actual FV state ledger; C and linear Robin exchange may be combined with the same weights as integral estimates.',
        explicit_limits=['Gradients in this check are inferred from source Robin fluxes. Their commutation defects do not independently validate the true derivative of the extrapolated field.',
            'Heat density is evaluated at the 21 physical output points, not integrated as a newly conserved FV state.',
            'Small nonlinear defects must not be added again as independent observations of space/time error.',
            'The actual continuum accuracy and rounding gate require spatial, temporal and independent comparisons.'])
    (folder/'nonlinear-commutation-audit.json').write_text(json.dumps(result,indent=2,default=serial),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');args=p.parse_args()
    print(json.dumps(audit(args.label),indent=2,default=serial))

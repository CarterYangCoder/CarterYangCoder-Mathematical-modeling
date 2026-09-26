"""Version-checked Q2 full-output comparisons and engineering error evidence.

No PDE solve, no manual digit editing. A comparison is not physical validation.
"""
from pathlib import Path
import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from q2_model import sha256

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'records/q2-full-20260911'


def rounded(values):
    values=np.asarray(values,dtype=float)
    q=Decimal('0.0001')
    return np.fromiter((float(Decimal.from_float(float(v)).quantize(q,rounding=ROUND_HALF_UP))
                        for v in values.ravel()),dtype=float,count=values.size).reshape(values.shape)


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def load_main(label):
    folder=BASE/'runs'/label
    r=json.loads((folder/'run.json').read_text(encoding='utf-8'))
    if r['status']!='COMPLETED' or r['end_s']!=10800:
        raise ValueError(f'Incomplete main run: {label}')
    for name,key in [('src/q2_model.py','model_sha256'),('src/q1_model.py','helper_sha256'),
                     ('src/q2_full_run.py','driver_sha256'),('config/q2-full.json','config_sha256')]:
        if sha256(ROOT/name)!=r['signature'][key]:raise ValueError(f'Changed {name}: {label}')
    path=folder/'solution.npz'
    if sha256(path)!=r['solution_sha256']:raise ValueError(f'Changed output: {label}')
    with np.load(path) as d:out={k:d[k].copy() for k in d.files}
    out['_source']=str(path.relative_to(ROOT));out['_meta']=r
    return out


def load_node(label):
    folder=BASE/'node-reference'/label
    r=json.loads((folder/'run.json').read_text(encoding='utf-8'))
    if r['status']!='COMPUTATION_COMPLETE_REVIEW_PENDING' or r['completed_through_s']!=10800:
        raise ValueError(f'Incomplete independent run: {label}')
    path=folder/'solution.npz'
    if sha256(path)!=r['result_sha256']:raise ValueError(f'Changed node output: {label}')
    for name,key in [('src/q2_node_reference.py','frozen_node_source'),('src/q2_full_node.py','full_driver_source'),
                     ('A题/附件/附件1.xlsx','environment')]:
        if sha256(ROOT/name)!=r['hashes'][key]:raise ValueError(f'Changed node dependency {name}: {label}')
    # All actual driver and frozen-source hashes are also bound by final manifest.
    with np.load(path) as d:out={k:d[k].copy() for k in d.files}
    out['_source']=str(path.relative_to(ROOT));out['_meta']=r
    return out


def same_grid(a,b):
    if not np.array_equal(a['time_s'],b['time_s']) or not np.array_equal(a['radius_m'],b['radius_m']):
        raise ValueError('Comparison must use identical physical coordinates')


def maximum(value,a,mask=None):
    if mask is None:mask=a['time_s']>=1
    it0=np.flatnonzero(mask);sub=np.asarray(value)[mask]
    ix=np.unravel_index(np.argmax(sub),sub.shape);it=int(it0[ix[0]]);ir=int(ix[1])
    return dict(value=float(sub[ix]),time_s=float(a['time_s'][it]),radius_m=float(a['radius_m'][ir]))


def compare(a,b):
    same_grid(a,b);mask=a['time_s']>=1
    integer=mask&(a['time_s']==np.round(a['time_s']))
    answer={}
    for field in ['T_C','C_kg_kg']:
        delta=abs(a[field]-b[field]);m=maximum(delta,a)
        it=int(np.flatnonzero(a['time_s']==m['time_s'])[0]);ir=int(np.flatnonzero(a['radius_m']==m['radius_m'])[0])
        m.update(value_a=float(a[field][it,ir]),value_b=float(b[field][it,ir]),
            integer_maximum=maximum(delta,a,integer),
            four_decimal_differences=int(np.count_nonzero(rounded(a[field][integer])!=rounded(b[field][integer]))))
        answer[field]=m
    return answer


def convergence(coarse,middle,fine):
    same_grid(coarse,middle);same_grid(middle,fine)
    ans={};mask=middle['time_s']>=1
    for field in ['T_C','C_kg_kg']:
        first=middle[field]-coarse[field];second=fine[field]-middle[field]
        m1=maximum(abs(first),middle);m2=maximum(abs(second),middle)
        pmax=float(np.log2(m1['value']/m2['value'])) if m2['value']>0 else None
        floor=1e-10 if field=='T_C' else 1e-12
        good=(abs(first)>20*floor)&(abs(second)>5*floor)&mask[:,None]&(first*second>0)
        orders=np.log2(abs(first[good]/second[good]))
        resolved=(abs(first)>20*floor)&(abs(second)>5*floor)&mask[:,None]
        reversed_mask=resolved&(first*second<0)
        unusual=np.zeros_like(good)
        unusual[good]=(orders<1.7)|(orders>2.3)
        ans[field]=dict(coarse_to_middle=m1,middle_to_fine=m2,norm_order=pmax,
            local_order_rule=f'same signed changes above 20*{floor:g}/5*{floor:g}; no inference for roundoff-small differences',
            resolved_local_orders=len(orders),local_order_quantiles=np.quantile(orders,[0,.01,.1,.5,.9,.99,1]) if len(orders) else [],
            same_direction_resolved_count=int(np.count_nonzero(good)),
            reversed_direction_resolved_count=int(reversed_mask.sum()),
            orders_outside_1p7_2p3_count=int(unusual.sum()),
            reversed_max_fine_difference=maximum(np.where(reversed_mask,abs(second),0),middle),
            unusual_order_max_fine_difference=maximum(np.where(unusual,abs(second),0),middle),
            qualification='A smooth norm order is insufficient; adverse local regimes remain explicit and require separate rounding/error examination')
    return ans


def error_evidence(candidate,coarse,fine,time_runs,nodecoarse,nodefine,nodetight):
    """Candidate is middle grid, fine spacing half; independent reference fine.

    1.25 and 2 are documented engineering margins, not rigorous constants.
    This helper does not silently resolve adverse roundings with extrapolation.
    """
    for x in [coarse,fine,*time_runs,nodecoarse,nodefine,nodetight]:same_grid(candidate,x)
    integer=(candidate['time_s']>=1)&(candidate['time_s']==np.round(candidate['time_s']))
    arrays={};records={};flags=[]
    for field in ['T_C','C_kg_kg']:
        dtime=np.maximum.reduce([abs(candidate[field]-x[field]) for x in time_runs])
        ntime=abs(nodefine[field]-nodetight[field])
        floor=1e-10 if field=='T_C' else 1e-11
        own=1.25*4/3*abs(fine[field]-candidate[field])
        ref=1.25/3*abs(nodefine[field]-nodecoarse[field])
        routeA=own+2*dtime+floor
        routeB=abs(candidate[field]-nodefine[field])+ref+2*ntime+floor
        env=np.maximum(routeA,routeB)
        boundary=(np.floor(candidate[field]*1e4)+.5)/1e4
        distance=abs(candidate[field]-boundary)
        flagged=(distance<=env)&integer[:,None]
        arrays[field+'_engineering_envelope']=env
        arrays[field+'_rounding_distance']=distance
        arrays[field+'_main_time_change']=dtime
        arrays[field+'_reference_time_change']=ntime
        records[field]=dict(engineering_envelope_max=maximum(env,candidate),
            candidate_time_change_max=maximum(dtime,candidate),reference_time_change_max=maximum(ntime,candidate),
            own_space_estimate_max=maximum(own,candidate),reference_space_estimate_max=maximum(ref,candidate),
            initial_rounding_flags=int(flagged.sum()))
        for it,ir in zip(*np.where(flagged)):
            flags.append(dict(field=field,time_s=float(candidate['time_s'][it]),radius_m=float(candidate['radius_m'][ir]),
                candidate=float(candidate[field][it,ir]),coarse=float(coarse[field][it,ir]),fine=float(fine[field][it,ir]),
                node=float(nodefine[field][it,ir]),boundary=float(boundary[it,ir]),distance=float(distance[it,ir]),
                engineering_envelope=float(env[it,ir]),status='UNRESOLVED_REQUIRES_TARGETED_EVIDENCE'))
    return dict(formula_A='1.25*4/3*abs(fine-candidate)+2*max(main time changes)+floor',
        formula_B='abs(candidate-reference_fine)+1.25/3*abs(reference_fine-reference_coarse)+2*reference_time_change+floor',
        combination='max(A,B); candidate time change not added again in direct-difference route B',
        limitation='Engineering estimates conditional on measured spatial regime; not strict bounds, physical validation or confidence intervals',
        values=records,integer_values_scanned=int(integer.sum()*len(candidate['radius_m'])*2),
        unresolved_rounding_count=len(flags),flags=flags),arrays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('labels',nargs='+');parser.add_argument('--node',action='store_true')
    args=parser.parse_args();load=load_node if args.node else load_main
    runs=[load(label) for label in args.labels]
    result={'sources':[x['_source'] for x in runs],
            'pairs':[compare(a,b) for a,b in zip(runs[:-1],runs[1:])]}
    if len(runs)==3:result['convergence']=convergence(*runs)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=serial))


if __name__=='__main__':main()

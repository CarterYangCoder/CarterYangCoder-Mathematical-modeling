"""Version-checked Q2 full-array engineering precision and rounding review.

The envelope is empirical, not a rigorous bound or physical uncertainty. The
same complete Richardson formula applies to every point. No array is patched.
"""
import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import numpy as np
from q2_full_analysis import BASE,ROOT,load_main,load_node,same_grid,compare,convergence,maximum,rounded,serial
from q2_grid_extrapolation import extrapolate
from q2_model import sha256

FIELDS=('T_C','C_kg_kg')


def half_distances(values):
    result=np.empty_like(values);boundaries=np.empty_like(values)
    for i,x in enumerate(values.ravel()):
        value=Decimal.from_float(float(x))
        lower=int(value//Decimal('.0001'))
        boundary=(Decimal(lower)+Decimal('.5'))*Decimal('.0001')
        result.ravel()[i]=float(abs(value-boundary));boundaries.ravel()[i]=float(boundary)
    return result,boundaries


def load_candidate(label):
    folder=BASE/'candidates'/label
    meta=json.loads((folder/'candidate.json').read_text(encoding='utf-8'))
    assert sha256(folder/'solution.npz')==meta['solution_sha256']
    assert sha256(ROOT/'src/q2_grid_extrapolation.py')==meta['postprocessor_sha256']
    for source,digest in meta['sources'].items():assert sha256(source)==digest
    with np.load(folder/'solution.npz') as q:data={k:q[k].copy() for k in q.files}
    return data,meta,folder


def level_spread(sequence,field):
    # Do not divide by 15 or assume an even-power expansion of the point values.
    return np.maximum(abs(sequence[2][field]-sequence[1][field]),
                      abs(sequence[1][field]-sequence[0][field]))


def load_ultra_prefix(n,strict_full):
    """Validate a fresh original-initial-state reference, never a pasted state."""
    folder=BASE/'node-reference'/f'prefix-N{n}-ultra-3900'
    meta=json.loads((folder/'run.json').read_text(encoding='utf-8'))
    if (meta['status']!='COMPUTATION_COMPLETE_REVIEW_PENDING'
            or meta['completed_through_s']!=3900
            or not meta['original_initial_state_used']
            or meta['resume_checkpoint'] is not None):
        raise ValueError('Ultra prefix must complete 0..3900 from original initial state')
    s=meta['settings']
    if (s['n_intervals'],s['rtol'],s['atol_T'],s['atol_C'],s['max_step'],s['threads'],s['stop_s']) != (n,3e-14,3e-14,3e-16,.25,1,3900.):
        raise ValueError('Unexpected prefix reference settings')
    for name,key in [('src/q2_node_reference.py','frozen_node_source'),
                     ('src/q2_full_node.py','full_driver_source'),('A题/附件/附件1.xlsx','environment')]:
        if sha256(ROOT/name)!=meta['hashes'][key] or meta['hashes'][key]!=strict_full['_meta']['hashes'][key]:
            raise ValueError(f'Prefix changed a spatial/input dependency: {name}')
    path=folder/'solution.npz'
    if sha256(path)!=meta['result_sha256']:raise ValueError('Changed prefix NPZ')
    with np.load(path,allow_pickle=False) as q:out={k:q[k].copy() for k in q.files}
    if not np.array_equal(out['radius_m'],strict_full['radius_m']):raise ValueError('Prefix radii changed')
    t=out['time_s'];ix=np.searchsorted(strict_full['time_s'],t)
    if (t[0]!=0 or t[-1]!=3900 or np.any(np.diff(t)<=0) or np.any(ix>=len(strict_full['time_s']))
            or not np.array_equal(strict_full['time_s'][ix],t)):
        raise ValueError('Prefix times require exact physical-coordinate matching')
    if not np.array_equal(t[(t==np.floor(t))],np.arange(3901)):
        raise ValueError('Prefix does not contain all integer times')
    from q2_node_reference import point_sampling_matrix
    expected_nodes=np.linspace(0.,.02,n+1)
    if not np.array_equal(out['node_radii_m'],expected_nodes):raise ValueError('Changed nodal geometry')
    sample=point_sampling_matrix(expected_nodes,out['radius_m'])
    initial_sampling={}
    for f,v in [('T_C',28.),('C_kg_kg',2.55)]:
        initial_points=sample@np.full(n+1,v)
        if (out[f].shape!=(len(t),21) or not np.all(np.isfinite(out[f]))
                or not np.array_equal(out[f][0],initial_points)):
            raise ValueError('Invalid prefix output or initial output differs from frozen sampler applied to original uniform nodes')
        initial_sampling[f]=dict(matches_frozen_sampling_exactly=True,
            maximum_sampling_roundoff=float(np.max(abs(initial_points-v))))
    if np.any(out['C_kg_kg']<=0) or np.any(out['T_C']+273.15<=0):raise ValueError('Invalid state domain')
    out['_source']=str(path.relative_to(ROOT));out['_meta']=meta
    out['_initial_sampling_check']=initial_sampling
    return out,ix


def review(name,screen=False,ultra_prefix=False):
    sources={}
    def mainrun(label):
        r=load_main(label);sources[label]=r;return r
    def noderun(label):
        r=load_node(label);sources[label]=r;return r
    main=[mainrun(x) for x in ['space-2048','space-4096','primary-8192','space-16384']]
    nodes=[noderun(f'full-N{n}-tight-s05') for n in [2048,4096,8192,16384]]
    node_base=[noderun(f'full-N{n}-s1') for n in [8192,16384]]
    rmain=[extrapolate(a,b) for a,b in zip(main[:-1],main[1:])]
    rnode=[extrapolate(a,b) for a,b in zip(nodes[:-1],nodes[1:])]
    candidate,cm,cfolder=load_candidate('richardson-16384-base')
    for x in [*main,*nodes,*node_base]:same_grid(candidate,x)
    for f in FIELDS:assert np.array_equal(candidate[f],rmain[-1][f])
    reference={k:rnode[-1][k].copy() for k in ['time_s','radius_m',*FIELDS]}
    prefix_record=None
    if ultra_prefix:
        u8,ix=load_ultra_prefix(8192,nodes[2]);u16,ix16=load_ultra_prefix(16384,nodes[3])
        same_grid(u8,u16)
        assert np.array_equal(ix,ix16)
        for n,x in [(8192,u8),(16384,u16)]:sources[f'prefix-N{n}-ultra-3900']=x
        ur=extrapolate(u8,u16)
        for f in FIELDS:reference[f][ix]=ur[f]
        prefix_record=dict(scope='Independent reference only on exact common 0..3900s output coordinates; candidate remains unchanged.',
            prefix_time_count=len(ix),integer_time_count=3901,matched_full_time_indices=ix.tolist(),
            full_prefix_points_kept_on_previous_reference=candidate['time_s'][(candidate['time_s']<=3900)&~np.isin(candidate['time_s'],u8['time_s'])].tolist(),
            change_policy='Apply both ultra reference grids uniformly over their whole common prefix. No isolated value/tail replacement. No state is transferred to later times.',
            spatial_policy='Retain the old four-level spread and add abs(E_ultra-E_strict), a triangle-inequality allowance; never reduce spatial evidence because time is tighter.',
            time_policy='Within prefix use weighted strict-to-ultra changes of BOTH source levels; outside retain previous full-run base-to-strict evidence.',
            old_screen='analysis/screen-r16-v1/analysis.json; its 1 unresolved T value and executed source are preserved unchanged',
            reference_initial_sampling_checks={'8192':u8['_initial_sampling_check'],'16384':u16['_initial_sampling_check']},
            initial_check_scope='Original nodal states are exactly uniform by the frozen initializer. Off-grid output is checked bitwise against the frozen sampler of those uniform nodes, with its roundoff retained. Authority initial outputs remain exactly28/2.55.')
    integer=(candidate['time_s']>=1)&(candidate['time_s']==np.floor(candidate['time_s']))
    assert integer.sum()==10800
    temporal=None
    if not screen:
        b8=mainrun('time-bdf-8192');b16=mainrun('time-bdf-16384')
        rs8=mainrun('time-start-radau-8192')
        rbdf=extrapolate(b8,b16)
        temporal=dict(same_8192_radau_to_BDF=compare(main[2],b8),
            same_16384_radau_to_BDF=compare(main[3],b16),
            same_8192_stricter_radau_and_smaller_first_step=compare(main[2],rs8),
            both_source_levels_Radau_to_BDF_extrapolates=compare(candidate,rbdf),
            startup_scope='The stricter Radau run simultaneously tightens tolerance and reduces every segment first step by 100; it is a joint perturbation, not an isolated startup-error measurement. Unchanged short-time separate-first-step evidence is retained.')
    arrays=dict(time_s=candidate['time_s'],radius_m=candidate['radius_m'])
    records={};flags=[];paper=[];margins=[]
    for field in FIELDS:
        sm=level_spread(rmain,field);sn=level_spread(rnode,field)
        sn_original=sn.copy();reference_migration=np.zeros_like(sn)
        nt=(4*abs(nodes[3][field]-node_base[1][field])+abs(nodes[2][field]-node_base[0][field]))/3
        floor=1e-10 if field=='T_C' else 1e-11
        if ultra_prefix:
            sn=sn.copy();nt=nt.copy()
            reference_migration[ix]=abs(ur[field]-rnode[-1][field][ix])
            sn+=reference_migration
            nt[ix]=(4*abs(u16[field]-nodes[3][field][ix])+abs(u8[field]-nodes[2][field][ix]))/3
        disagreement=abs(candidate[field]-reference[field])
        routeB=disagreement+2*sn+2*nt+floor
        if not screen:
            mt=(4*abs(main[3][field]-b16[field])+np.maximum(abs(main[2][field]-b8[field]),
                         abs(main[2][field]-rs8[field])))/3
            routeA=2*sm+2*mt+floor
            envelope=np.maximum(routeA,routeB)
            arrays[field+'_route_A']=routeA;arrays[field+'_weighted_main_time_changes']=mt
        else:
            mt=None;envelope=routeB
        dist,boundary=half_distances(candidate[field])
        marked=(dist<=envelope)&integer[:,None]
        # Agreement alone cannot pass the test; disagreements are separately fatal.
        rounding_disagrees=np.zeros_like(marked)
        for checkref in [rmain[1],rnode[-1],reference]:
            rounding_disagrees|=(rounded(candidate[field])!=rounded(checkref[field]))&integer[:,None]
        if not screen:
            rounding_disagrees|=(rounded(candidate[field])!=rounded(rbdf[field]))&integer[:,None]
        marked|=rounding_disagrees
        arrays[field+'_route_B']=routeB
        arrays[field+'_engineering_envelope']=envelope
        arrays[field+'_exact_decimal_half_distance']=dist
        arrays[field+'_main_extrapolate_spread']=sm;arrays[field+'_node_extrapolate_spread']=sn
        arrays[field+'_node_original_four_level_spread']=sn_original
        arrays[field+'_reference_migration_allowance']=reference_migration
        arrays[field+'_selected_independent_reference']=reference[field]
        arrays[field+'_weighted_reference_time_changes']=nt
        arrays[field+'_independent_extrapolate_difference']=disagreement
        arrays[field+'_unresolved']=marked
        records[field]=dict(target=1e-5,engineering_envelope_max=maximum(envelope,candidate),
            integer_envelope_max=maximum(envelope,candidate,integer),
            main_extrapolate_spread_max=maximum(sm,candidate),node_extrapolate_spread_max=maximum(sn,candidate),
            reference_migration_allowance_max=maximum(reference_migration,candidate),
            reference_spatial_label='Original four-level spatial evidence plus reference migration allowance; not a pure disjoint spatial error',
            weighted_reference_time_changes_max=maximum(nt,candidate),
            weighted_main_time_changes_max=maximum(mt,candidate) if mt is not None else None,
            independent_difference_max=maximum(disagreement,candidate),roundoff_floor=floor,
            unresolved_rounding_count=int(marked.sum()),rounding_disagreement_count=int(rounding_disagrees.sum()))
        for it,ir in zip(*np.where(marked)):
            flags.append(dict(field=field,time_s=float(candidate['time_s'][it]),radius_m=float(candidate['radius_m'][ir]),
                candidate=float(candidate[field][it,ir]),half_boundary=float(boundary[it,ir]),
                half_distance=float(dist[it,ir]),engineering_envelope=float(envelope[it,ir]),
                route_A=float(routeA[it,ir]) if not screen else None,
                route_B=float(routeB[it,ir]),reference=float(reference[field][it,ir]),
                reference_space_spread=float(sn[it,ir]),weighted_reference_time=float(nt[it,ir]),
                status='UNRESOLVED'))
        for tt in [1800,3600,5400,7200,9000,10800]:
            it=int(np.flatnonzero(candidate['time_s']==tt)[0])
            for ir in [0,5,10,15,20]:
                paper.append(dict(field=field,time_s=tt,radius_m=float(candidate['radius_m'][ir]),
                    full_precision=float(candidate[field][it,ir]),rounded4=float(rounded(candidate[field][it:it+1,ir:ir+1])[0,0]),
                    engineering_envelope=float(envelope[it,ir]),half_distance=float(dist[it,ir]),
                    numerical_evidence_supports_rounding=bool(not marked[it,ir])))
        flat=np.where(integer[:,None],dist-envelope,np.inf)
        for index in np.argsort(flat.ravel())[:15]:
            it,ir=np.unravel_index(index,flat.shape)
            margins.append(dict(field=field,time_s=float(candidate['time_s'][it]),radius_m=float(candidate['radius_m'][ir]),
                half_distance=float(dist[it,ir]),envelope=float(envelope[it,ir]),clearance=float(flat[it,ir])))
    error_pass=all(r['engineering_envelope_max']['value']<r['target'] for r in records.values())
    rawconv=convergence(main[1],main[2],main[3])
    leading_order_supported=all(1.8<rawconv[f]['norm_order']<2.2 for f in FIELDS)
    passed=not screen and error_pass and leading_order_supported and not flags
    report=dict(question='Q2',numerical_pass=passed,screen_only=screen,
        status='ENGINEERING_NUMERICAL_ACCEPTANCE_PASS' if passed else 'CANDIDATE_REVIEW_NOT_ACCEPTED',
        candidate_sha256=cm['solution_sha256'],accepted_candidate_sha256=cm['solution_sha256'] if passed else None,
        script_sha256=sha256(__file__),candidate_description_sha256=sha256(cfolder/'candidate.json'),
        integer_values_scanned=453600,paper_values_checked=60,
        complete_horizon_s=[0,10800],engineering_targets={'T_C':1e-5,'C_kg_kg':1e-5},
        leading_second_order_norm_supported=leading_order_supported,raw_main_convergence=rawconv,
        main_extrapolate_levels=[compare(a,b) for a,b in zip(rmain[:-1],rmain[1:])],
        node_extrapolate_levels=[compare(a,b) for a,b in zip(rnode[:-1],rnode[1:])],
        independent_extrapolate_comparison=compare(candidate,reference),
        independent_full_strict_comparison=compare(candidate,rnode[-1]),reference_prefix_refinement=prefix_record,time_checks=temporal,
        error_evidence=records,unresolved_rounding_count=len(flags),unresolved_points=flags,
        smallest_rounding_clearances=margins,
        evidence_formula=dict(spatial_spread='max(abs(E16-E8), abs(E8-E4)); no division by15 and no global fourth-order assertion',
            time_main='(4*abs(main16-BDF16)+max(abs(main8-BDF8),abs(main8-strict_Radau8)))/3',
            time_reference='(4*abs(node16_strict-node16_base)+abs(node8_strict-node8_base))/3',
            prefix_override='If present: S_new=S_old+abs(E_ultra-E_strict); e_time=(4*abs(node16_ultra-node16_strict)+abs(node8_ultra-node8_strict))/3, at every exact common prefix output. No factor/floor change.',
            route_A='2*main_spread+2*weighted_main_time_changes+floor',
            route_B='abs(candidate-selected_reference(t,r))+2*(original_node_spread+reference_migration_allowance)+2*weighted_reference_time_changes+floor; selected reference follows the prefix override if present',
            combination='max(A,B); the two routes are not added. Route B does not add main time errors again.',
            interpretation='Observed grid/extrapolate and time changes are not statistically independent error components. The factors2 and floors are engineering margins; the spread may contain temporal noise, so this is an empirical envelope, not an exact disjoint partition or rigorous bound.'),
        limitations=['Numerical evidence only for the approved equations and prescribed piecewise-linear input, not physical validation or confidence intervals.',
            'The 0.001s grid pressure failure remains outside the 1..10800s output acceptance interval; early-time propagation is covered by full trajectories and the recorded temporal perturbations.',
            'Original FV states have their own integral ledgers. The uniform extrapolated point array is not a new exactly conserved nonlinear FV state or a continuation checkpoint.',
            'Local reversals and phase-dependent higher terms are retained in the four-level reports; the whole field is not asserted fourth order.',
            'Four-decimal display has quantization up to0.00005 in the relevant unit; it does not preserve the full-precision envelope.'],
        source_runs={label:dict(solution=x['_source'],sha256=sha256(ROOT/x['_source']),
            run_sha256=sha256((ROOT/x['_source']).parent/'run.json')) for label,x in sources.items()})
    dest=BASE/'analysis'/name;dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'pointwise-evidence.npz',**arrays)
    report['pointwise_evidence_sha256']=sha256(dest/'pointwise-evidence.npz')
    report['pointwise_evidence_path']=str(dest/'pointwise-evidence.npz')
    for fn,rows in [('unresolved.csv',flags),('table60-evidence.csv',paper)]:
        with (dest/fn).open('w',encoding='utf-8',newline='') as stream:
            keys=list(rows[0]) if rows else ['field','time_s','radius_m','status']
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    (dest/'analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=serial),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],error_evidence=records,
        unresolved_rounding_count=len(flags),unresolved_points=flags,report=str(dest/'analysis.json')),indent=2,default=serial))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--screen-only',action='store_true')
    p.add_argument('--ultra-prefix',action='store_true')
    a=p.parse_args();review(a.name,a.screen_only,a.ultra_prefix)

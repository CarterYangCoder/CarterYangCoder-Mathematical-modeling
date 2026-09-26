"""Full Q2 independent-node reference driver; frozen spatial module reused.

Only this orchestration layer is new. Geometry, point unknowns, midpoint
constitutive fluxes, dynamic surface storage and analytic Jacobian are read-only
imports from the hash-locked short reference. SciPy BDF remains a shared library.
No formal Excel/authority export occurs here. Completed 60 s segments have full
state and integral checkpoints, permitting exact planned BDF restarts.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import BDF
from threadpoolctl import threadpool_limits, threadpool_info

ROOT=Path(__file__).resolve().parents[1]
OUT_ROOT=ROOT/'records/q2-full-20260911/node-reference'
FROZEN=ROOT/'src/q2_node_reference.py'
FROZEN_SHA='dd622951fd2f7a1d354221fecaffd2fc3d90a5e827a5a93fc87e3a769ef14ded'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if sha(FROZEN)!=FROZEN_SHA:
    raise RuntimeError('Frozen short node reference hash mismatch; no computation permitted.')
import q2_node_reference as core


def json_save(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def npz_save(path,**arrays):
    path=Path(path);tmp=path.with_name(path.name+'.writing')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    os.replace(tmp,path)


def default_grid(stop_s=10800.):
    """Integers plus pressure times and every true 60 s knot +/-0.1 s."""
    if not 0 < stop_s <= 10800:
        raise ValueError('Authorized complete Q2 interval is at most10800 s.')
    # Match the root plan: +/-0.1 around INTERNAL60 s knots; endpoint is saved
    # exactly, without inventing an additional final-endpoint neighborhood.
    knots=np.arange(60.,stop_s,60.)
    times=np.unique(np.r_[np.arange(0.,np.floor(stop_s)+1),stop_s,
                          [.001,.01,.1,.25,.5],knots-.1,knots+.1])
    times=times[(times>=0)&(times<=stop_s)]
    return times,np.arange(21,dtype=float)*.001


def output_hash(times,radii):
    return hashlib.sha256(np.asarray(times,dtype='<f8').tobytes()+np.asarray(radii,dtype='<f8').tobytes()).hexdigest()


def runtime_record():
    return {'python_executable':sys.executable,'python':sys.version,
            'packages':{key:version(key) for key in ['numpy','scipy','openpyxl','threadpoolctl']}}


def normalized_settings(n,rtol,atol_T,atol_C,max_step,stop_s):
    return {'n_intervals':int(n),'n_nodes':int(n)+1,'rtol':float(rtol),
            'atol_T':float(atol_T),'atol_C':float(atol_C),'max_step':float(max_step),
            'stop_s':float(stop_s),'R_m':core.RADIUS,'L_m':core.LENGTH,
            'initial_T_C':core.T_INITIAL,'initial_C_kg_kg':core.C_INITIAL,
            'method':'SciPy BDF','threads':1,
            'first_step_rule':'min(max_step,segment_length,.01*dr^2/max(k/W,D),.1*Euler_positivity_time)'}


def _update_ranges(stats,T,C):
    W,k,D,*_=core.properties(T,C)
    quantities={'T_C':T,'C_kg_kg':C,'W_J_m3K':W,'k_W_mK':k,'D_m2_s':D,'alpha_m2_s':k/W}
    for name,arr in quantities.items():
        low,high=float(np.min(arr)),float(np.max(arr))
        old=stats['accepted_state_ranges'].setdefault(name,[low,high])
        old[0]=min(old[0],low);old[1]=max(old[1],high)
    if not (np.all(np.isfinite(T)) and np.all(np.isfinite(C))
            and np.min(C)>0 and np.min(T)>-273.15 and np.min(W)>0
            and np.min(k)>0 and np.min(D)>0):
        raise FloatingPointError('Nonphysical accepted state/coefficient, no clipping.')


def _balance(model,y,ledger):
    mi,qi,chain,algC,algT=ledger
    W=core.properties(y[0::2],y[1::2])[0]
    dmass=float(np.dot(model.v,y[1::2]-core.C_INITIAL))
    energy=float(np.dot(model.v,W*(y[0::2]-core.T_INITIAL)))
    factor=2*np.pi*core.LENGTH
    return {'time_integrated_C_input_m3_kg_kg':factor*mi,
            'C_integral_change_m3_kg_kg':factor*dmass,
            'C_balance_defect_m3_kg_kg':factor*(dmass-mi),
            'C_balance_normalized_initial':(dmass-mi)/(float(np.sum(model.v))*core.C_INITIAL),
            'Qin_J':factor*qi,'Eref_change_J':factor*energy,'chain_correction_J':factor*chain,
            'Eref_chain_defect_J':factor*(energy-qi-chain),
            'max_algebraic_C_rhs_residual_reduced_units':float(algC),
            'max_algebraic_T_rhs_residual_reduced_units':float(algT),
            'meaning':'C balance has units m3*(kg/kg), not kg water; heat is effective PDE chain balance, not full physical enthalpy. No spatial accuracy follows from cancellation.'}


def run_full_node(n,times=None,radii=None,*,rtol=1e-12,atol_T=1e-12,
                  atol_C=1e-14,max_step=.25,stop_s=10800.,out_dir,
                  resume_checkpoint=None,wall_limit_s=1800.,progress_every_s=1800.,
                  segment_wall_limit_s=600.):
    """Run a reference and return full precision arrays; checkpoint-bound resume.

    A wall limit pauses only after a completed segment; the latest full state
    checkpoint is recoverable. Its fields never masquerade as a complete run.
    No 21-point output is used to reconstruct the dynamic state.
    """
    cfg=json.loads((ROOT/'project_config.json').read_text(encoding='utf-8-sig'))
    expected=os.path.expandvars(cfg['python'].replace('${USERPROFILE}',os.environ['USERPROFILE']))
    if os.path.normcase(str(Path(expected).resolve()))!=os.path.normcase(str(Path(sys.executable).resolve())):
        raise RuntimeError('Use the project-configured interpreter.')
    if sha(FROZEN)!=FROZEN_SHA: raise RuntimeError('Frozen node source changed.')
    if not 0 < stop_s <= 10800: raise ValueError('Unsupported problem2 endpoint.')
    if n<4 or int(n)!=n or any(x<=0 for x in [rtol,atol_T,atol_C,max_step,wall_limit_s,segment_wall_limit_s,progress_every_s]):
        raise ValueError('Invalid numerical settings.')
    settings=normalized_settings(n,rtol,atol_T,atol_C,max_step,stop_s)
    default_t,default_r=default_grid(stop_s)
    times=np.asarray(default_t if times is None else times,dtype=float)
    radii=np.asarray(default_r if radii is None else radii,dtype=float)
    if (times.ndim!=1 or not len(times) or not np.all(np.isfinite(times))
            or np.any(np.diff(times)<=0) or times[0]<0 or times[-1]>stop_s):
        raise ValueError('Output times must strictly increase within the run interval.')
    if (radii.ndim!=1 or not len(radii) or not np.all(np.isfinite(radii))
            or np.any(np.diff(radii)<=0) or radii[0]<0 or radii[-1]>core.RADIUS):
        raise ValueError('Output radii must strictly increase inside[0,R].')
    grid_hash=output_hash(times,radii)
    out=Path(out_dir).resolve()
    if not out.is_relative_to(OUT_ROOT.resolve()):
        raise ValueError('All new node-reference output must remain in its assigned records subtree.')
    out.mkdir(parents=True,exist_ok=False)
    (out/'segments').mkdir();(out/'checkpoints').mkdir()
    drive=core.OriginalDrive.read()
    model=core.NodeModel(int(n),drive)
    sampler=core.point_sampling_matrix(model.r,radii)
    hashes={'frozen_node_source':FROZEN_SHA,'full_driver_source':sha(__file__),
            'environment':drive.source_sha256,'output_grid':grid_hash}
    runtime=runtime_record()
    manifest={'status':'RUNNING','created_utc':datetime.now(timezone.utc).isoformat(),
              'settings':settings,'hashes':hashes,'runtime':runtime,
              'command':[sys.executable,*sys.argv],'wall_limit_s':wall_limit_s,
              'segment_wall_limit_s':segment_wall_limit_s,
              'timing_context':'At most two independent single-thread processes may share machine load during the full stage; wall time is not exclusive performance evidence.',
              'is_independent_software_stack':False,
              'independence':'Independent node CV, direct midpoint flux and dynamic surface; SciPy BDF library shared.',
              'output_count':len(times),'radius_count':len(radii),'formal_integer_count':int(np.sum((times>=1)&(times==np.rint(times)))),
              'resume_checkpoint':str(resume_checkpoint) if resume_checkpoint else None,
              'original_initial_state_used':True,'results_are_reference_not_authority':True}
    json_save(out/'run.json',manifest)
    fields=np.empty((len(times),2,len(radii)))
    y=np.tile([core.T_INITIAL,core.C_INITIAL],int(n)+1)
    ledger=np.zeros(5,dtype=float)
    cursor=0;begin_at=0.;chunks=[]
    stats={'accepted_steps':0,'segments':[],'accepted_state_ranges':{},
           'internal_rejected_steps':None,'rejection_note':'BDF internal rejected/Newton attempts not exposed.',
           'max_step_actual_s':0.,'min_step_s':None,'clipping_performed':False,
           'output_time_method':'Unchanged SciPy BDF dense polynomial; paired temporal tests required.',
           'point_semantics':'N+1 true point unknowns; nodal-control-volume mass lumping; local quadratic r2 point interpolation.',
           'surface_semantics':'Dynamic endpoint with finite half-ring storage, not exact finite-grid point-derivative Robin.',
           'failed_segments':[]}
    if times[0]==0:
        fields[0,0]=sampler@y[0::2];fields[0,1]=sampler@y[1::2];cursor=1
    if resume_checkpoint is not None:
        cp_path=Path(resume_checkpoint).resolve()
        if not cp_path.is_relative_to(OUT_ROOT.resolve()): raise ValueError('Foreign resume checkpoint.')
        cp=json.loads(cp_path.read_text(encoding='utf-8'))
        if cp['settings']!=settings or cp['hashes']!=hashes or cp['runtime']!=runtime:
            raise ValueError('Resume configuration, code, grid or runtime version differs.')
        data_path=cp_path.parent/cp['state_file']
        if sha(data_path)!=cp['state_sha256']: raise ValueError('Checkpoint state hash mismatch.')
        with np.load(data_path) as state:
            y=state['full_point_state'].copy();ledger=state['ledger'].copy()
            if not np.array_equal(state['node_radii_m'],model.r): raise ValueError('Checkpoint geometry mismatch.')
        if y.shape!=(2*(int(n)+1),) or ledger.shape!=(5,): raise ValueError('Checkpoint state/ledger shape mismatch.')
        begin_at=float(cp['time_s']);cursor=int(cp['output_count']);stats=cp['stats']
        if begin_at!=stop_s and not np.any(drive.times==begin_at): raise ValueError('Resume must be a planned environment-segment boundary.')
        chunks=cp['completed_chunks']
        covered=np.zeros(cursor,dtype=bool)
        for item in chunks:
            path=ROOT/item['path']
            if sha(path)!=item['sha256']: raise ValueError('Resume output segment hash mismatch.')
            with np.load(path) as seg:
                lo,hi=item['output_start'],item['output_end']
                if not np.array_equal(seg['time_s'],times[lo:hi]) or np.any(covered[lo:hi]):
                    raise ValueError('Repeated, reordered or altered output prefix.')
                fields[lo:hi,0]=seg['T_C'];fields[lo:hi,1]=seg['C_kg_kg'];covered[lo:hi]=True
        if not np.all(covered): raise ValueError('Output prefix incomplete; never infer it from21 points.')
    _update_ranges(stats,y[0::2],y[1::2])
    bounds=np.unique(np.r_[0.,drive.times,stop_s])
    bounds=bounds[(bounds>=begin_at)&(bounds<=stop_s)]
    atol=np.tile([atol_T,atol_C],int(n)+1)
    qx,qw=leggauss(3)
    started=time.perf_counter();last_notice=begin_at;last_checkpoint=None
    status='RUNNING'
    try:
        with threadpool_limits(limits=1):
            manifest['threadpools']=threadpool_info();json_save(out/'run.json',manifest)
            for lo,hi in zip(bounds[:-1],bounds[1:]):
                clock0=time.perf_counter();first_output=0 if lo==0 else cursor
                # Preserve exact short-driver operation order and formulas.
                p0=core.properties(y[0::2],y[1::2],model.constant)
                fastest=max(float(np.max(p0[1]/p0[0])),float(np.max(p0[2])))
                first_step=min(max_step,hi-lo,.01*model.h**2/fastest)
                f0=model.rhs(lo,y)
                for val,der in [(y[1::2],f0[1::2]),(y[0::2]+273.15,f0[0::2])]:
                    declining=der<0
                    if np.any(declining):
                        first_step=min(first_step,.1*float(np.min(val[declining]/(-der[declining]))))
                solver=BDF(model.rhs,lo,y,hi,rtol=rtol,atol=atol,first_step=first_step,
                           max_step=max_step,jac=model.jac,vectorized=False)
                count=0
                while solver.status=='running':
                    if time.perf_counter()-clock0>segment_wall_limit_s:
                        raise TimeoutError(f'Environment segment[{lo},{hi}] exceeded{segment_wall_limit_s}s; last complete segment checkpoint retained.')
                    old_t=solver.t;message=solver.step()
                    if solver.status=='failed': raise RuntimeError(f'BDF failure at{solver.t}: {message}')
                    dt=solver.t-old_t
                    if dt<=0: raise RuntimeError('BDF did not advance.')
                    stats['min_step_s']=dt if stats['min_step_s'] is None else min(stats['min_step_s'],dt)
                    stats['max_step_actual_s']=max(stats['max_step_actual_s'],dt)
                    count+=1;dense=solver.dense_output()
                    while cursor<len(times) and times[cursor]<=solver.t:
                        yy=solver.y if times[cursor]==solver.t else dense(times[cursor])
                        fields[cursor,0]=sampler@yy[0::2];fields[cursor,1]=sampler@yy[1::2];cursor+=1
                    _update_ranges(stats,solver.y[0::2],solver.y[1::2])
                    for gx,gw in zip(qx,qw):
                        tq=(old_t+solver.t)/2+dt*gx/2;Y=dense(tq)
                        TT,CC,WW,WWc,_,_,_,_,_,dTT,dCC,FT,FC,srcT,srcC=model.values(tq,Y)
                        mass_rhs=-FC+np.dot(model.v,srcC)
                        heat_rhs=-FT+np.dot(model.v,srcT)
                        ledger[0]+=dt*gw*mass_rhs/2
                        ledger[1]+=dt*gw*heat_rhs/2
                        ledger[2]+=dt*gw*np.dot(model.v,WWc*(TT-core.T_INITIAL)*dCC)/2
                        ledger[3]=max(ledger[3],abs(float(np.dot(model.v,dCC)-mass_rhs)))
                        ledger[4]=max(ledger[4],abs(float(np.dot(model.v*WW,dTT)-heat_rhs)))
                    if time.perf_counter()-clock0>segment_wall_limit_s:
                        raise TimeoutError(f'Environment segment[{lo},{hi}] exceeded{segment_wall_limit_s}s; last complete segment checkpoint retained.')
                y=solver.y.copy();stats['accepted_steps']+=count
                segstat={'start_s':float(lo),'end_s':float(hi),'accepted_steps':count,
                         'nfev':solver.nfev,'njev':solver.njev,'nlu':solver.nlu,
                         'requested_first_step_s':float(first_step),'wall_seconds':time.perf_counter()-clock0,
                         'endpoint_reached':bool(solver.t==hi),'balance':_balance(model,y,ledger)}
                stats['segments'].append(segstat)
                if sha(FROZEN)!=hashes['frozen_node_source'] or sha(__file__)!=hashes['full_driver_source']:
                    raise RuntimeError('A participating source changed during integration.')
                tag=f'{len(stats["segments"]):04d}-{hi:.6f}'
                segfile=out/'segments'/f'{tag}.npz'
                npz_save(segfile,time_s=times[first_output:cursor],radius_m=radii,
                         T_C=fields[first_output:cursor,0],C_kg_kg=fields[first_output:cursor,1])
                chunks.append({'path':str(segfile.relative_to(ROOT)),'sha256':sha(segfile),
                               'output_start':first_output,'output_end':cursor})
                statefile=out/'checkpoints'/f'{tag}.npz'
                npz_save(statefile,time_s=np.array(hi),full_point_state=y,node_radii_m=model.r,
                         ledger=ledger,output_count=np.array(cursor))
                cp={'time_s':float(hi),'state_file':statefile.name,'state_sha256':sha(statefile),
                    'settings':settings,'hashes':hashes,'runtime':runtime,'output_count':cursor,
                    'completed_chunks':chunks,'stats':stats,
                    'resume_semantics':'Full point state at planned BDF restart; no hidden BDF history is required at this segment boundary.'}
                last_checkpoint=statefile.with_suffix('.json');json_save(last_checkpoint,cp)
                json_save(out/'progress.json',{'status':'RUNNING','through_s':float(hi),
                    'output_count':cursor,'accepted_steps':stats['accepted_steps'],
                    'wall_seconds':time.perf_counter()-started,'latest_checkpoint':str(last_checkpoint.relative_to(ROOT))})
                if hi-last_notice>=progress_every_s or hi==stop_s:
                    print(json.dumps({'node_reference_n':n,'through_s':float(hi),
                        'stop_s':stop_s,'accepted_steps':stats['accepted_steps'],
                        'wall_seconds':time.perf_counter()-started}),flush=True);last_notice=hi
                if time.perf_counter()-started>wall_limit_s and hi<stop_s:
                    status='PAUSED_AT_VERIFIED_SEGMENT_CHECKPOINT';break
            else:
                status='COMPUTATION_COMPLETE_REVIEW_PENDING'
            manifest['threadpools_after']=threadpool_info()
        if status=='COMPUTATION_COMPLETE_REVIEW_PENDING' and cursor!=len(times):
            raise RuntimeError('Missing requested output; not complete.')
        if not np.all(np.isfinite(fields[:cursor])): raise RuntimeError('Nonfinite saved outputs.')
        if sha(ROOT/'A题/附件/附件1.xlsx')!=hashes['environment']: raise RuntimeError('Environment version changed.')
        solution={'time_s':times[:cursor],'radius_m':radii,'T_C':fields[:cursor,0],
                  'C_kg_kg':fields[:cursor,1],'final_point_state':y,'node_radii_m':model.r,'ledger':ledger}
        filename='solution.npz' if status=='COMPUTATION_COMPLETE_REVIEW_PENDING' else 'partial-solution.npz'
        npz_save(out/filename,**solution)
        manifest.update({'status':status,'wall_seconds':time.perf_counter()-started,
                         'completed_through_s':float(bounds[-1]) if status=='COMPUTATION_COMPLETE_REVIEW_PENDING' else float(hi),
                         'accepted_steps':stats['accepted_steps'],'result_file':filename,
                         'result_sha256':sha(out/filename),'latest_checkpoint':str(last_checkpoint.relative_to(ROOT)) if last_checkpoint else None,
                         'final_integral_audit':_balance(model,y,ledger)})
        json_save(out/'stats.json',stats);json_save(out/'run.json',manifest)
        return {**solution,'stats':stats,'run':manifest}
    except Exception as exc:
        manifest.update({'status':'FAILED_PARTIAL_NOT_A_RESULT','error':str(exc),'error_type':type(exc).__name__,
                         'wall_seconds':time.perf_counter()-started,
                         'last_valid_checkpoint':str(last_checkpoint.relative_to(ROOT)) if last_checkpoint else None,
                         'last_valid_output_count':cursor})
        json_save(out/'failure.json',manifest);json_save(out/'run.json',manifest)
        raise


def replay_short(label):
    olddir=ROOT/'records/q2-stage1-20260911/node-reference/startup-fixed-v3/real-N16384'
    with np.load(olddir/'solution.npz') as z:
        old={k:z[k].copy() for k in z.files}
    result=run_full_node(16384,old['times'],old['radii'],rtol=1e-12,atol_T=1e-12,
        atol_C=1e-14,max_step=.25,stop_s=121,out_dir=OUT_ROOT/label,
        wall_limit_s=120.,progress_every_s=60.)
    evidence={'scope':'Full driver replay of frozen16384-node short reference only',
              'frozen_source_sha256':FROZEN_SHA,'new_driver_source_sha256':sha(__file__),
              'frozen_result_sha256':sha(olddir/'solution.npz'),
              'new_result_sha256':result['run']['result_sha256'],'comparisons':{}}
    for key in ['T_C','C_kg_kg','final_point_state']:
        delta=result[key]-old[key]
        evidence['comparisons'][key]={'max_abs':float(np.max(np.abs(delta))),
                                     'array_equal':bool(np.array_equal(result[key],old[key]))}
        if key!='final_point_state':
            evidence['comparisons'][key]['four_decimal_differences']=int(np.count_nonzero(np.round(result[key],4)!=np.round(old[key],4)))
    evidence['accepted_steps_old']=json.loads((olddir/'stats.json').read_text(encoding='utf-8'))['accepted_steps']
    evidence['accepted_steps_new']=result['stats']['accepted_steps']
    evidence['unchanged_frozen_source']=sha(FROZEN)==FROZEN_SHA
    evidence['pass']=all(v['max_abs']<=1e-12 for v in evidence['comparisons'].values()) and evidence['unchanged_frozen_source']
    json_save(OUT_ROOT/label/'replay-comparison.json',evidence)
    print(json.dumps(evidence,ensure_ascii=False,indent=2),flush=True)
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,default=4096)
    parser.add_argument('--stop',type=float,default=10800)
    parser.add_argument('--rtol',type=float,default=1e-12)
    parser.add_argument('--atol-T',type=float,default=1e-12)
    parser.add_argument('--atol-C',type=float,default=1e-14)
    parser.add_argument('--max-step',type=float,default=.25)
    parser.add_argument('--label',required=True)
    parser.add_argument('--replay-short',action='store_true')
    parser.add_argument('--resume-checkpoint',type=Path)
    parser.add_argument('--wall-limit',type=float,default=1800.)
    parser.add_argument('--segment-wall-limit',type=float,default=600.)
    parser.add_argument('--progress-every',type=float,default=1800.)
    args=parser.parse_args()
    if Path(args.label).name!=args.label: parser.error('--label must be a directory name.')
    if args.replay_short:
        replay_short(args.label)
    else:
        result=run_full_node(args.n,rtol=args.rtol,atol_T=args.atol_T,atol_C=args.atol_C,
            max_step=args.max_step,stop_s=args.stop,out_dir=OUT_ROOT/args.label,
            resume_checkpoint=args.resume_checkpoint,wall_limit_s=args.wall_limit,
            segment_wall_limit_s=args.segment_wall_limit,progress_every_s=args.progress_every)
        print(json.dumps(result['run'],ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()

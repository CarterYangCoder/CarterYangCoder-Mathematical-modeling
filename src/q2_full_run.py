"""Q2 approved full-horizon driver. Reuse the frozen physical discretization.

Segment files contain complete ring states, passive integrals and point outputs.
No initial condition reset at environment knots; no spatial remapping on resume.
Radau collocation observation is passive and returns the library result unchanged.
"""
from pathlib import Path
import argparse
import inspect
import json
import platform
import sys
import time
import numpy as np
import scipy
from scipy.integrate import Radau, BDF
import scipy.integrate._ivp.radau as radau_module
from threadpoolctl import threadpool_limits, threadpool_info
from q2_model import CoupledFV, read_environment, sha256, harmonic

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'records/q2-full-20260911'


def save_json(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)


def output_grid(stop):
    knots=np.arange(60.,stop,60.)
    times=np.unique(np.r_[0.,.001,.01,.1,.25,.5,np.arange(1.,np.floor(stop)+1),
                          knots-.1,knots,knots+.1,stop])
    return times[times<=stop],np.arange(21)*.001


def tableau():
    nodes=np.array([(4-np.sqrt(6))/10,(4+np.sqrt(6))/10,1.])
    columns=[]
    for j in range(3):
        poly=np.poly1d([1.])
        for k in range(3):
            if k!=j:poly*=np.poly1d([1.,-nodes[k]])/(nodes[j]-nodes[k])
        integral=np.polyint(poly);columns.append(integral(nodes)-integral(0.))
    return nodes,np.array(columns).T


def solve(args):
    cfg=json.loads((ROOT/'config/q2-full.json').read_text(encoding='utf-8'))
    if not cfg['full_horizon_run_authorized'] or not 0<args.stop<=10800:
        raise ValueError('Full Q2 authority or interval mismatch')
    physical=json.loads((ROOT/'config/q2.json').read_text(encoding='utf-8'))
    for rel, expected in cfg['frozen_dependencies'].items():
        if sha256(ROOT/rel)!=expected:raise ValueError(f'Frozen dependency changed: {rel}')
    td,cd,raw=read_environment(ROOT,physical)
    times,radii=output_grid(args.stop)
    n=args.cells;m=CoupledFV(n,flux='fick',root_xtol=args.root_xtol)
    g,p=m.grid,m.props;v=g.volumes;vol=float(v.sum());ar=float(g.areas[-1])
    w0=float(p.thermal(2.55)[0]);weights=g.point_weights(radii)
    signature=dict(settings={k:value for k,value in vars(args).items() if k not in ('resume','label')},
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,blas_threads=1,
        driver_sha256=sha256(__file__),model_sha256=sha256(ROOT/'src/q2_model.py'),
        helper_sha256=sha256(ROOT/'src/q1_model.py'),config_sha256=sha256(ROOT/'config/q2-full.json'),
        physical_config_sha256=sha256(ROOT/'config/q2.json'),environment_sha256=physical['inputs']['environment']['sha256'])
    dest=BASE/'runs'/args.label
    if args.resume:
        record=json.loads((dest/'run.json').read_text(encoding='utf-8'))
        if record['signature']!=signature:raise ValueError('Resume source/settings mismatch')
        if record['status']=='COMPLETED':raise ValueError('Run already completed')
        segments=record['segments']
    else:
        dest.mkdir(parents=True,exist_ok=False)
        record=dict(status='RUNNING',signature=signature,command=sys.argv,interpreter=sys.executable,
            python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            threadpools=threadpool_info(),start_from_original_initial=True,segments=[],
            official_result=False,created_unix=time.time(),active_wall_s=0.)
        segments=[]
    y=np.r_[np.tile([28.,2.55],n),np.zeros(3)]
    T=np.full((len(times),21),np.nan);C=T.copy();T[0]=28.;C[0]=2.55
    exch=np.full((len(times),3),np.nan);exch[0]=0.
    balances=np.full((len(times),2),np.nan);balances[0]=0.
    filled=np.zeros(len(times),bool);filled[0]=True
    now=0.;ranges=[28.,28.,2.55,2.55]
    stats=dict(accepted_steps=0,collocation_calls=0,nonconverged_collocation_calls=0,
        maximum_collocation_iterations=0,stage_residual_T_C=0.,stage_residual_C_kg_kg=0.,
        Gauss_C_gap_m3_C=0.,Gauss_heat_chain_gap_J=0.,maximum_eta_at_outputs=0.,
        property_min=[float('inf')]*4,property_max=[-float('inf')]*4,
        root_max_equivalent_C=0.,root_max_heat_residual_W_m2=0.,root_max_moisture_residual_m_s=0.,root_max_iterations=0)
    stats.update(stage_audit_performed=bool(args.audit),Gauss_integral_audit_performed=bool(args.audit))
    if not args.audit:
        for key in ('stage_residual_T_C','stage_residual_C_kg_kg','Gauss_C_gap_m3_C','Gauss_heat_chain_gap_J'):
            stats[key]=None
    gauss_totals=np.zeros(3)
    if segments:
        for item in segments:
            path=dest/item['file']
            if sha256(path)!=item['sha256']:raise ValueError('Resume segment fingerprint changed')
            with np.load(path) as d:
                if not np.array_equal(d['faces_m'],g.faces):raise ValueError('Resume geometry mismatch')
                ix=np.searchsorted(times,d['time_s'])
                if not np.array_equal(times[ix],d['time_s']):raise ValueError('Resume output indices mismatch')
                T[ix],C[ix],exch[ix],balances[ix]=d['T_C'],d['C_kg_kg'],d['exchanges'],d['balances']
                filled[ix]=True;y=d['state'].copy();now=float(d['end_s']);gauss_totals=d['gauss_totals'].copy()
        stats=record['checkpoint_stats'];ranges=record['checkpoint_ring_ranges']
        audit_path=dest/'accepted-step-audit.csv'
        lines=audit_path.read_text(encoding='utf-8').splitlines(keepends=True)
        retained=[lines[0]]+[line for line in lines[1:] if float(line.split(',')[1])<=now]
        if len(retained)!=len(lines):
            (dest/f'aborted-step-audit-{int(time.time())}.csv').write_text(''.join(lines),encoding='utf-8')
            audit_path.write_text(''.join(retained),encoding='utf-8')
    initial_active=float(record.get('active_wall_s',0.));started=time.perf_counter()
    atol=np.r_[np.tile([args.atol_T,args.atol_C],n),args.atol_C,args.atol_T,args.atol_T]
    alpha=float(p.thermal(2.55)[1])/w0
    first=args.first_step_factor*np.min(np.diff(g.faces))**2/alpha
    nodes,A=tableau();gx,gw=np.polynomial.legendre.leggauss(5)
    original=radau_module.solve_collocation_system
    def observed(*aa,**kk):
        answer=original(*aa,**kk)
        stats['collocation_calls']+=1
        stats['nonconverged_collocation_calls']+=int(not answer[0])
        stats['maximum_collocation_iterations']=max(stats['maximum_collocation_iterations'],int(answer[1]))
        return answer
    if args.method=='Radau':radau_module.solve_collocation_system=observed
    logmode='a' if args.resume else 'w'
    stepfile=(dest/'accepted-step-audit.csv').open(logmode,encoding='utf-8',newline='')
    if not args.resume:stepfile.write('start_s,end_s,dt_s,stage_residual_T_C,stage_residual_C_kg_kg,C_step_gap_m3_C\n')
    try:
        save_json(dest/'run.json',record)
        ends=np.r_[td.times[(td.times>now)&(td.times<args.stop)],args.stop]
        for end in ends:
            segment_start=now;segment_clock=time.perf_counter();before_steps=stats['accepted_steps']
            def rhs(t,z):
                if time.perf_counter()-segment_clock>args.timeout:
                    raise TimeoutError(f'Full Q2 segment timeout {segment_start}--{end} at {t}')
                if initial_active+time.perf_counter()-started>args.run_budget:
                    raise TimeoutError(f'Full Q2 run budget exceeded at {t}')
                return m.augmented(z,td(t),cd(t))
            def jac(t,z):return m.augmented(z,td(t),cd(t),jacobian=True)
            cls=Radau if args.method=='Radau' else BDF
            solver=cls(rhs,now,y,float(end),rtol=args.rtol,atol=atol,jac=jac,
                max_step=args.max_step,first_step=min(first,args.max_step,end-now))
            while solver.status=='running':
                begin=float(solver.t);before=solver.y.copy();solver.step()
                if solver.status=='failed':raise ArithmeticError(f'{args.method} failed at {solver.t}')
                finish=float(solver.t);h=finish-begin
                if not h>0:raise ArithmeticError('Nonpositive accepted step')
                dense=solver.dense_output();stats['accepted_steps']+=1
                st=solver.y[:2*n:2];sc=solver.y[1:2*n:2]
                if not np.all(np.isfinite(solver.y)) or np.any(sc<=0) or np.any(st+273.15<=0):
                    raise ArithmeticError('Nonfinite/nonphysical accepted state')
                ranges=[min(ranges[0],float(st.min())),max(ranges[1],float(st.max())),
                        min(ranges[2],float(sc.min())),max(ranges[3],float(sc.max()))]
                rt=rc=float('nan')
                if args.method=='Radau':
                    stages=before[None,:]+solver.Z
                    if not np.all(np.isfinite(stages)) or np.any(stages[:,1:2*n:2]<=0) or np.any(stages[:,:2*n:2]+273.15<=0):
                        raise ArithmeticError('Nonphysical collocation stage')
                    if args.audit:
                        F=np.array([rhs(begin+h*c,z) for c,z in zip(nodes,stages)])
                        residual=solver.Z-h*A@F
                        rt=float(np.max(abs(residual[:,:2*n:2])));rc=float(np.max(abs(residual[:,1:2*n:2])))
                        stats['stage_residual_T_C']=max(stats['stage_residual_T_C'],rt)
                        stats['stage_residual_C_kg_kg']=max(stats['stage_residual_C_kg_kg'],rc)
                        # Independent quadrature of surface fluxes and the dense
                        # polynomial chain term. Not the passive ODE integrals.
                        xs=(gx+1)/2;qtimes=begin+h*xs;zs=dense(qtimes).T
                        dz=(dense.Q@np.array([np.ones(5),2*xs,3*xs**2])/h).T
                        for qt,qw,z,dd in zip(qtimes,gw,zs,dz):
                            sf=m.surface(z[-5],z[-4],td(qt),cd(qt))
                            wp=p.thermal(z[1:2*n:2])[2]
                            gauss_totals+=h*qw/2*np.array([ar*sf.flux_C,-ar*sf.flux_T,
                                (v*wp*(z[:2*n:2]-28))@dd[1:2*n:2]])
                        cg=float((sc-2.55)@v+gauss_totals[0]);ww=p.thermal(sc)[0]
                        hg=float((v*ww)@(st-28)-gauss_totals[1]-gauss_totals[2])
                        stats['Gauss_C_gap_m3_C']=max(stats['Gauss_C_gap_m3_C'],abs(cg))
                        stats['Gauss_heat_chain_gap_J']=max(stats['Gauss_heat_chain_gap_J'],abs(hg))
                step_gap=float((sc-before[1:2*n:2])@v+vol*(solver.y[-3]-before[-3]))
                stepfile.write(f'{begin:.17g},{finish:.17g},{h:.17g},{rt:.17g},{rc:.17g},{step_gap:.17g}\n')
                ix=np.flatnonzero((times>begin)&(times<=finish))
                if len(ix):
                    z=dense(times[ix]).T
                    if np.any(filled[ix]) or np.any(z[:,1:2*n:2]<=0) or not np.all(np.isfinite(z)):
                        raise ArithmeticError('Repeated or nonphysical output state')
                    T[ix]=z[:,:2*n:2]@weights.T;C[ix]=z[:,1:2*n:2]@weights.T
                    for out_i,zz in zip(ix,z):
                        ta,ce=td(times[out_i]),cd(times[out_i]);sf=m.surface(zz[-5],zz[-4],ta,ce)
                        T[out_i,-1],C[out_i,-1]=sf.T,sf.C
                        ki=float(p.thermal(zz[-4])[1]);_,km,_,kp,_=p.thermal(min(zz[-4],ce))
                        kmn=harmonic(ki,km);dtb=g.delta*m.hT*abs(zz[-5]-ta)*2*ki**2*kp/(ki+km)**2/(kmn+g.delta*m.hT)**2
                        eta=float(3850/(min(zz[-5],ta)+273.15)**2*dtb*abs(zz[-4]-ce))
                        stats['maximum_eta_at_outputs']=max(stats['maximum_eta_at_outputs'],eta)
                    exch[ix]=z[:,-3:]
                    balances[ix,0]=(z[:,1:2*n:2]-2.55)@v+vol*z[:,-3]
                    ww=p.thermal(z[:,1:2*n:2])[0]
                    balances[ix,1]=(ww*(z[:,:2*n:2]-28))@v-vol*w0*(z[:,-2]+z[:,-1])
                    filled[ix]=True
            if solver.status!='finished' or solver.t!=end:
                raise ArithmeticError('Did not reach exact environment knot')
            y=solver.y.copy();now=float(end)
            tt,cc=y[:2*n:2],y[1:2*n:2]
            props=[650+128*cc,1450+2736*cc/(1+cc),p.thermal(cc)[1],p.D(cc,tt)]
            for k,value in enumerate(props):
                if np.any(value<=0) or not np.all(np.isfinite(value)):raise ArithmeticError('Invalid material property')
                stats['property_min'][k]=min(stats['property_min'][k],float(value.min()))
                stats['property_max'][k]=max(stats['property_max'][k],float(value.max()))
            for key,value in [('root_max_equivalent_C',m.max_root_equiv),('root_max_heat_residual_W_m2',m.max_heat_residual),
                ('root_max_moisture_residual_m_s',m.max_moisture_residual),('root_max_iterations',m.max_root_iterations)]:
                stats[key]=max(stats[key],value)
            ix=np.flatnonzero((times>segment_start)&(times<=end))
            segment_file=dest/f'segment-{end:011.4f}.npz'
            np.savez_compressed(segment_file,time_s=times[ix],radius_m=radii,T_C=T[ix],C_kg_kg=C[ix],
                exchanges=exch[ix],balances=balances[ix],state=y,end_s=end,faces_m=g.faces,gauss_totals=gauss_totals)
            info=dict(start_s=segment_start,end_s=now,file=segment_file.name,sha256=sha256(segment_file),
                accepted_steps=stats['accepted_steps']-before_steps,nfev=solver.nfev,njev=solver.njev,nlu=solver.nlu,
                wall_s=time.perf_counter()-segment_clock)
            segments.append(info);record.update(status='RUNNING',end_s=now,segments=segments,stats=stats,
                accepted_ring_ranges=ranges,checkpoint_stats=json.loads(json.dumps(stats)),
                checkpoint_ring_ranges=list(ranges),active_wall_s=initial_active+time.perf_counter()-started)
            save_json(dest/'run.json',record);stepfile.flush()
            if end<=120 or end%600==0 or end==args.stop:
                print(json.dumps(dict(label=args.label,end_s=end,wall_s=record['active_wall_s'],steps=stats['accepted_steps'],
                    centre_T=float(T[ix[-1],0]),surface_C=float(C[ix[-1],-1]))),flush=True)
        if now!=args.stop or not np.all(filled) or not np.all(np.isfinite(T+C)):
            raise ArithmeticError('Incomplete final outputs')
        stats.update(maximum_C_balance_m3_C=float(np.max(abs(balances[:,0]))),
            maximum_heat_chain_balance_J=float(np.max(abs(balances[:,1]))),
            normalized_C_balance=float(np.max(abs(balances[:,0]))/(vol*2.55)),
            normalized_heat_chain_balance=float(np.max(abs(balances[:,1]))/(vol*w0)),
            internal_step_rejections=None,
            rejection_note='Collocation attempts/nonconvergence counted for Radau; error-control or BDF rejections are not falsely reported as zero.',
            dense_output='Radau cubic or BDF library polynomial, verified through time refinement and an output-endpoint test')
        np.savez_compressed(dest/'solution.npz',time_s=times,radius_m=radii,T_C=T,C_kg_kg=C,
            exchanges=exch,balances=balances,final_state=y,faces_m=g.faces)
        record.update(status='COMPLETED',end_s=now,stats=stats,active_wall_s=initial_active+time.perf_counter()-started,
            solution_sha256=sha256(dest/'solution.npz'),final_state_complete=True,
            integer_values=int(np.count_nonzero((times>=1)&(times==np.round(times)))*42))
    except BaseException as exc:
        record.update(status='FAILED',error=repr(exc),failure_wall_s=initial_active+time.perf_counter()-started,
            last_complete_knot_s=float(segments[-1]['end_s']) if segments else 0.,stats=stats)
        save_json(dest/'run.json',record)
        raise
    finally:
        radau_module.solve_collocation_system=original;stepfile.close()
    save_json(dest/'run.json',record)
    return record


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cells',type=int,default=8192);ap.add_argument('--method',choices=['Radau','BDF'],default='Radau')
    ap.add_argument('--rtol',type=float,default=1e-11);ap.add_argument('--atol-T',type=float,default=1e-12)
    ap.add_argument('--atol-C',type=float,default=1e-14);ap.add_argument('--max-step',type=float,default=.5)
    ap.add_argument('--first-step-factor',type=float,default=.01);ap.add_argument('--root-xtol',type=float,default=5e-15)
    ap.add_argument('--stop',type=float,default=10800.);ap.add_argument('--audit',action='store_true')
    ap.add_argument('--timeout',type=float,default=600.);ap.add_argument('--run-budget',type=float,default=7200.)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--label',required=True)
    args=ap.parse_args()
    if args.audit and args.method!='Radau':raise ValueError('Collocation audit requires Radau')
    with threadpool_limits(limits=1):
        r=solve(args)
    print(json.dumps(dict(status=r['status'],end_s=r['end_s'],active_wall_s=r['active_wall_s'],label=args.label)),flush=True)


if __name__=='__main__':main()

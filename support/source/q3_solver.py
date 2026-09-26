"""问题三：时变环境下的长时程温湿传递求解器。"""
from __future__ import annotations


from drying_model_core import *


from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import argparse, cProfile, csv, ctypes, hashlib, inspect, io, json, os
import platform, pstats, sys, time, marshal
import numpy as np
import scipy
from scipy.integrate import Radau, BDF
from scipy.sparse.linalg import splu
from scipy.optimize import brentq
import scipy.integrate._ivp.radau as rm
import scipy.integrate._ivp.bdf as bm
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'records/q3-acceleration-20260911'
Q2=ROOT/'records/q2-full-20260911/runs/primary-8192'
_IMPORT_SOURCE_BYTES=Path(__file__).read_bytes()
_IMPORT_SOURCE_SHA=hashlib.sha256(_IMPORT_SOURCE_BYTES).hexdigest()

def execution_source_snapshot(dest):

    compiled=compile(_IMPORT_SOURCE_BYTES,str(Path(__file__)), 'exec',dont_inherit=True)
    objects={c.co_name:c for c in compiled.co_consts if inspect.iscode(c)}
    signatures={}
    for name,func in list(globals().items()):
        if inspect.isfunction(func) and func.__module__==__name__:
            if func.__code__!=objects.get(name):
                raise RuntimeError('Loaded function differs from import snapshot: '+name)
            signatures[name]=hashlib.sha256(marshal.dumps(func.__code__)).hexdigest()
    (dest/'executed-driver.py').write_bytes(_IMPORT_SOURCE_BYTES)
    record={'import_snapshot_sha256':_IMPORT_SOURCE_SHA,'loaded_function_code_matches_snapshot':True,
            'loaded_function_code_sha256':signatures,
            'recorded_before_integration':True,
            'scope':'Own module functions checked against compiled import snapshot; frozen model dependencies separately verified.'}
    save(dest/'execution-source.json',record)
    return record

def save(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    temp.replace(path)

def peak_mb():
    class PMC(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    pm=PMC(); pm.cb=ctypes.sizeof(pm)
    ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    handle=ctypes.windll.kernel32.GetCurrentProcess()
    ok=ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle),ctypes.byref(pm),pm.cb)
    return float(pm.PeakWorkingSetSize/2**20) if ok else None

def frozen():
    full=json.loads((ROOT/'config/q2-full.json').read_text(encoding='utf8'))
    for rel,digest in full['frozen_dependencies'].items():
        if sha256(ROOT/rel)!=digest: raise ValueError('Frozen source mismatch: '+rel)
    return full['frozen_dependencies']

def drives():
    frozen()
    return read_environment(ROOT,json.loads((ROOT/'config/q2.json').read_text(encoding='utf8')))

def historical():
    j=json.loads((Q2/'run.json').read_text(encoding='utf8'))
    a=np.genfromtxt(Q2/'accepted-step-audit.csv',delimiter=',',names=True)
    out={'sources':frozen(),'run_sha256':sha256(Q2/'run.json'),'accepted_steps':len(a),
         'step_quantiles_s':dict(zip(['min','p10','median','p90','max'],map(float,np.quantile(a['dt_s'],[0,.1,.5,.9,1])))),
         'at_cap_count':int(np.count_nonzero(abs(a['dt_s']-.5)<1e-10)),
         'at_cap_duration_s':float(a['dt_s'][abs(a['dt_s']-.5)<1e-10].sum()),
         'nfev':sum(s['nfev'] for s in j['segments']),
         'njev':sum(s['njev'] for s in j['segments']),
         'nlu':sum(s['nlu'] for s in j['segments']),
         'segments':len(j['segments']),'active_wall_s':j['active_wall_s'],
         'rejections':j['stats'].get('internal_step_rejections'),
         'note':'Historical timing includes original audit, file I/O and machine load; not a Q3 runtime.'}
    out['later_segment_first_steps_s']=[float(a['dt_s'][np.flatnonzero(abs(a['start_s']-t)<1e-8)[0]]) for t in [60,3600,7200,10740]]
    td,cd,raw=drives(); out['environment_end']={'t_s':raw[-1,0],'Ta_C':raw[-1,1],'Ce':raw[-1,2]}
    save(BASE/'historical-profile.json',out); print(json.dumps(out,ensure_ascii=False),flush=True)

def old_profile(label,profile):
    import q2_full_run as old
    old.BASE=BASE/'old-driver'
    args=SimpleNamespace(cells=8192,method='Radau',rtol=1e-11,atol_T=1e-12,atol_C=1e-14,
        max_step=.5,first_step_factor=.01,root_xtol=5e-15,stop=121.,audit=True,
        timeout=120.,run_budget=240.,resume=False,label=label)
    pr=cProfile.Profile(); start=time.perf_counter()
    if profile: pr.enable()
    rec=old.solve(args)
    if profile:
        pr.disable(); pr.dump_stats(str(BASE/(label+'.prof')))
        stream=io.StringIO(); ps=pstats.Stats(pr,stream=stream).strip_dirs()
        ps.sort_stats('cumulative').print_stats(45); ps.sort_stats('tottime').print_stats(30)
        (BASE/(label+'-profile.txt')).write_text(stream.getvalue(),encoding='utf8')
    save(BASE/(label+'-timing.json'),{'wall_s':time.perf_counter()-start,'instrumented':profile,
        'peak_process_MB':peak_mb(),'result':rec['stats']})
    print(label,rec['status'],rec['active_wall_s'],flush=True)

class Meter:
    def __init__(self): self.seconds=defaultdict(float); self.calls=defaultdict(int)
    def wrap(self,name,func):
        def f(*a,**kw):
            t=time.perf_counter()
            try: return func(*a,**kw)
            finally: self.seconds[name]+=time.perf_counter()-t; self.calls[name]+=1
        return f
    def result(self):return {'seconds_inclusive_do_not_sum_nested':dict(self.seconds),'calls':dict(self.calls)}

def seed(start,n):
    if start==0:return np.r_[np.tile([28.,2.55],n),np.zeros(3)],{'kind':'original uniform initial state'}
    folder=Q2 if n==8192 else ROOT/f'records/q2-full-20260911/runs/space-{n}'
    path=folder/f'segment-{start:011.4f}.npz'
    meta=json.loads((folder/'run.json').read_text(encoding='utf8'))
    expected=next(s['sha256'] for s in meta['segments'] if s['end_s']==start)
    if sha256(path)!=expected: raise ValueError('Seed checkpoint mismatch')
    with np.load(path) as z: y=z['state'].copy()
    if y.shape!=(2*n+3,): raise ValueError('Not a complete same-grid state')
    return y,{'kind':'full same-grid Q2 checkpoint; local accuracy only','path':str(path),'sha256':expected}

def transferred_seed(path,model):
    sys.path.insert(0,str(BASE/'event-review'))
    from radial_extrema import RingMaximumPlan
    path=ROOT/path
    with np.load(path) as z:old=z['state'];faces=z['faces_m'];start=float(z['end_s'])
    k=len(faces)-1;n=model.grid.cells
    if k==n and np.array_equal(faces,model.grid.faces):
        return old.copy(),{'kind':'same-grid complete local raw state, no transfer or output reconstruction',
            'path':str(path),'sha256':sha256(path),'start_s':start,
            'note':'Retains the source run history and its limitations; not a global Q3 accuracy certificate.'}
    plan=RingMaximumPlan(faces);new=np.zeros(2*n+3);new[-3:]=old[-3:]
    edges=model.grid.faces**2;mid=(edges[:-1]+edges[1:])/2
    idx=np.minimum(np.searchsorted(faces**2,mid,side='right')-1,k-1)
    for f in [0,1]:
        pp=plan.profile(old[f:2*k:2],float(old[2*k-2+f]))
        lo=(edges[:-1]-pp.origin_x[idx])/pp.scale_x[idx]
        hi=(edges[1:]-pp.origin_x[idx])/pp.scale_x[idx]
        a=pp.coefficients[idx]
        new[f:2*n:2]=a[:,0]+a[:,1]*(lo+hi)/2+a[:,2]*(lo*lo+lo*hi+hi*hi)/3
    vo=np.pi*.25*np.diff(faces**2);vn=model.grid.volumes
    dg=[float((np.bincount(idx,weights=new[f:2*n:2]*vn,minlength=k)-old[f:2*k:2]*vo).max()) for f in [0,1]]
    absolute=[float(np.max(abs(np.bincount(idx,weights=new[f:2*n:2]*vn,minlength=k)-old[f:2*k:2]*vo))) for f in [0,1]]
    w=model.props.thermal
    heat_change=float(vn@(w(new[1:2*n:2])[0]*(new[:2*n:2]-28))-vo@(w(old[1:2*k:2])[0]*(old[:2*k:2]-28)))
    if np.min(new[1:2*n:2])<=0:raise ValueError('Transfer positivity failed; do not clip')
    return new,{'kind':'coarse polynomial conservative transfer; NO global accuracy claim',
        'path':str(path),'sha256':sha256(path),'coarse_cells':k,'fine_cells':n,'start_s':start,
        'max_per_parent_integral_transfer_T_C':absolute,'effective_heat_change_J':heat_change,
        'minimum_transferred_C':float(new[1:2*n:2].min()),'note':'Preserves T and C volume averages; nonlinear effective storage changes are recorded, not erased.'}

def query_grid(start,end,sample_step=1):
    knots=np.arange(np.ceil(start/60)*60,min(end,14400)+1,60)
    a=np.r_[start,end,np.arange(np.ceil(start/sample_step)*sample_step,end+.1,sample_step),knots-.1,knots,knots+.1]
    if start==0:a=np.r_[a,.001,.01,.1,.25,.5]
    return np.unique(a[(a>=start)&(a<=end)])

def run_local(args):
    if args.end-args.start>3600 or args.start<0:
        raise ValueError('Bounded local research only')
    dest=BASE/'local'/args.label
    if dest.exists(): raise FileExistsError('Use a new experiment label')
    dest.mkdir(parents=True)
    execution_source=execution_source_snapshot(dest)
    t0=time.perf_counter(); td,cd,raw=drives(); n=args.cells
    if args.end>14400:
        # 长时程延拓仅作用于边界函数，不修改附件中的实测节点。
        from q3_prelocalize import continued
        td,cd,raw=continued()
    from q3_scalar_properties import ScalarProperties
    m=CoupledFV(n,flux='fick',properties=ScalarProperties() if args.scalar else None)
    y,parent=transferred_seed(args.seed_npz,m) if args.seed_npz else seed(args.start,n)
    if args.seed_npz and parent['start_s']!=args.start:raise ValueError('Seed time mismatch')
    y0=y.copy()
    times=query_grid(args.start,args.end,args.sample_step); radii=np.arange(21)*.001
    weights=m.grid.point_weights(radii); wi=[np.flatnonzero(row) for row in weights]
    ww=[row[ids] for row,ids in zip(weights,wi)]
    def reconstruct(z):
        if args.sparse_output:
            return np.array([[w@z[2*ix] for w,ix in zip(ww,wi)],
                             [w@z[2*ix+1] for w,ix in zip(ww,wi)]])
        return np.array([weights@z[:2*n:2],weights@z[1:2*n:2]])
    values=np.zeros((len(times),2*n+3)); values[0]=y
    points=np.zeros((len(times),2,21)); points[0]=reconstruct(y)
    if args.start==0: points[0]=np.array([[28.]*21,[2.55]*21])
    else:
        sf=m.surface(y[2*n-2],y[2*n-1],td(args.start),cd(args.start));points[0,:,-1]=sf.T,sf.C
    meter=Meter(); m.surface=meter.wrap('surface_subset_of_RHS_Jacobian_output',m.surface)
    m.interior=meter.wrap('interior_subset_of_RHS_Jacobian',m.interior)
    m.props
    def rhs(t,z):
        if time.perf_counter()-t0>args.budget:raise TimeoutError('Per-experiment wall budget exceeded')
        try:return m.augmented(z,td(t),cd(t))
        except (ValueError,ArithmeticError) as exc:
            save(dest/'rejected-domain-trial.json',{'t_trial_s':float(t),'minimum_trial_C':float(np.nanmin(z[1:2*n:2])),
                'minimum_trial_T_K':float(np.nanmin(z[:2*n:2])+273.15),'error':repr(exc),
                'note':'Rejected trial only; never clipped or accepted'})
            raise
    def jac(t,z): return m.augmented(z,td(t),cd(t),jacobian=True)
    rhs=meter.wrap('RHS_total',rhs); jac=meter.wrap('Jacobian_total',jac)
    knots=raw[:,0]; ends=np.r_[knots[(knots>args.start)&(knots<args.end)],args.end]
    atols=np.r_[np.tile([args.atol_T,args.atol_C],n),args.atol_C,args.atol_T,args.atol_T]
    # 被动记录求解信息，不改变库函数返回结果。
    original=rm.solve_collocation_system if args.method=='Radau' else bm.solve_bdf_system
    attempts=[]; nonlinear_fail=0; maxiter=0
    def observer(*aa,**kk):
        nonlocal nonlinear_fail,maxiter
        key=(float(aa[1]),float(aa[3])) if args.method=='Radau' else (float(aa[1]),float(aa[3]))
        attempts.append(key)
        out=original(*aa,**kk);nonlinear_fail+=int(not out[0]);maxiter=max(maxiter,int(out[1]))
        return out
    if args.method=='Radau':rm.solve_collocation_system=observer
    else:bm.solve_bdf_system=observer
    steps=[]; nf=nj=nl=0; rejected=0; newton_retries=0; now=args.start; nextout=1; fill=[]
    init_dt=.01*np.min(np.diff(m.grid.faces))**2/(float(m.props.thermal(2.55)[1])/float(m.props.thermal(2.55)[0]))
    lastdt=None; maxgap=0.; minc=float(y[1:2*n:2].min()); mintk=float(y[:2*n:2].min()+273.15)
    v=m.grid.volumes; vol=v.sum(); w0=float(m.props.thermal(2.55)[0]); ar=m.grid.areas[-1]
    gauss=np.zeros(3);gx,gw=np.polynomial.legendre.leggauss(5)
    cstage=1e100; outputclock=0.; auditclock=0.; densecount=0
    sys.path.insert(0,str(BASE/'event-review'))
    from radial_extrema import RingMaximumPlan,scan_profile
    maxplan=RingMaximumPlan(m.grid.faces); event_record=None;event_monitor_clock=0.
    def maximum(t,z):
        sf=m.surface(z[2*n-2],z[2*n-1],td(t),cd(t))
        return scan_profile(maxplan.profile(z[1:2*n:2],sf.C))
    oldmaximum=maximum(args.start,y)
    try:
        for end in ends:
            if args.first=='legacy' or now==0: first=min(init_dt,end-now)
            else:first=min(1. if lastdt is None else lastdt,end-now,args.max_step)
            solver=(Radau if args.method=='Radau' else BDF)(rhs,now,y,float(end),jac=jac,
                    rtol=args.rtol,atol=atols,max_step=args.max_step,first_step=first)
            old_lu=solver.lu
            def factor(a):
                if args.natural:
                    solver.nlu+=1
                    lu=splu(a,permc_spec='NATURAL')
                else:lu=old_lu(a)
                fill.append([int(a.nnz),int(lu.L.nnz+lu.U.nnz)])
                return lu
            solver.lu=meter.wrap('LU_factorization',factor)
            solver.solve_lu=meter.wrap('linear_triangular_solve',solver.solve_lu)
            while solver.status=='running':
                begin=float(solver.t); before=solver.y.copy(); attempts.clear()
                msg=solver.step()
                if solver.status=='failed':raise ArithmeticError(str(msg))
                unique=[]
                for key in attempts:
                    if not unique or key!=unique[-1]:unique.append(key)
                rejected+=max(0,len(unique)-1);newton_retries+=len(attempts)-len(unique)
                finish=float(solver.t);lastdt=finish-begin;steps.append((begin,finish,lastdt))
                z=solver.y
                minc=min(minc,float(z[1:2*n:2].min()));mintk=min(mintk,float(z[:2*n:2].min()+273.15))
                if args.method=='Radau':
                    cstage=min(cstage,float((before[None,:]+solver.Z)[:,1:2*n:2].min()))
                if minc<=0 or mintk<=0 or not np.all(np.isfinite(z)):raise ArithmeticError('Nonphysical accepted state')
                maxgap=max(maxgap,abs(float((z[1:2*n:2]-before[1:2*n:2])@v+vol*(z[-3]-before[-3]))))
                # 在每个接受步核验守恒量，且不依赖额外的被动积分状态。
                ac=time.perf_counter(); dense=solver.dense_output();densecount+=1
                qs=(finish+begin)/2+(finish-begin)/2*gx
                uz=dense(qs); flux=np.zeros((5,3))
                for j,tq in enumerate(qs):
                    zz=uz[:,j];sf=m.surface(zz[2*n-2],zz[2*n-1],td(tq),cd(tq))
                    cc=zz[1:2*n:2];tt=zz[:2*n:2]
                    dc=m.evaluate(zz,td(tq),cd(tq))[0][1::2]
                    wp=m.props.thermal(cc)[2]
                    flux[j]=[ar*sf.flux_C/vol,-ar*sf.flux_T/(vol*w0),float((v*wp*(tt-28))@dc/(vol*w0))]
                gauss+=(finish-begin)/2*(gw@flux);auditclock+=time.perf_counter()-ac
                mc=time.perf_counter();mm=maximum(finish,z)
                if oldmaximum['maximum_C']>=.15 and mm['maximum_C']<.15:
                    te=brentq(lambda tq:maximum(tq,dense(tq))['maximum_C']-.15,begin,finish,xtol=1e-5,rtol=1e-14)
                    hh=min(.1,(te-begin)/2,(finish-te)/2)
                    slope=(maximum(te+hh,dense(te+hh))['maximum_C']-maximum(te-hh,dense(te-hh))['maximum_C'])/(2*hh) if hh>1e-9 else None
                    event_record={'time_s':te,'bracket_s':[begin,finish],'local_slope_C_per_s':slope,
                        'maximum_scan':maximum(te,dense(te)),'scope':'Local event from shared possibly coarse seed; not production event'}
                    np.savez_compressed(dest/'event-state.npz',state=dense(te),end_s=te,faces_m=m.grid.faces)
                oldmaximum=mm;event_monitor_clock+=time.perf_counter()-mc
                oc=time.perf_counter()
                while nextout<len(times) and times[nextout]<=finish+1e-12:
                    qt=times[nextout]; zz=dense(qt);values[nextout]=zz
                    pt=reconstruct(zz);sf=m.surface(zz[2*n-2],zz[2*n-1],td(qt),cd(qt))
                    pt[:,-1]=sf.T,sf.C;points[nextout]=pt;nextout+=1
                outputclock+=time.perf_counter()-oc
            nf+=solver.nfev; nj+=solver.njev;nl+=solver.nlu
            y=solver.y.copy();now=float(end)
        if now!=args.end or nextout!=len(times):raise ArithmeticError('Incomplete interval/output')
        solvewall=time.perf_counter()-t0
        cc=y[1:2*n:2];tt=y[:2*n:2];cc0=y0[1:2*n:2];tt0=y0[:2*n:2]
        dh=float((v*(m.props.thermal(cc)[0]*(tt-28)-m.props.thermal(cc0)[0]*(tt0-28))).sum()/(vol*w0))
        dcmean=float(v@(cc-cc0)/vol)
        stats={'status':'COMPLETED_LOCAL_ONLY','parameters':vars(args),'parent':parent,
          'time_s':[args.start,args.end],'cells':n,'actual_end_s':now,'accepted_steps':len(steps),
          'rejected_step_sizes':rejected,'same_step_Jacobian_retries':newton_retries,
          'nonlinear_attempt_failures':nonlinear_fail,'maximum_newton_iterations':maxiter,
          'nfev':nf,'njev':nj,'nlu':nl,'dt_quantiles_s':np.quantile(np.array(steps)[:,2],[0,.1,.5,.9,1]).tolist(),
          'max_C_step_balance_m3_C':maxgap,'local_C_integral_balance_kgkg':dcmean+gauss[0],
          'local_heat_chain_balance_equivalent_K':dh-gauss[1]-gauss[2],
          'passive_vs_independent_integrals':(y[-3:]-y0[-3:]-gauss).tolist(),
          'min_accepted_C':minc,'min_stage_C':cstage if args.method=='Radau' else None,'min_T_K':mintk,
          'surface_root_equivalent_C':m.max_root_equiv,'surface_heat_residual_W_m2':m.max_heat_residual,
          'surface_C_residual_m_s':m.max_moisture_residual,'timer':meter.result(),
          'audit_wall_s':auditclock,'output_reconstruction_wall_s':outputclock,
          'event_monitor_wall_s':event_monitor_clock,'event':event_record,'end_maximum_scan':oldmaximum,
          'dense_output_objects_requested':densecount,'wall_before_save_s':solvewall,
          'LU_A_and_factors_nnz_minmax':np.array([np.min(fill,axis=0),np.max(fill,axis=0)]).tolist(),
          'peak_process_MB':peak_mb(),'frozen_sources':frozen(),
          'driver_sha256':_IMPORT_SOURCE_SHA,'driver_disk_sha256_at_finish':sha256(__file__),
          'execution_source':execution_source,
          'python':platform.python_version(),'scipy':scipy.__version__,'numpy':np.__version__}
        sc=time.perf_counter();np.savez_compressed(dest/'solution.npz',time_s=times,radius_m=radii,
            states=values,T_C=points[:,0],C_kg_kg=points[:,1],final_state=y,faces_m=m.grid.faces)
        np.savetxt(dest/'steps.csv',steps,delimiter=',',header='start_s,end_s,dt_s',comments='')
        stats['save_wall_s']=time.perf_counter()-sc;stats['total_wall_s']=time.perf_counter()-t0
        save(dest/'run.json',stats); print(json.dumps({k:stats[k] for k in ['status','total_wall_s','accepted_steps','rejected_step_sizes','nfev','njev','nlu','dt_quantiles_s','local_C_integral_balance_kgkg','local_heat_chain_balance_equivalent_K']}),flush=True)
    except BaseException as e:
        save(dest/'failure.json',{'error':repr(e),'now':now,'elapsed':time.perf_counter()-t0,'settings':vars(args)})
        raise
    finally:
        if args.method=='Radau':rm.solve_collocation_system=original
        else:bm.solve_bdf_system=original

def main():
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=['historical','old-profile','local'])
    ap.add_argument('--label',default='untitled');ap.add_argument('--profile',action='store_true')
    ap.add_argument('--cells',type=int,default=8192);ap.add_argument('--start',type=float,default=0)
    ap.add_argument('--end',type=float,default=121);ap.add_argument('--method',choices=['Radau','BDF'],default='Radau')
    ap.add_argument('--max-step',type=float,default=30);ap.add_argument('--first',choices=['legacy','warm'],default='warm')
    ap.add_argument('--rtol',type=float,default=1e-11);ap.add_argument('--atol-T',type=float,default=1e-12)
    ap.add_argument('--atol-C',type=float,default=1e-14);ap.add_argument('--sparse-output',action='store_true')
    ap.add_argument('--budget',type=float,default=180)
    ap.add_argument('--scalar',action='store_true')
    ap.add_argument('--natural',action='store_true',help='Same matrices; SuperLU natural order, no frozen Jacobian')
    ap.add_argument('--sample-step',type=float,default=1)
    ap.add_argument('--seed-npz')
    a=ap.parse_args();BASE.mkdir(exist_ok=True,parents=True)
    with threadpool_limits(1):
        if a.task=='historical':historical()
        elif a.task=='old-profile':old_profile(a.label,a.profile)
        else:run_local(a)


import time
import numpy as np
from scipy.sparse.linalg import splu


def factor(matrix,physical_size):
    A=matrix.tocsc();m=int(physical_size)
    if A.shape!=(m+3,m+3) or m<=0:raise ValueError('Expected physical block plus three passive states')
    if np.any(A[:m,m:].data!=0):raise ValueError('Passive states feed back: block rule invalid')
    S=A[m:,m:].toarray();diagonal=np.diag(S).copy()
    if np.any(S!=np.diag(diagonal)) or np.any(diagonal==0) or not np.all(np.isfinite(diagonal)):
        raise ValueError('Auxiliary block must be nonsingular diagonal')
    P=A[:m,:m].tocsc();B=A[m:,:m].tocsr()
    lu=splu(P)
    return lu,B,diagonal


def solve(factors,rhs):
    lu,B,diagonal=factors;m=lu.shape[0]
    b=np.asarray(rhs);x=lu.solve(b[:m])
    rest=(b[m:]-B@x)/(diagonal if b.ndim==1 else diagonal[:,None])
    return np.concatenate((x,rest),axis=0)


def install(solver,physical_size,stats):


    for key in ['block_factor_seconds','block_solve_seconds','block_factor_calls','block_solve_calls','block_max_LU_nnz']:
        stats.setdefault(key,0)
    def lu(A):
        started=time.perf_counter();out=factor(A,physical_size)
        solver.nlu+=1;stats['block_factor_calls']+=1
        stats['block_max_LU_nnz']=max(stats['block_max_LU_nnz'],out[0].L.nnz+out[0].U.nnz)
        stats['block_factor_seconds']+=time.perf_counter()-started
        return out
    def solve_lu(LU,b):
        started=time.perf_counter();out=solve(LU,b)
        stats['block_solve_calls']+=1;stats['block_solve_seconds']+=time.perf_counter()-started
        return out
    solver.lu=lu;solver.solve_lu=solve_lu


def algebra_checks():
    from scipy.sparse import eye
    from q2_model import CoupledFV
    from q3_scalar_properties import ScalarProperties
    from q3_full_run import environment
    td,cd,_=environment();results=[]
    seed=20260912;rng=np.random.default_rng(seed)
    for n in [64,256]:
        model=CoupledFV(n,flux='fick',properties=ScalarProperties())
        r=model.grid.faces[1:];u=r/.02
        y=np.zeros(2*n+3);y[:2*n:2]=42+3*u*u;y[1:2*n:2]=.8-.3*u*u
        J=model.augmented(y,td(7200),cd(7200),jacobian=True)
        for shift in [1.,.005,3+2j]:
            A=(shift*eye(2*n+3,format='csc')-J).tocsc()
            b=rng.normal(size=(2*n+3,2))
            if isinstance(shift,complex):b=b+1j*rng.normal(size=b.shape)
            full=splu(A);x=full.solve(b);fac=factor(A,2*n);z=solve(fac,b)
            scale=float(np.max(np.asarray(abs(A).sum(axis=1))))*np.max(abs(z))+np.max(abs(b))
            residual=float(np.max(abs(A@z-b))/scale)
            gap=float(np.max(abs(x-z))/max(1.,np.max(abs(x))))
            if residual>1e-13 or gap>1e-10:raise ArithmeticError('Block linear algebra discrepancy')
            results.append(dict(cells=n,shift=str(shift),normalized_residual=residual,
                                relative_solution_difference=gap,
                                full_LU_nnz=full.L.nnz+full.U.nnz,physical_LU_nnz=fac[0].L.nnz+fac[0].U.nnz))
    return dict(passed=True,seed_for_linear_rhs_test_only=seed,cases=results,
                scope='Exact-system algebra tests only; integration equivalence and recovered histories checked separately')


from pathlib import Path
import argparse, csv, hashlib, inspect, json, platform, sys, time, traceback
import numpy as np
import scipy
from scipy.integrate import BDF, Radau
import scipy.integrate._ivp.bdf as bm
import scipy.integrate._ivp.radau as rm
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits, threadpool_info

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'records/q3-full-20260911'
EXTREMA=ROOT/'records/q3-acceleration-20260911/event-review'
sys.path.insert(0,str(EXTREMA))
SOURCE_BYTES=Path(__file__).read_bytes()


class HoldDrive:
    def __init__(self, source):
        self.source=source;self.times=source.times
        self.end=float(self.times[-1]);self.last=float(source(self.end))
    def __call__(self,t):
        return self.last if t>self.end else self.source(t)


def environment(attachments_dir):
    path=(Path(attachments_dir)/'附件1.xlsx').resolve()
    if not path.is_file(): raise FileNotFoundError(f'缺少题目附件：{path}')
    cfg={'inputs':{'environment':{'path':str(path),'sheet':'Sheet1','sha256':sha256(path)}}}
    td,cd,raw=read_environment(ROOT,cfg)
    return HoldDrive(td),HoldDrive(cd),raw


def q2_terminal_state(result2_path, centres_m):


    from openpyxl import load_workbook
    path=Path(result2_path).resolve()
    if not path.is_file(): raise FileNotFoundError(f'缺少问题二结果：{path}')
    book=load_workbook(path,read_only=True,data_only=True)
    try:
        sheets=book.worksheets
        if len(sheets)<2: raise ValueError('result2.xlsx 必须含温度与水分浓度工作表')
        def terminal(sheet):
            rows=sheet.iter_rows(values_only=True); header=next(rows)
            radii=np.asarray(header[1:],float)*.01
            selected=None
            for row in rows:
                if row[0] is not None and float(row[0])<=10800.: selected=row
            if selected is None or not np.isclose(float(selected[0]),10800.): raise ValueError('result2.xlsx 缺少 10800 s 末状态')
            return radii,np.asarray(selected[1:],float)
        radii,T=terminal(sheets[0]); radii_c,C=terminal(sheets[1])
    finally: book.close()
    if not np.array_equal(radii,radii_c): raise ValueError('result2 温度与水分径向节点不一致')
    return np.interp(centres_m,radii,T),np.interp(centres_m,radii,C)


def snapshot(dest):
    folder=dest/'source';folder.mkdir()
    source=Path(__file__); hashes={source.name:hashlib.sha256(source.read_bytes()).hexdigest()}
    (folder/source.name).write_bytes(source.read_bytes())
    save(dest/'execution-source.json',{'recorded_before_integration':True,'sha256':hashes,
         'dependency_note':'仅依赖同目录 drying_model_core.py、附件1.xlsx 与 result2.xlsx。'})
    return hashes


def dense_arrays(dense,method):
    out={'begin_s':np.array(dense.t_old),'end_s':np.array(dense.t),'method':np.array(method)}
    if method=='BDF':out.update(D=dense.D,t_shift=dense.t_shift,denom=dense.denom)
    else:out.update(Q=dense.Q,y_old=dense.y_old,h=np.array(dense.h))
    return out


def dense_value(data,t):

    if str(data['method'])=='BDF':
        x=(float(t)-data['t_shift'])/data['denom']
        return data['D'][0]+np.cumprod(x)@data['D'][1:]
    x=(float(t)-float(data['begin_s']))/float(data['h'])
    return data['y_old']+data['Q']@np.array([x,x*x,x*x*x])


def point_plan(model):
    radii=np.arange(21)*.001
    rows=model.grid.point_weights(radii)
    ids=[np.flatnonzero(row) for row in rows]
    return radii,ids,[row[ix] for row,ix in zip(rows,ids)]


def solve(args):
    if sum(bool(x) for x in [args.prefix,args.restart,args.q3_prefix])>1:raise ValueError('Choose one source of prefix/restart')
    dest=args.output.resolve()/args.label;dest.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();source=snapshot(dest)
    td,cd,raw=environment(args.attachments_dir);n=args.cells
    m=CoupledFV(n,flux='fick',properties=ScalarProperties(),root_xtol=args.root_xtol)
    g=m.grid;v=g.volumes;vol=float(v.sum());ar=float(g.areas[-1]);w0=float(m.props.thermal(2.55)[0])
    if g.radius!=.02 or g.length!=.25:raise ValueError('Geometry changed')
    maxplan=RingMaximumPlan(g.faces);radii,ix,ww=point_plan(m)
    T0,C0=q2_terminal_state(args.result2,g.centres)
    y0=np.r_[np.column_stack((T0,C0)).ravel(),np.zeros(3)]
    y=y0.copy();now=10800.
    atols=np.r_[np.tile([args.atol_T,args.atol_C],n),args.atol_C,args.atol_T,args.atol_T]
    dtcold=args.first_factor*(g.faces[1]-g.faces[0])**2/(float(m.props.thermal(2.55)[1])/w0)
    record={'status':'INITIALIZING','settings':vars(args),'created_unix':time.time(),'command':sys.argv,
            'interpreter':sys.executable,'python':platform.python_version(),'scipy':scipy.__version__,
            'numpy':np.__version__,'threadpools':threadpool_info(),'sources':source,
            'model':'Q2-R with approved last-value continuation','original_clock_s':now,
            'initial_state_source':'result2.xlsx at 10800 s; progressive Q2→Q3 inheritance, not records continuation',
            'official_result':False,'blocks':[],'checkpoints':[],'event_dense':[],'prefix':None,
            'segments':[],'event':None,'active_wall_s':0.}
    save(dest/'run.json',record)
    buffer=[];block_number=0;new_outputs=0;last_output=-1.;maxbalance=np.zeros(2)
    gauss=np.zeros(3);prefix_gauss=None;independent_since=0.;stats={
        'accepted_steps':0,'rejected_step_sizes':0,'same_step_jacobian_retries':0,
        'nonlinear_failures':0,'maximum_newton_iterations':0,'recoveries':0,
        'min_accepted_C':2.55,'min_accepted_T_K':301.15,'max_accepted_C':2.55,'max_accepted_T_C':28.,
        'dense_formula_max_difference':0.,'output_seconds':0.,'audit_seconds':0.,'save_seconds':0.,
        'event_monitor_seconds':0.,'max_C_increase_accepted':0.,'max_C_increase_output':0.}
    lastscan=None;last_accepted_max=None

    def surface(t,z):return m.surface(float(z[2*n-2]),float(z[2*n-1]),td(t),cd(t))
    def maximum(t,z):return scan_profile(maxplan.profile(z[1:2*n:2],surface(t,z).C))
    def add_output(t,z,initial=False):
        nonlocal last_output,new_outputs,lastscan
        if t<=last_output:raise ValueError('Duplicate or reversed output time')
        clock=time.perf_counter()
        if not np.all(np.isfinite(z)) or np.min(z[1:2*n:2])<=0:raise ValueError('Nonphysical output')
        sf=surface(t,z)
        point=np.array([[w@z[2*i] for w,i in zip(ww,ix)],[w@z[2*i+1] for w,i in zip(ww,ix)]])
        point[:,-1]=sf.T,sf.C
        if initial:point[:]=np.array([[28.]*21,[2.55]*21])
        scan=scan_profile(maxplan.profile(z[1:2*n:2],2.55 if initial else sf.C))
        if initial:scan['maximum_C']=scan['minimum_C']=2.55
        balance=np.array([v@(z[1:2*n:2]-2.55)/vol+z[-3],
            v@(m.props.thermal(z[1:2*n:2])[0]*(z[:2*n:2]-28))/(vol*w0)-z[-2]-z[-1]])
        maxbalance[:]=np.maximum(maxbalance,abs(balance))
        if lastscan is not None:stats['max_C_increase_output']=max(stats['max_C_increase_output'],scan['maximum_C']-lastscan['maximum_C'])
        buffer.append((float(t),z.copy(),point,balance,scan))
        lastscan=scan;last_output=float(t);new_outputs+=1
        stats['output_seconds']+=time.perf_counter()-clock
    def flush():
        nonlocal block_number
        if not buffer:return
        clock=time.perf_counter();block_number+=1
        p=dest/f'block-{block_number:04d}.npz'
        np.savez_compressed(p,time_s=np.array([q[0] for q in buffer]),states=np.array([q[1] for q in buffer]),
            T_C=np.array([q[2][0] for q in buffer]),C_kg_kg=np.array([q[2][1] for q in buffer]),
            balances=np.array([q[3] for q in buffer]),radius_m=radii,faces_m=g.faces)
        record['blocks'].append({'file':p.name,'sha256':sha256(p),'start_s':buffer[0][0],
            'end_s':buffer[-1][0],'rows':len(buffer)})
        with (dest/'output-maxima.jsonl').open('a',encoding='utf8') as f:
            for q in buffer:f.write(json.dumps({'time_s':q[0],**q[4]},ensure_ascii=False)+'\n')
        buffer.clear();stats['save_seconds']+=time.perf_counter()-clock
    def checkpoint(t,z,solver=None):
        if record['checkpoints'] and record['checkpoints'][-1]['time_s']==float(t):
            old=record['checkpoints'][-1];p=dest/old['file']
            if sha256(p)!=old['sha256']:raise ValueError('Existing checkpoint changed')
            with np.load(p) as previous:
                if not np.array_equal(previous['state'],z) or not np.array_equal(previous['gauss'],gauss):
                    raise ValueError('Different states requested at the same checkpoint time')
            return
        flush();clock=time.perf_counter();p=dest/f'checkpoint-{t:014.6f}.npz'
        extra={}
        if solver is not None and args.method=='BDF':
            extra={'bdf_D':solver.D,'bdf_order':solver.order,'bdf_h_abs':solver.h_abs,
                   'bdf_n_equal_steps':solver.n_equal_steps,'bdf_t_old':solver.t_old}
        np.savez_compressed(p,state=z,end_s=t,faces_m=g.faces,gauss=gauss,**extra)
        record['checkpoints'].append({'file':p.name,'sha256':sha256(p),'time_s':float(t),
            'recovery':'Full raw state; separate restart validation required. BDF difference history saved when present.'})
        record.update(status='RUNNING',actual_end_s=float(t),active_wall_s=time.perf_counter()-started,
                      stats=stats,maximum_balances=maxbalance.tolist(),last_output_s=last_output)
        save(dest/'run.json',record);stats['save_seconds']+=time.perf_counter()-clock

    add_output(now,y,True)
    if args.prefix:
        folder=ROOT/args.prefix;meta=json.loads((folder/'run.json').read_text(encoding='utf8'))
        sig=meta['signature'];sett=sig['settings']
        if meta['status']!='COMPLETED' or sett['cells']!=n or sett['stop']!=10800.:
            raise ValueError('Incomplete or wrong-grid prefix')
        for key,rel in [('model_sha256','src/q2_model.py'),('helper_sha256','src/q1_model.py'),('physical_config_sha256','config/q2.json')]:
            if sig[key]!=sha256(ROOT/rel):raise ValueError('Prefix physics fingerprint mismatch')
        prev=0.
        for seg in meta['segments']:
            p=folder/seg['file']
            if sha256(p)!=seg['sha256']:raise ValueError('Prefix segment fingerprint mismatch')
            with np.load(p) as z:
                end=float(z['end_s']);state=z['state'].copy()
                if state.shape!=(2*n+3,) or not np.array_equal(z['faces_m'],g.faces) or end!=prev+60:
                    raise ValueError('Prefix geometry/time/state mismatch')
                prefix_gauss=z['gauss_totals'].copy()
            add_output(end,state);prev=end
            if len(buffer)>=60:flush()
        now=prev;y=state
        record['prefix']={'folder':str(folder.relative_to(ROOT)),'run_sha256':sha256(folder/'run.json'),
             'segments':len(meta['segments']),'end_s':now,'source_method':sett['method'],
             'full_same_grid_raw_history':True,'prefix_settings':sett,
             'note':f'Reuse original raw trajectory; initialize a new {args.method}, no property switch or Richardson restart.'}
        independent_since=now;checkpoint(now,y)
    elif args.q3_prefix:
        folder=ROOT/args.q3_prefix;parent=json.loads((folder/'run.json').read_text(encoding='utf8'))
        if parent['settings']['cells']!=n:raise ValueError('Q3 prefix has wrong grid')
        for rel in ['src/q1_model.py','src/q2_model.py','src/q3_scalar_properties.py','config/q2.json',
                    'records/q3-acceleration-20260911/event-review/radial_extrema.py']:
            if parent['sources'][rel]!=sha256(ROOT/rel):raise ValueError('Q3 prefix physical source mismatch')
        until=args.prefix_until
        available=[cp for cp in parent['checkpoints'] if until is None or cp['time_s']==until]
        if not available:raise ValueError('Q3 prefix endpoint must be an actual full-state checkpoint')
        cp=available[-1];until=cp['time_s'];p=folder/cp['file']
        if sha256(p)!=cp['sha256']:raise ValueError('Q3 parent checkpoint changed')
        with np.load(p) as z:
            state=z['state'].copy()
            if float(z['end_s'])!=until or not np.array_equal(z['faces_m'],g.faces):raise ValueError('Parent state/time/grid mismatch')
        if state.shape!=(2*n+3,):raise ValueError('Parent complete state dimension differs')
        used=[];seen=[0.];last_state=None
        for b in parent['blocks']:
            if b['start_s']>until:break
            p=folder/b['file']
            if sha256(p)!=b['sha256']:raise ValueError('Q3 parent block changed')
            with np.load(p) as z:
                ts=z['time_s'];ys=z['states']
                if ys.shape!=(len(ts),2*n+3) or not np.array_equal(z['faces_m'],g.faces):raise ValueError('Parent block dimensions/grid')
                for qt,qy in zip(ts,ys):
                    if qt==0:
                        if not np.array_equal(qy,y):raise ValueError('Parent initial physical/auxiliary state changed')
                        continue
                    if qt>until:break
                    if not np.all(np.isfinite(qy)) or qt<=seen[-1]:raise ValueError('Parent state/time ordering')
                    add_output(float(qt),qy);seen.append(float(qt));last_state=qy.copy()
                    if len(buffer)>=60:flush()
            used.append(b)
        if seen[-1]!=until or last_state is None:raise ValueError('Parent block endpoint missing')
        endpoint_gap=abs(last_state-state)
        if np.any(endpoint_gap>256*np.finfo(float).eps*np.maximum(1.,abs(state))):
            raise ValueError('Parent dense endpoint and accepted checkpoint differ beyond floating allowance')
        if len(np.setdiff1d(np.arange(0,until+1,60.),seen)):raise ValueError('Parent history missing required minutes')
        now=until;y=state
        record['prefix']={'folder':str(folder.relative_to(ROOT)),'run_sha256':sha256(folder/'run.json'),
            'kind':'Complete same-grid Q3 raw history through verified checkpoint; original parent may have later failed',
            'parent_status':parent['status'],'end_s':now,'checkpoint':cp,'blocks':used,
            'full_same_grid_raw_history':True,'source_method':parent['settings']['method'],
            'prefix_settings':parent['settings'],
            'dense_endpoint_vs_accepted_checkpoint':{'max_T_C':float(np.max(endpoint_gap[:2*n:2])),
                'max_C_kg_kg':float(np.max(endpoint_gap[1:2*n:2])),
                'max_auxiliary':float(np.max(endpoint_gap[-3:])),
                'rule':'Retain archived dense outputs; resume from actual accepted full state. Difference must fit 256-eps component scale, never rounded reconstruction.'},
            'note':f'New {args.method} initialized from full raw state; no interpolation or Richardson initial state'}
        independent_since=now;checkpoint(now,y)
    elif args.restart:
        if args.prefix:raise ValueError('Choose prefix OR restart')
        rp=ROOT/args.restart
        with np.load(rp) as z:
            if not np.array_equal(z['faces_m'],g.faces):raise ValueError('Restart grid mismatch')
            y=z['state'].copy();now=float(z['end_s'])
        if y.shape!=(2*n+3,):raise ValueError('Restart state dimension mismatch')
        # 问题二的初始状态必须来自规范结果，不以插值伪造。
        buffer.clear();last_output=-1.;new_outputs=0;add_output(now,y)
        record['prefix']={'restart_path':str(rp.relative_to(ROOT)),'sha256':sha256(rp),
                         'end_s':now,'parent_history_required':True}
        independent_since=now;checkpoint(now,y)
    ystart=y.copy();start_time=now
    query=np.unique(np.r_[np.arange(60.,args.limit+60.,60.),
        raw[1:,0]-.1,raw[1:,0]+.1,1.,.001,.01,.1,.25,.5,121.])
    query=query[(query>now)&(query<=args.limit)];qi=0
    original=rm.solve_collocation_system if args.method=='Radau' else bm.solve_bdf_system
    attempts=[]
    def observer(*a,**kw):
        attempts.append((float(a[1]),float(a[3])))
        out=original(*a,**kw);stats['nonlinear_failures']+=int(not out[0])
        stats['maximum_newton_iterations']=max(stats['maximum_newton_iterations'],int(out[1]))
        return out
    if args.method=='Radau':rm.solve_collocation_system=observer
    else:bm.solve_bdf_system=observer
    gx,gw=np.polynomial.legendre.leggauss(5)
    stepfile=(dest/'steps.csv').open('w',newline='',encoding='utf8');sw=csv.writer(stepfile)
    sw.writerow(['begin_s','end_s','dt_s','maximum_C','argmax_r_m','argmax_kind','C_integral_balance','step_rejections'])
    lastdt=None;last_report=started;next_checkpoint=(np.floor(now/3600)+1)*3600
    ends=np.r_[raw[(raw[:,0]>now)&(raw[:,0]<args.limit),0],args.limit]
    event=None;done=False;last_accepted_max=maximum(now,y)['maximum_C'];recovery_first=None
    def rhs(t,z):
        if time.perf_counter()-started>args.budget:raise TimeoutError('Run wall budget; last checkpoint retained')
        return m.augmented(z,td(t),cd(t))
    def jac(t,z):return m.augmented(z,td(t),cd(t),jacobian=True)
    try:
        for end in ends:
            # 已完成分段释放求解器内部循环引用，不改变当前状态或输出节点。
            if 'solver' in locals():del solver
            memory=collect_completed_cycles()
            with (dest/'memory.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps({'time_s':now,'stage':'before_input_segment',**memory})+'\n')
            segstart=now;segclock=time.perf_counter();cap=args.max_step_input if now<14400 else args.max_step_late
            first=min(dtcold if now==0 else (.001 if lastdt is None else lastdt),cap,float(end)-now)
            nf=nj=nl=0
            while now<float(end):
                solver=(BDF if args.method=='BDF' else Radau)(rhs,now,y,float(end),jac=jac,
                    rtol=args.rtol,atol=atols,max_step=cap,first_step=min(first,float(end)-now))
                if args.linear=='block':install_block_linear(solver,2*n,stats)
                retry=False
                while solver.status=='running':
                    begin=float(solver.t);before=solver.y.copy();attempts.clear()
                    try:msg=solver.step()
                    except (ValueError,ArithmeticError) as exc:
                        log={'start_s':begin,'error':repr(exc),'attempts':attempts,
                             'rule':'Unaccepted trial rejected; restart from last accepted raw state with cold first step. No clipping.'}
                        with (dest/'failures.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(log)+'\n')
                        stats['recoveries']+=1
                        if stats['recoveries']>8:raise
                        now=begin;y=before;first=min(dtcold,float(end)-begin);checkpoint(now,y)
                        retry=True;break
                    if solver.status=='failed':raise ArithmeticError(str(msg))
                    finish=float(solver.t);z=solver.y;dt=finish-begin
                    if dt<=0 or not np.all(np.isfinite(z)) or np.min(z[1:2*n:2])<=0 or np.min(z[:2*n:2])+273.15<=0:
                        raise ArithmeticError('Invalid accepted state')
                    unique=[]
                    for key in attempts:
                        if not unique or key!=unique[-1]:unique.append(key)
                    rejected=max(0,len(unique)-1)
                    stats['rejected_step_sizes']+=rejected;stats['same_step_jacobian_retries']+=len(attempts)-len(unique)
                    stats['accepted_steps']+=1;lastdt=dt
                    stats['min_accepted_C']=min(stats['min_accepted_C'],float(z[1:2*n:2].min()))
                    stats['max_accepted_C']=max(stats['max_accepted_C'],float(z[1:2*n:2].max()))
                    stats['min_accepted_T_K']=min(stats['min_accepted_T_K'],float(z[:2*n:2].min())+273.15)
                    stats['max_accepted_T_C']=max(stats['max_accepted_T_C'],float(z[:2*n:2].max()))
                    dense=solver.dense_output()
                    ac=time.perf_counter()
                    if args.audit:
                        qs=(finish+begin)/2+dt/2*gx;uz=dense(qs);fl=[]
                        for k,tq in enumerate(qs):
                            zz=uz[:,k];dy,sf=m.evaluate(zz,td(tq),cd(tq));cc=zz[1:2*n:2];tt=zz[:2*n:2]
                            fl.append([ar*sf.flux_C/vol,-ar*sf.flux_T/(vol*w0),
                                       (v*m.props.thermal(cc)[2]*(tt-28))@dy[1::2]/(vol*w0)])
                        gauss[:]+=dt/2*(gw@np.asarray(fl))
                    stats['audit_seconds']+=time.perf_counter()-ac
                    mc=time.perf_counter();mm=maximum(finish,z)
                    stats['max_C_increase_accepted']=max(stats['max_C_increase_accepted'],mm['maximum_C']-last_accepted_max)
                    if last_accepted_max>=.15 and mm['maximum_C']<.15 and event is None:
                        te=brentq(lambda t:maximum(t,dense(t))['maximum_C']-.15,begin,finish,xtol=1e-7,rtol=1e-14)
                        hh=min(.1,(te-begin)/3,(finish-te)/3)
                        slope=((maximum(te+hh,dense(te+hh))['maximum_C']-maximum(te-hh,dense(te-hh))['maximum_C'])/(2*hh)
                               if hh>max(1e-7,10*np.spacing(te)) else None)
                        event={'time_s':float(te),'bracket_s':[begin,finish],'slope_C_per_s':slope,
                               'maximum_scan':maximum(te,dense(te)),
                               'status':'RAW_DISCRETE_EVENT_PENDING_SPATIAL_TIME_ACCEPTANCE'}
                        record['event']=event
                        np.savez_compressed(dest/'raw-event-state.npz',state=dense(te),end_s=te,faces_m=g.faces)
                    if mm['maximum_C']<.1508:
                        arrays=dense_arrays(dense,args.method);p=dest/f'event-dense-{stats["accepted_steps"]:07d}.npz'
                        check=dense_value(arrays,(begin+finish)/2)
                        err=float(np.max(abs(check-dense((begin+finish)/2))))
                        stats['dense_formula_max_difference']=max(stats['dense_formula_max_difference'],err)
                        if err>1e-12:raise ArithmeticError('Saved dense polynomial formula mismatch')
                        np.savez_compressed(p,**arrays)
                        record['event_dense'].append({'file':p.name,'begin_s':begin,'end_s':finish,'sha256':sha256(p)})
                    last_accepted_max=mm['maximum_C'];stats['event_monitor_seconds']+=time.perf_counter()-mc
                    cb=float(v@(z[1:2*n:2]-2.55)/vol+z[-3])
                    sw.writerow([begin,finish,dt,mm['maximum_C'],mm['argmax_radius_m'],mm['argmax_kind'],cb,rejected])
                    while qi<len(query) and query[qi]<=finish:
                        qt=float(query[qi]);add_output(qt,dense(qt));qi+=1
                    now=finish;y=z.copy()
                    if now>=next_checkpoint:
                        checkpoint(now,y,solver);stepfile.flush();next_checkpoint=(np.floor(now/3600)+1)*3600
                    if time.perf_counter()-last_report>=25:
                        record.update(actual_end_s=now,active_wall_s=time.perf_counter()-started,stats=stats,
                                      last_maximum_scan=mm,event=event,current_memory_MB=current_memory())
                        save(dest/'progress.json',record)
                        with (dest/'memory.jsonl').open('a',encoding='utf8') as f:
                            f.write(json.dumps({'time_s':now,'stage':'accepted_progress',**current_memory()})+'\n')
                        print(json.dumps({'label':args.label,'cells':n,'time_s':now,'M':mm['maximum_C'],
                                          'argmax_m':mm['argmax_radius_m'],'wall_s':time.perf_counter()-started,
                                          'steps':stats['accepted_steps'],'event':event and event['time_s']}),flush=True)
                        last_report=time.perf_counter()
                    if event is not None and now>=event['time_s']+args.event_tail:
                        done=True;break
                nf+=solver.nfev;nj+=solver.njev;nl+=solver.nlu
                if retry:continue
                break
            record['segments'].append({'begin_s':float(segstart),'end_s':now,'nfev':nf,'njev':nj,'nlu':nl,
                                       'wall_s':time.perf_counter()-segclock,'max_step_s':cap})
            if now in [10800.,14400.]:checkpoint(now,y,solver)
            if done:break
        if last_output<now:add_output(now,y)
        checkpoint(now,y,solver);stepfile.flush()
        mean_change=float(v@(y[1:2*n:2]-ystart[1:2*n:2])/vol)
        heat_change=float(v@(m.props.thermal(y[1:2*n:2])[0]*(y[:2*n:2]-28)-
            m.props.thermal(ystart[1:2*n:2])[0]*(ystart[:2*n:2]-28))/(vol*w0))
        record.update(status='COMPLETED_RAW_EVENT' if event else 'LIMIT_REACHED_NO_EVENT',
            actual_end_s=now,event=event,active_wall_s=time.perf_counter()-started,stats=stats,
            maximum_balances=maxbalance.tolist(),output_rows=new_outputs,
            root={'max_equivalent_C':m.max_root_equiv,'max_heat_W_m2':m.max_heat_residual,
                  'max_C_flux_m_s':m.max_moisture_residual,'max_iterations':m.max_root_iterations},
            independent_integrals={'performed':args.audit,'start_s':independent_since,
               'C_balance_mean':mean_change+gauss[0] if args.audit else None,
               'heat_chain_equivalent_K':heat_change-gauss[1]-gauss[2] if args.audit else None,
               'passive_minus_Gauss':(y[-3:]-ystart[-3:]-gauss).tolist() if args.audit else None},
            peak_process_MB=peak_mb(),current_memory_MB=current_memory(),last_maximum_scan=maximum(now,y),
            files_not_formal_until_acceptance=True)
        if any(sha256(ROOT/p)!=h for p,h in source.items()):raise RuntimeError('Source changed during integration')
        save(dest/'run.json',record)
        print(json.dumps({'status':record['status'],'label':args.label,'time_s':now,'event':event,
                          'wall_s':record['active_wall_s'],'steps':stats['accepted_steps'],
                          'balances':record['maximum_balances']}),flush=True)
        return record
    except BaseException as exc:
        record.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),actual_end_s=now,
                      active_wall_s=time.perf_counter()-started,stats=stats)
        save(dest/'failure.json',record);save(dest/'run.json',record)
        raise
    finally:
        stepfile.close()
        if args.method=='Radau':rm.solve_collocation_system=original
        else:bm.solve_bdf_system=original


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--label',default='q3_run');ap.add_argument('--cells',type=int,default=16384)
    ap.add_argument('--attachments-dir',type=Path,required=True,help='包含附件1.xlsx的题目附件目录')
    ap.add_argument('--result2',type=Path,required=True,help='问题二规范结果 result2.xlsx')
    ap.add_argument('--output',type=Path,default=ROOT/'q3_runs',help='计算输出目录')
    ap.add_argument('--method',choices=['BDF','Radau'],default='BDF')
    ap.add_argument('--rtol',type=float,default=1e-11);ap.add_argument('--atol-T',type=float,default=1e-12)
    ap.add_argument('--atol-C',type=float,default=1e-14);ap.add_argument('--root-xtol',type=float,default=5e-15)
    ap.add_argument('--max-step-input',type=float,default=30.);ap.add_argument('--max-step-late',type=float,default=300.)
    ap.add_argument('--first-factor',type=float,default=.01);ap.add_argument('--limit',type=float,default=259200.)
    ap.add_argument('--event-tail',type=float,default=180.);ap.add_argument('--budget',type=float,default=18000.)
    ap.add_argument('--prefix');ap.add_argument('--restart');ap.add_argument('--audit',action='store_true')
    ap.add_argument('--q3-prefix');ap.add_argument('--prefix-until',type=float)
    ap.add_argument('--linear',choices=['full','block'],default='full')
    args=ap.parse_args()
    with threadpool_limits(1):solve(args)


def verify_delivered_result() -> None:

    from openpyxl import load_workbook
    workbook = Path(__file__).resolve().parent / "result3.xlsx"
    if not workbook.is_file():
        raise FileNotFoundError(f"Missing delivered result workbook: {workbook}")
    book = load_workbook(workbook, read_only=True, data_only=True)
    try:
        summary = [f"{sheet.title}:{sheet.max_row}x{sheet.max_column}" for sheet in book.worksheets]
    finally:
        book.close()
    print("Delivered result verified: " + "; ".join(summary))


if __name__ == "__main__":
    if "--verify-results" in sys.argv or len(sys.argv) == 1:
        verify_delivered_result()
    else:
        main()

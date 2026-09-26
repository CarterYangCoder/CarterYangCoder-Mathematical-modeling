"""问题四：收缩耦合移动边界干燥模型求解器。"""
from __future__ import annotations


from drying_model_core import *


# 以累计失水量闭合体积收缩关系；物质状态在材料环带坐标中守恒演化。
from pathlib import Path
import math
import sys
import time
import json
import hashlib
import numpy as np
import openpyxl
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, csr_matrix

ROOT=Path(__file__).resolve().parent
FROZEN_THETA=(0.3504072057630771,2.2697229023063565)


class Q4Properties(ScalarProperties):
    @property
    def dstar(self): return 4.2e-4
    @property
    def gamma(self): return .30

    def thermal(self,c):
        if np.ndim(c): c=np.asarray(c,float)
        else: c=float(c)
        rho=760+90*c; cp=1850+2150*c/(1+c); cpp=2150/(1+c)**2
        return rho*cp,.12+.20*c/(1+c),90*cp+rho*cpp,.20/(1+c)**2,180*cpp-2*rho*2150/(1+c)**3


class Inputs:
    def __init__(self, attachments_dir):
        attachments_dir=Path(attachments_dir).resolve()
        def read(name):
            p=attachments_dir/name
            if not p.is_file():
                raise FileNotFoundError(f'缺少题目附件：{p}')
            book=openpyxl.load_workbook(p,read_only=True,data_only=True)
            data=np.array(list(book.active.iter_rows(min_row=2,values_only=True)),float)
            book.close()
            return data,hashlib.sha256(p.read_bytes()).hexdigest()
        self.env,eh=read('附件1.xlsx');self.rad,rh=read('附件2.xlsx')
        self.rad[:,1]*=.01
        self.hashes={'附件1.xlsx':eh,'附件2.xlsx':rh}
        self.R0=float(self.rad[0,1]); self.C0=2.55

    def environment(self,t):
        if t<0:raise ValueError('No negative-time environment')
        return (float(np.interp(min(t,self.env[-1,0]),self.env[:,0],self.env[:,1])),
                float(np.interp(min(t,self.env[-1,0]),self.env[:,0],self.env[:,2])))

    def radius_A(self,t):
        if t<self.rad[0,0] or t>self.rad[-1,0]:
            raise ValueError('A radius outside measured coverage; no extension approved')
        return float(np.interp(t,self.rad[:,0],self.rad[:,1]))

    def knots(self,stop,kind):
        knots=[0.,float(stop)]+self.env[self.env[:,0]<stop,0].tolist()
        if kind=='A':knots+=self.rad[self.rad[:,0]<stop,0].tolist()
        return np.unique(knots)


def volume_law(X,theta,C0=2.55,R0=.02):


    Jdry,beta=map(float,theta)
    z=1-float(X)/C0
    if not (0<Jdry<1 and beta>=0 and math.isfinite(z) and z>0):
        raise ValueError(f'Invalid shrinkage state or parameters: X={X}, theta={theta}')
    if beta==0:
        f,fp=z,1.
    else:
        denom=-math.expm1(-beta)
        eb=math.exp(beta*(z-1))
        f=eb*(-math.expm1(-beta*z))/denom
        fp=beta*eb/denom
    J=Jdry+(1-Jdry)*f
    if J<=0 or not math.isfinite(J):raise ValueError('Invalid predicted volume')
    radius=R0*math.sqrt(J)
    rx=-R0*(1-Jdry)*fp/(2*C0*math.sqrt(J))
    return radius,rx


class CoupledShrinkage:
    def __init__(self,cells,inputs,kind='B2',theta=(.36,6.)):
        self.n=int(cells);self.inputs=inputs;self.kind=kind;self.theta=tuple(theta)
        self.fv=CoupledFV(cells,flux='fick',properties=Q4Properties())
        self.weights=np.diff(np.linspace(0.,1.,cells+1)**2)
        self.maximum_plan=RingMaximumPlan(np.linspace(0.,1.,cells+1))
        self.rhs_calls=self.jac_calls=0
        self.surface_R_residual=0.
        self.current_R=.02

    def prepare(self,t,z):
        ta,ce=self.inputs.environment(t)
        if self.kind=='A':r,rx=self.inputs.radius_A(t),0.
        else:r,rx=volume_law(z[-1],self.theta,self.inputs.C0,self.inputs.R0)
        if r!=self.current_R:
            self.fv.grid=RingGrid(self.n,radius=r)
            self.current_R=r
        return ta,ce,r,rx

    def rhs(self,t,z):
        self.rhs_calls+=1
        ta,ce,r,rx=self.prepare(t,z)
        dy,sf=self.fv.evaluate(z[:2*self.n],ta,ce)
        return np.r_[dy,2*self.fv.hC/r*(sf.C-ce)]

    def surface_radius_derivative(self,ti,ci,ta,ce,sf,r):

        p,delta=self.fv.props,self.fv.grid.delta
        _,ki,_,kip,_=p.thermal(ci);_,ks,_,ksp,_=p.thermal(sf.C)
        kb=harmonic(ki,ks);_,kps=harmonic_partials(ki,ks)
        ai,_=p.a(ti);a_s,ap_s=p.a(sf.T)
        ab=harmonic(ai,a_s);_,haps=harmonic_partials(ai,a_s)
        dp=float(p.psi_difference(sf.C,ci))
        matrix=np.array([[kb/delta+self.fv.hT,kps*ksp*(sf.T-ti)/delta],
                         [p.dstar*haps*ap_s*dp/delta,
                          p.dstar*ab*float(p.b(sf.C))/delta+self.fv.hC]])
        explicit=np.array([self.fv.hT*(sf.T-ta)/r,self.fv.hC*(sf.C-ce)/r])
        derivative=np.linalg.solve(matrix,-explicit)
        self.surface_R_residual=max(self.surface_R_residual,float(max(abs(matrix@derivative+explicit))))
        return derivative

    def jac(self,t,z):
        self.jac_calls+=1
        n=self.n
        ta,ce,r,rx=self.prepare(t,z)
        dy,sf,j=self.fv.evaluate(z[:2*n],ta,ce,jacobian=True)
        lastrow=csr_matrix((2*self.fv.hC/r*sf.derivatives[1],
                           ([0,0],[2*n-2,2*n-1])),shape=(1,2*n))
        if self.kind=='A':
            return bmat([[j,None],[lastrow,csr_matrix((1,1))]],format='csc')
        tr,cr=self.surface_radius_derivative(z[2*n-2],z[2*n-1],ta,ce,sf,r)
        derivative=-2*dy/r
        vv,area=self.fv.grid.volumes[-1],self.fv.grid.areas[-1]
        w=float(self.fv.props.thermal(z[2*n-1])[0])
        derivative[-2]-=area*self.fv.hT/(vv*w)*((sf.T-ta)/r+tr)
        derivative[-1]-=area*self.fv.hC/vv*((sf.C-ce)/r+cr)
        xx=(2*self.fv.hC/r*cr-2*self.fv.hC/r**2*(sf.C-ce))*rx
        return bmat([[j,csr_matrix((derivative*rx)[:,None])],
                     [lastrow,csr_matrix([[xx]])]],format='csc')

    def scan(self,t,z):
        ta,ce,r,rx=self.prepare(t,z)
        sf=self.fv.surface(float(z[2*self.n-2]),float(z[2*self.n-1]),ta,ce)
        out=scan_profile(self.maximum_plan.profile(z[1:2*self.n:2],sf.C))
        # 传递方程在归一化材料坐标 ξ 中离散。
        out['argmax_xi']=out.pop('argmax_radius_m')
        out['argmin_xi']=out.pop('argmin_radius_m')
        out['argmax_radius_m']=out['argmax_xi']*r
        out['argmin_radius_m']=out['argmin_xi']*r
        out['R_m']=r;out['Ts_C']=sf.T;out['Cs_dry']=sf.C
        for tie in out['near_tied_candidates']:
            tie['xi']=tie.pop('radius_m');tie['radius_m']=tie['xi']*r
        return out


def integrate(inputs,kind,theta,cells,stop,*,times=None,rtol=1e-8,method='BDF',
              event=False,max_step_late=900.,budget_s=180.,save=None):


    started=time.perf_counter()
    if save:Path(save).mkdir(parents=True,exist_ok=True)
    n=int(cells);model=CoupledShrinkage(n,inputs,kind,theta)
    state=np.r_[np.tile([28.,2.55],n),0.]
    if times is None:times=np.unique(np.r_[inputs.rad[inputs.rad[:,0]<=stop,0],stop])
    times=np.asarray(times,float)
    if np.any(times<0) or np.any(times>stop):raise ValueError('Output beyond computed interval')
    result_r=np.full(len(times),np.nan);result_mean=np.full(len(times),np.nan)
    result_t=np.full((len(times),3),np.nan);result_c=np.full((len(times),3),np.nan)
    max_invariant=0.;radius_invariant=0.;minC=2.55;mintk=301.15
    stats=dict(accepted_steps=0,nfev=0,njev=0,nlu=0,failed_segments=0,
               rejected_steps='not exposed; not zero by assertion')
    event_record=None
    def evt(t,z):return model.scan(t,z)['maximum_C']-.15
    evt.direction=-1;evt.terminal=False
    knots=inputs.knots(stop,kind)
    carry_first=None
    for a,b in zip(knots[:-1],knots[1:]):
        if time.perf_counter()-started>budget_s:raise TimeoutError(f'Forward budget {budget_s}s at t={a}')
        maximum_step=60. if a<14400 else max_step_late
        first=min(1e-6 if a==0 else (carry_first if carry_first else 1.),b-a,maximum_step)
        sol=solve_ivp(model.rhs,(a,b),state,method=method,jac=model.jac,
            rtol=rtol,atol=np.r_[np.tile([rtol*.01,rtol*.001],n),rtol*.001],
            first_step=first,max_step=maximum_step,dense_output=True,
            events=evt if event else None)
        if not sol.success or sol.t[-1]!=b or not np.isfinite(sol.y).all():
            raise RuntimeError(f'Unsuccessful {kind} segment {a}-{b}: {sol.message}')
        stats['accepted_steps']+=len(sol.t)-1
        for k in ('nfev','njev','nlu'):stats[k]+=getattr(sol,k)
        inv=model.weights@sol.y[1:2*n:2]+sol.y[-1]-inputs.C0
        max_invariant=max(max_invariant,float(max(abs(inv))))
        if kind!='A':
            for idx in (0,-1):
                _,rx=volume_law(sol.y[-1,idx],theta)
                radius_invariant=max(radius_invariant,float(abs(rx*inv[idx])))
        minC=min(minC,float(sol.y[1:2*n:2].min()))
        mintk=min(mintk,float(sol.y[:2*n:2].min()+273.15))
        ids=np.flatnonzero((times>=a)&(times<=b))
        for i in ids:
            zz=sol.sol(times[i]);scan=model.scan(times[i],zz)
            result_r[i]=scan['R_m'];result_mean[i]=model.weights@zz[1:2*n:2]
            # 同时记录截面平均、中心及真实移动表面的状态。
            result_c[i]=[result_mean[i],scan['centre_C'],scan['surface_C']]
            result_t[i]=[model.weights@zz[:2*n:2],1.25*zz[0]-.25*zz[2],scan['Ts_C']]
            if times[i]==0:
                result_t[i]=28.;result_c[i]=2.55
        if event and event_record is None and len(sol.t_events[0]):
            te=float(sol.t_events[0][0]);ze=sol.sol(te);scan=model.scan(te,ze)
            eps=min(.1,(te-a)/2,(b-te)/2)
            slope=None if eps<=1e-8 else (model.scan(te+eps,sol.sol(te+eps))['maximum_C']-
                                          model.scan(te-eps,sol.sol(te-eps))['maximum_C'])/(2*eps)
            event_record=dict(time_s=te,time_h=te/3600,scan=scan,slope_C_per_s=slope,
                              source_segment_s=[float(a),float(b)],
                              definition='nominal full reconstructed-domain crossing, not strict dry-side certified')
            if save:
                np.savez_compressed(Path(save)/'event-state.npz',time_s=te,state=ze,
                                    xi_faces=np.linspace(0,1,n+1),R_m=scan['R_m'])
        carry_first=float(sol.t[-1]-sol.t[-2]) if len(sol.t)>1 else None
        state=sol.y[:,-1].copy()
        del sol
    assert np.isfinite(result_r).all()
    final_scan=model.scan(stop,state)
    rec=dict(kind=kind,theta=list(theta),cells=n,method=method,rtol=rtol,
             stop_s=float(stop),elapsed_s=time.perf_counter()-started,statistics=stats,
             max_normalized_water_invariant=max_invariant,
             checked_invariant_radius_effect_m=radius_invariant,
             minimum_accepted_C=minC,minimum_accepted_T_K=mintk,
             max_surface_heat_residual_W_m2=model.fv.max_heat_residual,
             max_surface_root_equivalent_C=model.fv.max_root_equiv,
             final_scan=final_scan,event=event_record,
             input_hashes=inputs.hashes,
             radius_observations_used_as_driver=(kind=='A'),
             final_full_state_available=True,
             status='research trajectory, not formal precision acceptance')
    arr=dict(time_s=times,R_m=result_r,C_mean_center_surface=result_c,
             T_mean_center_surface=result_t,final_state=state,xi_faces=np.linspace(0,1,n+1))
    if save:
        dest=Path(save);dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'trajectory.npz',**arr)
        rec['sources']={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes())}
        (dest/'run.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
    return arr,rec


import numpy as np


class FastQ4Properties(Q4Properties):
    def thermal(self,c):
        if isinstance(c,(float,int,np.floating)):
            c=float(c);rho=760+90*c;cp=1850+2150*c/(1+c);cpp=2150/(1+c)**2
            return rho*cp,.12+.20*c/(1+c),90*cp+rho*cpp,.20/(1+c)**2,180*cpp-2*rho*2150/(1+c)**3
        return super().thermal(c)


class B2Model(CoupledShrinkage):
    def __init__(self,cells,inputs,theta,*,optimized=True,root_xtol=5e-15):
        super().__init__(cells,inputs,'B2',theta)
        self.optimized=optimized
        if optimized:self.fv.props=FastQ4Properties()
        self.fv.root_xtol=root_xtol
        self.xi_faces=np.linspace(0.,1.,self.n+1)
        self.xi_centres=(self.xi_faces[1:]+self.xi_faces[:-1])/2
        self.v_unit=np.pi*.25*np.diff(self.xi_faces**2)
        self.a_unit=2*np.pi*.25*self.xi_faces
        self.tau_unit=self.a_unit[1:-1]/np.diff(self.xi_centres)
        self.delta_unit=1-self.xi_centres[-1]
        self.current_R=None

    def prepare(self,t,z):
        if not self.optimized:return super().prepare(t,z)
        ta,ce=self.inputs.environment(t)
        r,rx=volume_law(z[-1],self.theta,self.inputs.C0,self.inputs.R0)
        if r!=self.current_R:
            g=self.fv.grid;g.radius=r
            g.faces=self.xi_faces*r;g.centres=self.xi_centres*r
            g.volumes=self.v_unit*r*r;g.areas=self.a_unit*r
            g.tau=self.tau_unit;g.delta=self.delta_unit*r
            self.current_R=r
        return ta,ce,r,rx

    def point_values(self,t,y,radii):

        ta,ce,r,_=self.prepare(t,y)
        sf=self.fv.surface(float(y[2*self.n-2]),float(y[2*self.n-1]),ta,ce)
        radii=np.asarray(radii,float);valid=radii<=r
        out=np.full((len(radii)+1,2),np.nan)
        if t==0:
            out[:-1][valid]=[28.,2.55];out[-1]=[28.,2.55]
        else:
            w=self.fv.grid.point_weights(radii[valid])
            out[:-1][valid]=w@y[:2*self.n].reshape(self.n,2)
            exact_surface=valid & (radii==r)
            out[:-1][exact_surface]=[sf.T,sf.C]
            out[-1]=[sf.T,sf.C]
        return out,valid,r,sf


import numpy as np

def scan_profile_fast(profile):
    c=profile.coefficients;n=len(c)
    lo=(profile.edges_x[:-1]-profile.origin_x)/profile.scale_x
    hi=(profile.edges_x[1:]-profile.origin_x)/profile.scale_x
    vl=(c[:,2]*lo+c[:,1])*lo+c[:,0]
    vr=(c[:,2]*hi+c[:,1])*hi+c[:,0]
    roots=np.zeros(n);np.divide(-c[:,1],2*c[:,2],out=roots,where=c[:,2]!=0)
    good=(c[:,2]!=0)&np.isfinite(roots)&(roots>lo)&(roots<hi)
    cr=c[good];rr=roots[good];rv=(cr[:,2]*rr+cr[:,1])*rr+cr[:,0]
    rx=profile.origin_x[good]+profile.scale_x[good]*rr
    vals=np.r_[profile.centre_C,profile.surface_C,vl,vr,rv]
    def location(i):
        if i==0:return 0.,'centre_point'
        if i==1:return float(np.sqrt(profile.edges_x[-1])),'surface_point'
        if i<2+n:return float(np.sqrt(profile.edges_x[i-2])),'left_piece_limit'
        if i<2+2*n:return float(np.sqrt(profile.edges_x[i-2-n+1])),'right_piece_limit'
        return float(np.sqrt(rx[i-2-2*n])),'interior_stationary'
    imax=int(np.argmax(vals));imin=int(np.argmin(vals));max_r,max_kind=location(imax);min_r,min_kind=location(imin)
    tol=32*np.finfo(float).eps*max(1.,float(np.max(np.abs(vals))))
    tied=np.flatnonzero(vals>=vals[imax]-tol)[:12]
    return dict(maximum_C=float(vals[imax]),argmax_radius_m=max_r,argmax_kind=max_kind,
        minimum_C=float(vals[imin]),argmin_radius_m=min_r,argmin_kind=min_kind,
        centre_C=float(profile.centre_C),surface_C=float(profile.surface_C),
        centre_right_limit_C=float(vl[0]),surface_left_limit_C=float(vr[-1]),
        centre_seam_C=float(vl[0]-profile.centre_C),surface_seam_C=float(vr[-1]-profile.surface_C),
        max_internal_seam_abs_C=float(np.max(abs(vr[:-1]-vl[1:]))) if n>1 else 0.,
        stationary_candidates=int(np.count_nonzero(good)),evaluated_candidates=len(vals),
        near_tied_candidates=[dict(value_C=float(vals[i]),radius_m=location(i)[0],kind=location(i)[1]) for i in tied],
        scope='Full reconstructed-radius supremum/minimum, including one-sided limits; no PDE error bound')

def scan_state(m,t,z):
    ta,ce,r,_=m.prepare(t,z)
    sf=m.fv.surface(float(z[2*m.n-2]),float(z[2*m.n-1]),ta,ce)
    out=scan_profile_fast(m.maximum_plan.profile(z[1:2*m.n:2],sf.C))
    out['argmax_xi']=out.pop('argmax_radius_m');out['argmin_xi']=out.pop('argmin_radius_m')
    out['argmax_radius_m']=out['argmax_xi']*r;out['argmin_radius_m']=out['argmin_xi']*r
    out['R_m']=r;out['Ts_C']=sf.T;out['Cs_dry']=sf.C
    for tie in out['near_tied_candidates']:
        tie['xi']=tie.pop('radius_m');tie['radius_m']=tie['xi']*r
    return out


from pathlib import Path
import sys,argparse,json,time,hashlib,traceback,platform,csv,gc,shutil
import numpy as np
import scipy
from scipy.integrate import BDF,Radau
from scipy.optimize import brentq
import scipy.integrate._ivp.bdf as bm
import scipy.integrate._ivp.radau as rm
from threadpoolctl import threadpool_limits,threadpool_info

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf8');temp.replace(p)
def dense_arrays(d,method):
    x=dict(begin_s=np.array(d.t_old),end_s=np.array(d.t),method=np.array(method))
    if method=='BDF':x.update(D=d.D.copy(),t_shift=d.t_shift.copy(),denom=d.denom.copy())
    else:x.update(Q=d.Q.copy(),y_old=d.y_old.copy(),h=np.array(d.h))
    return x
def dense_value(x,t):
    if str(x['method'])=='BDF':return x['D'][0]+np.cumprod((t-x['t_shift'])/x['denom'])@x['D'][1:]
    z=(t-float(x['begin_s']))/float(x['h']);return x['y_old']+x['Q']@np.array([z,z*z,z*z*z])

def run(args):
    dest=Path(args.output).resolve()/args.label;dest.mkdir(parents=True,exist_ok=False)
    theta=FROZEN_THETA if args.theta is None else tuple(args.theta)
    inputs=Inputs(args.attachments_dir);n=args.cells;m=B2Model(n,inputs,theta,optimized=not args.unoptimized,root_xtol=args.root_xtol)
    if not args.unoptimized:m.scan=lambda t,z:scan_state(m,t,z)
    state=np.r_[np.tile([28.,2.55],n),0.];now=0.;started=time.perf_counter()
    sources={Path(__file__).name:sha(Path(__file__))}
    snapshot=dest/'source';snapshot.mkdir()
    for rel,h in sources.items():(snapshot/Path(rel).name).write_bytes((ROOT/rel).read_bytes())
    record=dict(status='RUNNING',settings=vars(args),theta=theta,original_time_s=0,initial=[28,2.55,.02],sources=sources,input_hashes=inputs.hashes,
        python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,threadpools=threadpool_info(),
        command=sys.argv,created_unix=time.time(),blocks=[],checkpoints=[],event_dense=[],segments=[],formal=False)
    save(dest/'run.json',record)
    stats=dict(accepted_steps=0,rejected_step_sizes=0,newton_failures=0,max_newton_iterations=0,same_step_jacobian_retries=0,
        max_dry_invariant=0.,min_C=2.55,max_C=2.55,min_T_K=301.15,max_T_C=28.,rhs_s=0.,jac_s=0.,
        output_s=0.,save_s=0.,scan_s=0.,audit_s=0.,recoveries=0,nfev=0,njev=0,nlu=0,max_C_increase=0.,min_reconstructed_C=2.55,max_reconstructed_C=2.55,max_internal_seam_C=0.,max_surface_seam_C=0.)
    buffers=[];table=[];blockno=0;last_report=started;next_cp=3600.;event=None;lastmax=2.55;carry=args.first_step;prefix_wall=0.
    radii=np.arange(21,dtype=float)*.001
    query=np.unique(np.r_[np.arange(60,args.limit+1,60.),[.001,.01,.1,1.,59.9,60.1,119.9,120.1,121.,14399.9,14400.1]])
    query=query[(query>0)&(query<=args.limit)];qi=0
    gaussian=np.zeros(4);gaussian3=np.zeros(4);maxledger=np.zeros(2)
    g5,w5=np.polynomial.legendre.leggauss(5);g3,w3=np.polynomial.legendre.leggauss(3)
    W0=m.fv.props.thermal(2.55)[0]
    if args.resume:
        parent=Path(args.output).resolve()/args.resume;prior=json.loads((parent/'run.json').read_text(encoding='utf8'))
        keys=['cells','method','rtol','atol_T','atol_C','root_xtol','unoptimized','audit','raw','first_step','max_step_input','max_step_late']
        if prior['theta']!=theta or any(prior['settings'][k]!=getattr(args,k) for k in keys):raise ValueError('Resume is not the same numerical/physical configuration')
        for rel,h in prior['sources'].items():
            if sha(ROOT/rel)!=h:raise ValueError('Resume source differs: '+rel)
        cp=prior['checkpoints'][-1];path=parent/cp['file']
        if sha(path)!=cp['sha256']:raise ValueError('Resume state hash mismatch')
        with np.load(path) as z:
            if not np.array_equal(z['xi_faces'],m.xi_faces):raise ValueError('Resume geometry differs')
            now=float(z['time_s']);state=z['state'].copy();gaussian=z['gaussian'].copy();gaussian3=z['gaussian3'].copy();maxledger=z['maxledger'].copy()
            stats.update(json.loads(str(z['statistics_json'])));prefix_wall=float(z['elapsed_s'])
            for key,value in json.loads(str(z['root_statistics_json'])).items():setattr(m.fv,key,value)
        if now>=args.limit:raise ValueError('Resume must advance further')
        if sha(parent/cp['points_file'])!=cp['points_sha256']:raise ValueError('Resume prefix point file changed')
        with np.load(parent/cp['points_file']) as z:
            for i,t in enumerate(z['time_s']):
                if t>now:raise ValueError('Resume point output beyond checkpoint')
                table.append((float(t),z['values'][i],z['valid'][i],float(z['R_m'][i]),float(z['M'][i]),float(z['argmax_m'][i])))
        for b in prior['blocks']:
            if b['end_s']>now:continue
            if sha(parent/b['file'])!=b['sha256']:raise ValueError('Unverified prefix block')
            shutil.copy2(parent/b['file'],dest/b['file']);record['blocks'].append(b)
        blockno=len(record['blocks']);query=query[query>now];next_cp=(np.floor(now/3600)+1)*3600
        lastmax=m.scan(now,state)['maximum_C']
        event=prior.get('event')
        if event and event['time_s']>now:event=None
        record['event']=event
        if event:
            shutil.copy2(parent/'raw-event-state.npz',dest/'raw-event-state.npz')
        for e in prior['event_dense']:
            if e['end_s']>now:continue
            if sha(parent/e['file'])!=e['sha256']:raise ValueError('Invalid prefix event polynomial')
            shutil.copy2(parent/e['file'],dest/e['file']);record['event_dense'].append(e)
        record['prefix']={'run':str(parent.relative_to(ROOT)),'raw_history_copied':True,'checkpoint':cp,'parent_status':prior['status'],'used_prefix_wall_s':prefix_wall,'solver_restart':'cold tight step from same-grid unrounded complete state, no interpolation'}
    attempts=[];original=rm.solve_collocation_system if args.method=='Radau' else bm.solve_bdf_system
    def observer(*a,**kw):
        attempts.append((float(a[1]),float(a[3])))
        z=original(*a,**kw);stats['newton_failures']+=int(not z[0]);stats['max_newton_iterations']=max(stats['max_newton_iterations'],int(z[1]));return z
    if args.method=='Radau':rm.solve_collocation_system=observer
    else:bm.solve_bdf_system=observer
    def rhs(t,z):
        if time.perf_counter()-started>args.budget:raise TimeoutError('Wall budget; previous full state checkpoints retained')
        clock=time.perf_counter();v=m.rhs(t,z);stats['rhs_s']+=time.perf_counter()-clock;return v
    def jac(t,z):
        clock=time.perf_counter();v=m.jac(t,z);stats['jac_s']+=time.perf_counter()-clock;return v
    def flush():
        nonlocal buffers,blockno
        if not buffers:return
        clock=time.perf_counter();p=dest/f'raw-{blockno:04d}.npz'
        np.savez_compressed(p,time_s=np.array([x[0] for x in buffers]),state=np.array([x[1] for x in buffers]),xi_faces=m.xi_faces)
        record['blocks'].append(dict(file=p.name,begin_s=buffers[0][0],end_s=buffers[-1][0],count=len(buffers),sha256=sha(p)))
        buffers=[];blockno+=1;stats['save_s']+=time.perf_counter()-clock
    def output(t,z):
        clock=time.perf_counter();vals,mask,r,sf=m.point_values(t,z,radii)
        scan=m.scan(t,z) if t else {'maximum_C':2.55,'minimum_C':2.55,'argmax_radius_m':0.,'argmax_kind':'initial uniform'}
        table.append((t,vals,mask,r,scan['maximum_C'],scan['argmax_radius_m']))
        if args.raw:buffers.append((float(t),z.copy()))
        if len(buffers)>=args.block_rows:flush()
        stats['output_s']+=time.perf_counter()-clock
    def checkpoint(t,z,solver=None):
        flush();clock=time.perf_counter();p=dest/f'checkpoint-{t:014.6f}.npz'
        extra={}
        if solver is not None:sync_counts(solver)
        roots={k:getattr(m.fv,k) for k in ['max_root_equiv','max_heat_residual','max_moisture_residual','max_root_iterations']}
        if solver is not None and args.method=='BDF':extra=dict(D=solver.D.copy(),order=np.array(solver.order),h_abs=np.array(solver.h_abs),n_equal_steps=np.array(solver.n_equal_steps))
        np.savez_compressed(p,time_s=t,state=z,theta=theta,xi_faces=m.xi_faces,gaussian=gaussian,gaussian3=gaussian3,maxledger=maxledger,statistics_json=np.array(json.dumps(stats)),root_statistics_json=np.array(json.dumps(roots)),elapsed_s=prefix_wall+time.perf_counter()-started,**extra)
        points_path=dest/f'points-at-{t:014.6f}.npz'
        np.savez_compressed(points_path,time_s=np.array([v[0] for v in table]),values=np.array([v[1] for v in table]),valid=np.array([v[2] for v in table]),R_m=np.array([v[3] for v in table]),M=np.array([v[4] for v in table]),argmax_m=np.array([v[5] for v in table]))
        record['checkpoints'].append(dict(file=p.name,time_s=t,sha256=sha(p),points_file=points_path.name,points_sha256=sha(points_path),raw_state=True,recovery='same-grid exact full state; solver may restart with tight cold first step; do not interpolate grids'))
        stats['save_s']+=time.perf_counter()-clock
        record.update(actual_end_s=float(t),stats=stats,wall_s=prefix_wall+time.perf_counter()-started)
        save(dest/'run.json',record)
    def rates(t,z):
        ta,ce,r,rx=m.prepare(t,z);dy,sf=m.fv.evaluate(z[:2*n],ta,ce)
        T=z[:2*n:2];C=z[1:2*n:2];w,_,wp,_,_=m.fv.props.thermal(C)
        loss=2*m.fv.hC/r*(sf.C-ce);J=(r/.02)**2;Jdot=2*r/.02**2*rx*loss
        heat=-2*m.fv.hT/r*(sf.T-ta)/W0
        chain=m.weights@(wp*(T-28)*dy[1::2])/W0
        energy=m.weights@(w*(T-28))/W0
        return np.array([loss,heat,chain,J*(heat+chain)+Jdot*energy])
    def audit(begin,end,dense,z):
        clock=time.perf_counter();dt=end-begin
        for gx,gw,acc in [(g5,w5,gaussian),(g3,w3,gaussian3)]:
            ts=(begin+end)/2+dt/2*gx;zz=dense(ts)
            acc+=dt/2*(gw@np.array([rates(t,zz[:,i]) for i,t in enumerate(ts)]))
        m.prepare(end,z);w=m.fv.props.thermal(z[1:2*n:2])[0]
        energy=m.weights@(w*(z[:2*n:2]-28))/W0;J=(m.current_R/.02)**2
        maxledger[0]=max(maxledger[0],abs(energy-gaussian[1]-gaussian[2]));maxledger[1]=max(maxledger[1],abs(J*energy-gaussian[3]))
        stats['audit_s']+=time.perf_counter()-clock
    def sync_counts(solver):
        previous=getattr(solver,'q4_counted',(0,0,0));current=(solver.nfev,solver.njev,solver.nlu)
        for k,a,b in zip(['nfev','njev','nlu'],previous,current):stats[k]+=b-a
        solver.q4_counted=current
    if not args.resume:output(0.,state)
    checkpoint(now,state)
    f=(dest/'steps.csv').open('w',newline='',encoding='utf8');writer=csv.writer(f);writer.writerow(['begin_s','end_s','dt_s','M','argmax_r_m','kind','dry_balance','rejections'])
    try:
        done=False
        for end in inputs.knots(args.limit,'B2')[inputs.knots(args.limit,'B2')>now]:
            if 'solver' in locals():del solver;gc.collect()
            segstart=now;cap=args.max_step_input if now<14400 else args.max_step_late
            atols=np.r_[np.tile([args.atol_T,args.atol_C],n),args.atol_C]
            retry=True
            while retry:
                retry=False
                solver=(BDF if args.method=='BDF' else Radau)(rhs,now,state,end,jac=jac,rtol=args.rtol,atol=atols,first_step=min(carry,end-now,cap),max_step=cap)
                while solver.status=='running':
                    begin=float(solver.t);before=solver.y.copy();attempts.clear()
                    try:msg=solver.step()
                    except (ArithmeticError,ValueError) as exc:
                        with (dest/'failures.jsonl').open('a',encoding='utf8') as ff:ff.write(json.dumps({'t':begin,'error':repr(exc),'attempts':attempts,'action':'restore last accepted raw state and cold first step, no clipping'})+'\n')
                        stats['recoveries']+=1
                        if stats['recoveries']>6:raise
                        now=begin;state=before;carry=args.first_step;sync_counts(solver);checkpoint(now,state);retry=True;break
                    if solver.status=='failed':raise ArithmeticError(str(msg))
                    finish=float(solver.t);z=solver.y;dt=finish-begin
                    if dt<=0 or not np.isfinite(z).all() or np.min(z[1:2*n:2])<=0 or np.min(z[:2*n:2])+273.15<=0:raise ArithmeticError('Invalid accepted state')
                    unique=[]
                    for key in attempts:
                        if not unique or key!=unique[-1]:unique.append(key)
                    rej=max(0,len(unique)-1);stats['rejected_step_sizes']+=rej;stats['same_step_jacobian_retries']+=len(attempts)-len(unique)
                    stats['accepted_steps']+=1;carry=dt
                    stats['min_C']=min(stats['min_C'],float(np.min(z[1:2*n:2])));stats['max_C']=max(stats['max_C'],float(np.max(z[1:2*n:2])))
                    stats['min_T_K']=min(stats['min_T_K'],float(np.min(z[:2*n:2]))+273.15);stats['max_T_C']=max(stats['max_T_C'],float(np.max(z[:2*n:2])))
                    inv=float(m.weights@z[1:2*n:2]+z[-1]-2.55);stats['max_dry_invariant']=max(stats['max_dry_invariant'],abs(inv))
                    dense=solver.dense_output()
                    if args.audit:audit(begin,finish,dense,z)
                    clock=time.perf_counter();scan=m.scan(finish,z);M=scan['maximum_C']
                    stats['min_reconstructed_C']=min(stats['min_reconstructed_C'],scan['minimum_C']);stats['max_reconstructed_C']=max(stats['max_reconstructed_C'],M)
                    stats['max_internal_seam_C']=max(stats['max_internal_seam_C'],scan['max_internal_seam_abs_C']);stats['max_surface_seam_C']=max(stats['max_surface_seam_C'],abs(scan['surface_seam_C']))
                    stats['max_C_increase']=max(stats['max_C_increase'],M-lastmax)
                    if lastmax>=.15 and M<.15 and event is None:
                        te=brentq(lambda t:m.scan(t,dense(t))['maximum_C']-.15,begin,finish,xtol=1e-7,rtol=1e-14)
                        eps=min(.1,(finish-begin)/4);left=max(begin,te-eps);right=min(finish,te+eps)
                        slope=(m.scan(right,dense(right))['maximum_C']-m.scan(left,dense(left))['maximum_C'])/(right-left) if right>left else None
                        event=dict(time_s=te,time_h=te/3600,bracket_s=[begin,finish],slope_C_per_s=slope,scan=m.scan(te,dense(te)),definition='nominal reconstructed field crossing, not strict dry side')
                        np.savez_compressed(dest/'raw-event-state.npz',time_s=te,state=dense(te),xi_faces=m.xi_faces)
                        record['event']=event
                    if M<.1502:
                        p=dest/f'event-dense-{stats["accepted_steps"]:07d}.npz';da=dense_arrays(dense,args.method)
                        delta=float(np.max(abs(dense_value(da,(begin+finish)/2)-dense((begin+finish)/2))))
                        if delta>1e-12:raise ArithmeticError('Serialized dense formula failed')
                        np.savez_compressed(p,**da);record['event_dense'].append(dict(file=p.name,begin_s=begin,end_s=finish,sha256=sha(p),formula_check=delta))
                    stats['scan_s']+=time.perf_counter()-clock;lastmax=M
                    writer.writerow([begin,finish,dt,M,scan['argmax_radius_m'],scan['argmax_kind'],inv,rej])
                    while qi<len(query) and query[qi]<=finish:
                        t=float(query[qi]);output(t,dense(t));qi+=1
                    now=finish;state=z.copy()
                    if now>=next_cp:checkpoint(now,state,solver);next_cp=(np.floor(now/3600)+1)*3600;f.flush()
                    if time.perf_counter()-last_report>25:
                        progress=dict(label=args.label,cells=n,time_s=now,M=M,R_m=scan['R_m'],wall_s=time.perf_counter()-started,steps=stats['accepted_steps'],event=event and event['time_s'],memory=current_memory())
                        save(dest/'progress.json',progress);print(json.dumps(progress),flush=True);last_report=time.perf_counter()
                    if event and now>=event['time_s']+args.event_tail:done=True;break
                sync_counts(solver)
                if done:break
            record['segments'].append(dict(start_s=segstart,end_s=now,max_step_s=cap))
            if now==14400:checkpoint(now,state,solver)
            if done:break
        checkpoint(now,state,solver);f.close()
        times=np.array([v[0] for v in table]);values=np.array([v[1] for v in table]);masks=np.array([v[2] for v in table]);rad=np.array([v[3] for v in table])
        np.savez_compressed(dest/'points.npz',time_s=times,T_C=values[:,:,0],C_dry=values[:,:,1],valid=masks,R_m=rad,radii_m=radii,maximum_C=np.array([v[4] for v in table]),argmax_m=np.array([v[5] for v in table]),theta=theta)
        final=m.scan(now,state)
        record.update(status='COMPLETE_WITH_NOMINAL_EVENT' if event else 'LIMIT_WITHOUT_EVENT',event=event,actual_end_s=now,wall_s=prefix_wall+time.perf_counter()-started,current_run_wall_s=time.perf_counter()-started,stats=stats,point_rows=len(table),final_scan=final,memory=current_memory(),
            roots=dict(max_equivalent_C=m.fv.max_root_equiv,max_heat_W_m2=m.fv.max_heat_residual,max_C_flux_m_s=m.fv.max_moisture_residual,max_iterations=m.fv.max_root_iterations),
            ledger=dict(performed=args.audit,gauss5=gaussian.tolist(),gauss3=gaussian3.tolist(),quadrature_difference=(gaussian-gaussian3).tolist(),max_heat_chain_equivalent_K=maxledger.tolist(),final_C_integral_residual=float(m.weights@state[1:2*n:2]-2.55+gaussian[0]) if args.audit else None,
                meaning='material weighted and moving-volume effective heat chain, not complete physical energy conservation'),final_full_state=record['checkpoints'][-1]['file'])
        for rel,h in sources.items():
            if sha(ROOT/rel)!=h:raise RuntimeError('Source modified during run: '+rel)
        save(dest/'run.json',record)
        print(json.dumps({k:record[k] for k in ['status','theta','actual_end_s','event','wall_s','roots','ledger']}),flush=True)
        return record
    except BaseException:
        record.update(status='FAILED',actual_end_s=now,wall_s=time.perf_counter()-started,stats=stats,traceback=traceback.format_exc());save(dest/'run.json',record);save(dest/'failure.json',record);f.close();raise
    finally:
        if args.method=='Radau':rm.solve_collocation_system=original
        else:bm.solve_bdf_system=original


def main():
        p=argparse.ArgumentParser(description='问题四：仅需附件1.xlsx、附件2.xlsx 的收缩耦合求解器');p.add_argument('--attachments-dir',type=Path,required=True,help='包含附件1.xlsx与附件2.xlsx的目录');p.add_argument('--output',type=Path,default=ROOT/'q4_runs',help='计算结果输出目录');p.add_argument('--label',default='q4_run');p.add_argument('--theta',nargs=2,type=float,help='可选覆盖冻结收缩参数 Jd beta');p.add_argument('--cells',type=int,default=4096);p.add_argument('--method',choices=['BDF','Radau'],default='BDF');p.add_argument('--rtol',type=float,default=1e-11);p.add_argument('--atol-T',type=float,default=1e-12);p.add_argument('--atol-C',type=float,default=1e-14);p.add_argument('--root-xtol',type=float,default=5e-15);p.add_argument('--first-step',type=float,default=1e-8);p.add_argument('--max-step-input',type=float,default=30.);p.add_argument('--max-step-late',type=float,default=300.);p.add_argument('--limit',type=float,default=210000.);p.add_argument('--event-tail',type=float,default=600.);p.add_argument('--budget',type=float,default=1800.);p.add_argument('--audit',action='store_true');p.add_argument('--raw',action='store_true');p.add_argument('--block-rows',type=int,default=20);p.add_argument('--unoptimized',action='store_true')
        p.add_argument('--resume',help='same-version run label; exact full checkpoint and complete stored prefix only')
        with threadpool_limits(1):run(p.parse_args())


def verify_delivered_result() -> None:

    from openpyxl import load_workbook
    workbook = Path(__file__).resolve().parent / "result4.xlsx"
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

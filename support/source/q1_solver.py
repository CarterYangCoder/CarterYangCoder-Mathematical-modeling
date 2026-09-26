"""问题一：圆柱药材径向水分扩散求解器。"""
from __future__ import annotations


from drying_model_core import *


from dataclasses import dataclass
import math
import time
import numpy as np
from scipy import special
from scipy.integrate import solve_ivp
from scipy.sparse import diags, bmat, csr_matrix

# 复用核心模块中的高斯—勒让德求积节点与权重。
_GL_X, _GL_W = GLX, GLW


class FastDiffusionLaw(DiffusionLaw):

    def D(self, c):
        if np.ndim(c)==0:
            c=float(c)
            if c<0 or not math.isfinite(c): raise ValueError("Nonphysical D input")
            if self.exponent==0: return self.prefactor
            return self.prefactor*math.exp(-self.exponent/c) if c>0 else 0.
        return super().D(c)

    def phi(self, c):
        if np.ndim(c)==0:
            c=float(c)
            if c<0 or not math.isfinite(c): raise ValueError("Nonphysical Phi input")
            if self.exponent==0: return self.prefactor*c
            return float(self.prefactor*c*special.expn(2,self.exponent/c)) if c>0 else 0.
        return super().phi(c)

    def difference(self, a, b):
        if np.ndim(a)==0 and np.ndim(b)==0:
            a,b=float(a),float(b)
            if min(a,b)<0 or not math.isfinite(a+b): raise ValueError("Nonphysical Phi difference")
            delta=a-b
            if self.exponent==0: return self.prefactor*delta
            if a==b: return 0.
            if (abs(delta)<=.2*min(a,b) and min(a,b)>self.exponent/700 and
                    abs(self.exponent/a-self.exponent/b)<=1.):
                args=(a+b)/2+delta/2*_GL_X
                return float(delta/2*self.prefactor*(np.exp(-self.exponent/args)@_GL_W))
            return self.phi(a)-self.phi(b)
        a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
        if np.any(a<0) or np.any(b<0) or not np.all(np.isfinite(a+b)):
            raise ValueError("Nonphysical Phi difference")
        delta=a-b
        if self.exponent==0: return self.prefactor*delta
        out=np.zeros_like(delta)
        unequal=a!=b
        near=(np.abs(delta)<=.2*np.minimum(a,b)) & unequal
        near &= (a>self.exponent/700)&(b>self.exponent/700)
        gentle=np.zeros_like(near)
        gentle[near]=np.abs(self.exponent/a[near]-self.exponent/b[near])<=1.
        near &= gentle
        if np.any(near):
            args=(a[near]+b[near])[:,None]/2+delta[near,None]/2*_GL_X
            out[near]=delta[near]/2*self.prefactor*(np.exp(-self.exponent/args)@_GL_W)
        far=unequal & ~near
        if np.any(far): out[far]=self.phi(a[far])-self.phi(b[far])
        return out


@dataclass
class TimeReference:
    times: np.ndarray
    means: np.ndarray
    surface: np.ndarray
    exchange_mean_C: np.ndarray
    diagnostics: dict


def integrate_reference(model: MoistureFV, drive: LinearDrive, times, initial=2.55,
                        *, method="BDF", rtol=1e-11, atol=1e-13,
                        max_step=.5, timeout=900., progress=None):


    ts=np.asarray(times,float)
    if ts[0]!=0 or np.any(np.diff(ts)<=0) or ts[-1]>121:
        raise ValueError("Diagnostic scope or time ordering invalid")
    n=model.grid.cells
    start=time.perf_counter()
    total_volume=float(model.grid.volumes.sum())
    ratio=model.grid.areas[-1]/total_volume
    y=np.r_[np.full(n,initial),0.]
    yy=np.empty((len(ts),n+1)); yy[0]=y
    knots=np.r_[drive.times[(drive.times>0)&(drive.times<ts[-1])],ts[-1]]
    stats={"method":method,"rtol":rtol,"atol":atol,"max_step_s":max_step,
           "nfev":0,"njev":0,"nlu":0,"accepted_solver_steps":0,
           "spatial_operator":"Original uniform ring FV with independent algebraic Cs",
           "dense_output":"Library polynomial; checked separately against tolerance/method refinement",
           "maximum_surface_residual_m_s":0.,"segments":[]}
    # 指定扩散尺度初始步长，避免细网格下初始猜测越出非负扩散系数范围。
    first_step=min(max_step,.01*float(np.min(np.diff(model.grid.faces)))**2/model.law.D(initial))
    stats["initial_step_policy"]="min(max_step,0.01*dr_min^2/D(C0),segment_length)"
    stats["first_step_s"]=first_step
    def fun(t,z):
        if time.perf_counter()-start>timeout: raise TimeoutError(f"{method} diagnostic timeout")
        div,_,s=model.fluxes_and_jacobian(z[:n],drive(t))
        stats["maximum_surface_residual_m_s"]=max(stats["maximum_surface_residual_m_s"],abs(s.residual_flux))
        return np.r_[-div,ratio*s.outward_flux]
    def jac(t,z):
        _,(lo,di,up),s=model.fluxes_and_jacobian(z[:n],drive(t))
        mat=diags([-lo,-di,-up],[-1,0,1],shape=(n,n),format="csc")
        bottom=csr_matrix(([ratio*s.derivative_flux_last],([0],[n-1])),shape=(1,n))
        return bmat([[mat,None],[bottom,csr_matrix((1,1))]],format="csc")
    now=0.
    for end in knots:
        sol=solve_ivp(fun,(now,float(end)),y,method=method,jac=jac,
                      rtol=rtol,atol=atol,max_step=max_step,dense_output=True,
                      first_step=min(first_step,float(end)-now))
        if not sol.success: raise RuntimeError(sol.message)
        select=(ts>now)&(ts<=end)
        yy[select]=sol.sol(ts[select]).T
        y=sol.y[:,-1]; now=float(end)
        for key in ["nfev","njev","nlu"]: stats[key]+=getattr(sol,key)
        stats["accepted_solver_steps"]+=len(sol.t)-1
        stats["segments"].append({"end_s":now,"steps":len(sol.t)-1,
            "min_dt_s":float(np.min(np.diff(sol.t))),"max_dt_s":float(np.max(np.diff(sol.t)))})
        if progress: progress(now,y[:n],y[n],"segment")
    surface=np.r_[initial,[model.surface(float(c[-1]),drive(float(t))).concentration
                          for t,c in zip(ts[1:],yy[1:,:n])]]
    balance=(yy[:,:n]-initial)@model.grid.volumes+total_volume*yy[:,n]
    stats.update({"wall_seconds":time.perf_counter()-start,
        "minimum_ring_C":float(yy[:,:n].min()),"maximum_ring_C":float(yy[:,:n].max()),
        "maximum_balance_m3_C":float(np.max(abs(balance))),
        "maximum_normalized_balance":float(np.max(abs(balance)))/(total_volume*initial)})
    return TimeReference(ts,yy[:,:n],surface,yy[:,n],stats)


from dataclasses import dataclass
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, bmat, csr_matrix


@dataclass
class SegmentResult:
    times: np.ndarray
    means: np.ndarray
    surface: np.ndarray
    exchange: np.ndarray
    balance: np.ndarray
    steps: np.ndarray
    rejected: list
    next_dt: float
    diagnostics: dict


def be_segment(model, drive, initial_state, times, *, exchange_start=0., initial_dt=.1,
               rtol=1e-11, atol=1e-13, max_dt=.5, min_dt=1e-10,
               timeout=1200., checkpoint=None):


    targets = np.asarray(times, float)
    assert targets.ndim == 1 and len(targets) >= 2 and np.all(np.diff(targets) > 0)
    assert 0 <= targets[0] < targets[-1] <= 1800
    drive(targets)
    c = np.asarray(initial_state, float).copy()
    assert c.shape == (model.grid.cells,) and np.all(np.isfinite(c)) and np.all(c >= 0)
    now, dt, exchange = float(targets[0]), float(initial_dt), float(exchange_start)
    surface0 = float(c[-1]) if now == 0 else model.surface(float(c[-1]), drive(now)).concentration
    if now == 0:
        assert np.all(c == 2.55) and exchange == 0
    volume = model.grid.volumes
    rings, surfaces, exchanges = [c.copy()], [surface0], [exchange]
    balances = [float(volume @ (c-2.55) + exchange)]
    accepted, rejected = [], []
    start = time.perf_counter()
    for target in targets[1:]:
        while now < target:
            if time.perf_counter()-start > timeout:
                if checkpoint: checkpoint(now, c, exchange, dt, "timeout")
                raise TimeoutError(f"BE segment budget exhausted at {now:.17g}")
            endpoint = min(float(target), drive.next_knot(now), now+min(dt, max_dt))
            step = endpoint-now
            if step < min_dt and target-now > 2*min_dt:
                raise ArithmeticError("Adaptive step below minimum")
            mid = now+step/2
            try:
                coarse, sc, _ = model.solve_balance(c, step, drive(endpoint))
                half, sh, dh = model.solve_balance(c, step/2, drive(mid))
                fine, sf, df = model.solve_balance(half, step/2, drive(endpoint))
                scale = atol+rtol*np.maximum(np.abs(coarse), np.abs(fine))
                error = float(np.max(np.abs(fine-coarse)/scale))
                es = abs(sf.concentration-sc.concentration)/(atol+rtol*max(
                    abs(sf.concentration), abs(sc.concentration)))
                error = max(error, es)
                low_half, high_half = min(float(c.min()), drive(mid)), max(float(c.max()), drive(mid))
                low_fine = min(float(half.min()), drive(endpoint))
                high_fine = max(float(half.max()), drive(endpoint))
                if (fine.min() < low_fine-2e-11 or fine.max() > high_fine+2e-11 or
                        half.min() < low_half-2e-11 or half.max() > high_half+2e-11):
                    raise ArithmeticError("Discrete range principle violated")
            except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
                rejected.append({"t": now, "dt": step, "reason": str(exc)})
                dt = step*.25
                if dt < min_dt: raise ArithmeticError("Unrecoverable implicit step") from exc
                continue
            if not np.isfinite(error) or error > 1.:
                rejected.append({"t": now, "dt": step, "error_norm": error,
                                 "reason": "local step-doubling error"})
                dt = step*max(.2, .9/np.sqrt(max(error, 1.)))
                continue
            for before, after, surface, diagnostic, end in (
                    (c, half, sh, dh, mid), (half, fine, sf, df, endpoint)):
                exchanged = model.grid.areas[-1]*surface.outward_flux*step/2
                balance = float(volume @ (after-before)+exchanged)
                exchange += exchanged
                accepted.append([end, step/2, surface.concentration, surface.outward_flux,
                    diagnostic["residual_C"], surface.residual_flux, balance,
                    surface.bracket_low, surface.bracket_high, surface.root_error_C,
                    diagnostic["iterations"]])
            c, now = fine, endpoint
            dt = min(max_dt, step*min(2., max(.5, .9/np.sqrt(max(error, 1e-12)))))
        rings.append(c.copy()); surfaces.append(sf.concentration); exchanges.append(exchange)
        balances.append(float(volume @ (c-2.55)+exchange))
        if checkpoint: checkpoint(now, c, exchange, dt, "output")
    steps = np.asarray(accepted)
    diagnostic = {"wall_seconds": time.perf_counter()-start,
        "accepted_half_steps": len(steps), "rejected_steps": len(rejected),
        "maximum_nonlinear_residual_C": float(np.max(steps[:,4])),
        "maximum_surface_flux_residual_m_s": float(np.max(abs(steps[:,5]))),
        "maximum_step_balance_m3_C": float(np.max(abs(steps[:,6]))),
        "maximum_root_residual_equivalent_C": float(np.max(steps[:,9])),
        "maximum_Newton_iterations": int(np.max(steps[:,10])),
        "maximum_global_balance_m3_C": float(np.max(np.abs(balances))),
        "minimum_ring_C": float(np.min(rings)), "maximum_ring_C": float(np.max(rings))}
    return SegmentResult(targets, np.asarray(rings), np.asarray(surfaces), np.asarray(exchanges),
                         np.asarray(balances), steps, rejected, dt, diagnostic)


def fv_time_reference(model, drive, times, radii, *, method="BDF", rtol=1e-12,
                      atol=1e-14, max_step=.5, timeout_per_segment=180., checkpoint=None):


    ts = np.asarray(times, float)
    assert ts[0] == 0 and ts[-1] <= 1800 and np.all(np.diff(ts)>0)
    n = model.grid.cells
    volume = model.grid.volumes
    total_volume = float(volume.sum())
    ratio = model.grid.areas[-1]/total_volume
    weights = model.grid.point_weights(radii)
    y = np.r_[np.full(n, 2.55), 0.]
    values = np.empty((len(ts), len(radii))); values[0] = 2.55
    exchange = np.zeros(len(ts)); balance = np.zeros(len(ts))
    ring_min = 2.55; ring_max = 2.55
    now = 0.
    segments = []
    first_step = min(max_step, .01*float(np.min(np.diff(model.grid.faces)))**2/model.law.D(2.55))
    max_root_residual = 0.
    segment_clock = [time.perf_counter()]
    def fun(t,z):
        nonlocal max_root_residual
        if time.perf_counter()-segment_clock[0] > timeout_per_segment:
            raise TimeoutError(f"{method} reference segment timed out at {t}")
        div,_,s = model.fluxes_and_jacobian(z[:n],drive(t))
        max_root_residual = max(max_root_residual, abs(s.residual_flux))
        return np.r_[-div,ratio*s.outward_flux]
    def jac(t,z):
        _,(lo,di,up),s = model.fluxes_and_jacobian(z[:n],drive(t))
        mat = diags([-lo,-di,-up],[-1,0,1],shape=(n,n),format="csc")
        bottom = csr_matrix(([ratio*s.derivative_flux_last],([0],[n-1])),shape=(1,n))
        return bmat([[mat,None],[bottom,csr_matrix((1,1))]],format="csc")
    for end in np.r_[drive.times[(drive.times>0)&(drive.times<ts[-1])],ts[-1]]:
        segment_clock[0] = time.perf_counter()
        sol = solve_ivp(fun,(now,float(end)),y,method=method,jac=jac,rtol=rtol,atol=atol,
                        max_step=max_step,dense_output=True,first_step=min(first_step,float(end)-now))
        if not sol.success: raise ArithmeticError(sol.message)
        select = (ts>now)&(ts<=end)
        queried = sol.sol(ts[select]).T
        assert np.all(queried[:,:n] > 0) and np.all(np.isfinite(queried))
        surface = np.asarray([model.surface(float(c[-1]),drive(float(t))).concentration
                       for t,c in zip(ts[select],queried[:,:n])])
        values[select] = queried[:,:n] @ weights.T
        values[select,-1] = surface
        exchange[select] = total_volume*queried[:,n]
        balance[select] = (queried[:,:n]-2.55) @ volume + exchange[select]
        ring_min = min(ring_min,float(queried[:,:n].min()))
        ring_max = max(ring_max,float(queried[:,:n].max()))
        y,now = sol.y[:,-1],float(end)
        segments.append({"end_s":now,"steps":len(sol.t)-1,"nfev":sol.nfev,"njev":sol.njev,
                         "nlu":sol.nlu,"wall_s":time.perf_counter()-segment_clock[0]})
        if checkpoint: checkpoint(now,y[:n],total_volume*y[n],values,exchange,balance,segments)
    stats = {"method":method,"rtol":rtol,"atol":atol,"max_step_s":max_step,
        "initial_step_s":first_step,"initial_step_policy":"0.01*min_dr^2/D(C0)",
        "segments":segments,"maximum_surface_residual_m_s":max_root_residual,
        "maximum_balance_m3_C":float(np.max(abs(balance))),
        "maximum_normalized_balance":float(np.max(abs(balance)))/(total_volume*2.55),
        "minimum_ring_C":ring_min,"maximum_ring_C":ring_max}
    return values,exchange,balance,stats


from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.metadata
import json
import os
import sys
import time
import numpy as np


ROOT = Path(__file__).resolve().parent


def queries(stop=1800.):
    extra = [.001,.01,.1,.25,.5,2.5]
    knots = np.arange(60.,1800.,60.)
    t = np.unique(np.r_[np.arange(int(stop)+1,dtype=float),extra,knots-.1,knots+.1,stop])
    return t[t<=stop]


def write_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",choices=["BE","BDF","Radau"],required=True)
    ap.add_argument("--cells",type=int,required=True)
    ap.add_argument("--rtol",type=float,required=True)
    ap.add_argument("--atol",type=float)
    ap.add_argument("--stop",type=float,default=1800.)
    ap.add_argument("--label",default="")
    ap.add_argument("--fresh",action="store_true")
    ap.add_argument("--max-step",type=float,default=.5)
    ap.add_argument("--nonlinear-tol",type=float,default=2e-13)
    ap.add_argument("--root-tol",type=float,default=5e-15)
    ap.add_argument("--segment-budget",type=float,default=1200.)
    ap.add_argument("--attachments-dir",type=Path,required=True,help="包含附件1.xlsx的题目附件目录")
    ap.add_argument("--output",type=Path,default=ROOT/"q1_runs",help="计算输出目录")
    args = ap.parse_args()
    assert 0<args.stop<=1800
    if args.atol is None: args.atol=float(f"{args.rtol*.01:.15g}")
    environment_path=(args.attachments_dir/"附件1.xlsx").resolve()
    if not environment_path.is_file(): raise FileNotFoundError(f"缺少题目附件：{environment_path}")
    physical={"inputs":{"environment":str(environment_path),"environment_sheet":"Sheet1",
        "environment_sha256":sha256(environment_path)}}
    _,drive,nodes = read_drives(ROOT,physical)
    ts,rr = queries(args.stop),np.arange(21)*.001
    key = f"{args.method}-N{args.cells}-r{args.rtol:g}-t{args.stop:g}{args.label}"
    out = args.output.resolve()/key;out.mkdir(parents=True,exist_ok=False)
    grid = RingGrid(args.cells)
    model = MoistureFV(grid,FastDiffusionLaw(),root_xtol=args.root_tol,nonlinear_tol=args.nonlinear_tol)
    start_clock = time.perf_counter()
    record = {"status":"RUNNING_NOT_ACCEPTED","case":key,"args":vars(args),
        "command":[sys.executable,*sys.argv],"started_utc":datetime.now(timezone.utc).isoformat(),
        "physical_model_changes":[],"workers":1,"random_seed":None,
        "thread_environment":{k:os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]},
        "python":sys.version,"versions":{k:importlib.metadata.version(k) for k in ["numpy","scipy","openpyxl"]},
        "source_sha256":{Path(__file__).name:sha256(Path(__file__))},
        "environment_sha256":sha256(environment_path),"segments":[]}
    write_json(out/"case.json",record)
    try:
        if args.method == "BE":
            can_reuse = False  # 独立从 t=0 求解；不读取任何历史续算状态。
            all_t=[];all_c=[];all_s=[];all_e=[];all_b=[];all_steps=[]
            if can_reuse:
                pm=json.loads((prefix/"case.json").read_text(encoding="utf-8"))
                assert sha256(prefix/"solution.npz")==config["continuation"]["prefix_solution_sha256"]
                for p,h in pm["hashes"].items(): assert sha256(ROOT/p)==h, p
                with np.load(prefix/"solution.npz",allow_pickle=False) as z:
                    initial=z["ring_C"][-1].copy();exchange=float(z["exchange_m3_C"][-1])
                    begin=float(z["time_s"][-1]);dt=2*float(z["accepted_half_steps"][-1,1])
                    steps=np.column_stack([z["accepted_half_steps"],np.full(len(z["accepted_half_steps"]),-1.)])
                    all_t.append(z["time_s"].copy());all_c.append(z["ring_C"].copy())
                    all_s.append(z["surface_C"].copy());all_e.append(z["exchange_m3_C"].copy())
                    all_b.append(z["balance_m3_C"].copy());all_steps.append(steps)
                record["prefix"]={"path":prefix.relative_to(ROOT).as_posix(),
                    "solution_sha256":sha256(prefix/"solution.npz"),"start_s":begin,
                    "initial_proposal_s":dt,"controller_proposal_reset":True,
                    "full_state_used":True,"diagnostics":pm["diagnostics"]}
            else:
                initial=np.full(args.cells,2.55);exchange=0.;begin=0.;dt=.1
                record["prefix"]=None
            ends=[x for x in [121.,300.,600.,900.,1200.,1500.,1800.] if begin<x<args.stop]+[args.stop]
            for end in ends:
                segment_t=np.r_[begin,ts[(ts>begin)&(ts<=end)]]
                last_print=[time.perf_counter()];last_checkpoint=[begin]
                def progress(t,c,q,next_dt,reason):
                    now_clock=time.perf_counter()
                    if t>=last_checkpoint[0]+60 or t==end or reason=="timeout":
                        np.savez_compressed(out/f"checkpoint-{t:.9g}.npz",t_s=t,ring_C=c,
                            exchange_m3_C=q,next_dt_s=next_dt,faces_m=grid.faces,
                            model_sha256=record["source_sha256"][Path(__file__).name],reason=reason)
                        last_checkpoint[0]=t
                    if now_clock-last_print[0]>=20 or t==end or reason=="timeout":
                        print(f"{key} t={t:.9g}/{args.stop:g} wall={now_clock-start_clock:.1f}s",flush=True)
                        last_print[0]=now_clock
                result=be_segment(model,drive,initial,segment_t,exchange_start=exchange,initial_dt=dt,
                    rtol=args.rtol,atol=args.atol,max_dt=args.max_step,timeout=args.segment_budget,checkpoint=progress)
                path=out/f"segment-{begin:g}-{end:g}.npz"
                np.savez_compressed(path,time_s=result.times,ring_C=result.means,surface_C=result.surface,
                    exchange_m3_C=result.exchange,balance_m3_C=result.balance,accepted_half_steps=result.steps,
                    next_dt_s=result.next_dt,faces_m=grid.faces)
                write_json(out/f"rejections-{begin:g}-{end:g}.json",result.rejected)
                record["segments"].append({"start_s":begin,"end_s":end,"file":path.name,
                    "sha256":sha256(path),"diagnostics":result.diagnostics})
                write_json(out/"case.json",record)
                cut=1 if all_t else 0
                all_t.append(result.times[cut:]);all_c.append(result.means[cut:]);all_s.append(result.surface[cut:])
                all_e.append(result.exchange[cut:]);all_b.append(result.balance[cut:]);all_steps.append(result.steps)
                begin,initial,exchange,dt=end,result.means[-1],float(result.exchange[-1]),result.next_dt
            t=np.concatenate(all_t);means=np.concatenate(all_c);surface=np.concatenate(all_s)
            # 前缀输出沿用全时域查询点，便于与完整计算衔接。
            assert np.array_equal(t,ts)
            C=grid.reconstruct(means,rr,surface);C[0]=2.55
            extra={"ring_C":means,"surface_C":surface,"exchange_m3_C":np.concatenate(all_e),
                "balance_m3_C":np.concatenate(all_b),"accepted_half_steps":np.concatenate(all_steps),
                "faces_m":grid.faces,"next_dt_s":dt}
            record["diagnostics"]={"maximum_balance_m3_C":float(np.max(abs(extra["balance_m3_C"]))),
                "normalized_balance":float(np.max(abs(extra["balance_m3_C"])))/(grid.volumes.sum()*2.55),
                "maximum_nonlinear_residual_C":float(np.max(extra["accepted_half_steps"][:,4])),
                "maximum_root_residual_m_s":float(np.max(abs(extra["accepted_half_steps"][:,5]))),
                "accepted_half_steps":len(extra["accepted_half_steps"]),
                "prefix_newton_counts":"not stored per step; -1 unavailable markers only in ledger column 11"}
        else:
            def progress(t,state,q,C,e,b,segments):
                np.savez_compressed(out/f"checkpoint-{t:g}.npz",t_s=t,ring_C=state,
                    exchange_m3_C=q,faces_m=grid.faces)
                print(f"{key} t={t:g}/{args.stop:g} wall={time.perf_counter()-start_clock:.1f}s",flush=True)
            C,e,b,stats=fv_time_reference(model,drive,ts,rr,method=args.method,rtol=args.rtol,
                atol=args.atol,max_step=args.max_step,timeout_per_segment=180.,checkpoint=progress)
            record["diagnostics"]=stats
            extra={"exchange_m3_C":e,"balance_m3_C":b,"faces_m":grid.faces}
        assert np.all(np.isfinite(C)) and np.array_equal(C[0],np.full(21,2.55))
        np.savez_compressed(out/"solution.npz",time_s=ts,radius_m=rr,C=C,**extra)
        record.update(status="CALCULATION_COMPLETE_NUMERICAL_ACCEPTANCE_PENDING",
            solution_sha256=sha256(out/"solution.npz"),wall_seconds=time.perf_counter()-start_clock,
            point_minimum_C=float(C.min()),point_maximum_C=float(C.max()),C_surface_final=float(C[-1,-1]))
        write_json(out/"case.json",record)
        print(json.dumps({k:record[k] for k in ["case","status","wall_seconds","C_surface_final"]}),flush=True)
    except Exception:
        import traceback
        record.update(status="FAILED",traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-start_clock)
        write_json(out/"case.json",record)
        raise


def verify_delivered_result() -> None:

    from openpyxl import load_workbook
    workbook = Path(__file__).resolve().parent / "result1.xlsx"
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

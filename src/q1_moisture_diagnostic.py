"""Short-time diagnostics; preserve the approved spatial FV and BE algorithm.

FastDiffusionLaw evaluates the SAME Phi and 12-point local integrals, avoiding
unused exponentials and potential values. BDF/Radau are semidiscrete time
references ONLY, not an unannounced replacement of backward Euler.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
import numpy as np
from scipy import special
from scipy.integrate import solve_ivp
from scipy.sparse import diags, bmat, csr_matrix

from q1_model import DiffusionLaw, MoistureFV, RingGrid, LinearDrive, _GL_X, _GL_W


class FastDiffusionLaw(DiffusionLaw):
    """Algebraically identical evaluation; verify against the stage-1 kernel."""
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
    """Strict time reference of the ORIGINAL ring-average semidiscretisation.

    Uses the solver's polynomial dense output, validated by tolerance and
    independent method comparisons. Cumulative exchange is an extra ODE state.
    Environment knots are exact segment endpoints. Uniform initial trace is
    stored exactly at t=0, including the incompatible moisture corner.
    """
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
    # The library's exploratory explicit-Euler initial-step estimate can leave
    # the nonnegative D domain on a very fine mesh. Supply a diffusion-scale
    # starting step; accepted BDF/Radau steps remain adaptive and implicit.
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

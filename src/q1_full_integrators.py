"""Full-period continuation of the approved Q1 integrators.

Physics, FV fluxes, surface closure and Newton are imported unchanged.
BE is a one-step method: the entire ring state is required; no history
polynomial is reconstructed. Checkpoints additionally retain next dt and
the exchange ledger for exact continuation of this controller.
"""
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
    """Same accepted BE step and controller as q1_model.integrate_moisture.

    Differences are explicit nonuniform restart state, nonzero start time,
    cumulative ledger and next-step checkpointing. t=0 still has exact C0.
    """
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
    """High-accuracy time reference of the unchanged ring FV, from exact t=0.

    Retains each segment's complete final state; output reconstruction uses
    the same ring moments as the primary method. Not an independent spatial
    validation. Polynomial dense output is validated against Radau/refinement.
    """
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

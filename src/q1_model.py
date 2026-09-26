"""Approved Q1 R1 solver components; no file output and no automatic full run.

C is dry-basis kg/kg. The moisture integral has units m^3*(kg/kg), NOT kg
of water. FV states approximate ring volume averages. Surface states are
independent point values. All production volumes/areas use L=0.25 m.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import time
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import special
from scipy.linalg import solve_banded
from scipy.optimize import brentq


@dataclass(frozen=True)
class Parameters:
    radius_m: float = 0.02
    length_m: float = 0.25
    initial_temperature_C: float = 28.0
    initial_moisture_kg_kg: float = 2.55
    density_kg_m3: float = 820.0
    heat_capacity_J_kgK: float = 2600.0
    conductivity_W_mK: float = 0.36
    heat_transfer_W_m2K: float = 25.0
    moisture_transfer_m_s: float = 8e-7
    diffusivity_prefactor_m2_s: float = 7e-9
    diffusivity_exponent_kg_kg: float = 0.89

    @property
    def alpha(self):
        return self.conductivity_W_mK / (self.density_kg_m3*self.heat_capacity_J_kgK)

    @property
    def bi_heat(self):
        return self.heat_transfer_W_m2K*self.radius_m/self.conductivity_W_mK


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LinearDrive:
    """Original nodes, exact linear segments, no extrapolation."""
    def __init__(self, times, values):
        self.times = np.asarray(times, dtype=float)
        self.values = np.asarray(values, dtype=float)
        if (self.times.ndim != 1 or self.times.size < 2 or
                self.times.shape != self.values.shape or
                not np.all(np.isfinite(self.times)) or
                not np.all(np.isfinite(self.values)) or
                not np.all(np.diff(self.times) > 0)):
            raise ValueError("Invalid or duplicate/missing driver nodes")

    def __call__(self, t):
        tt = np.asarray(t, dtype=float)
        if np.any(tt < self.times[0]) or np.any(tt > self.times[-1]):
            raise ValueError("Extrapolation is prohibited")
        ans = np.interp(tt, self.times, self.values)
        return float(ans) if ans.ndim == 0 else ans

    def next_knot(self, t: float):
        idx = np.searchsorted(self.times, t, side="right")
        return float(self.times[idx]) if idx < len(self.times) else np.inf


def read_drives(root: Path, config: dict):
    import openpyxl
    spec = config["inputs"]
    path = root / spec["environment"]
    if sha256(path) != spec["environment_sha256"]:
        raise ValueError("Original environmental workbook fingerprint changed")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[spec["environment_sheet"]]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    vals = np.array([row[:3] for row in rows], dtype=float)
    wb.close()
    if vals.ndim != 2 or vals.shape[1] != 3 or not np.all(np.isfinite(vals)):
        raise ValueError("Missing or non-numeric environmental node")
    if not np.array_equal(vals[:, 0], np.arange(0., 14400.1, 60.)):
        raise ValueError("Environmental times differ from the audited 60 s sequence")
    q1 = vals[vals[:, 0] <= 1800].copy()
    if len(q1) != 31:
        raise ValueError("Insufficient Q1 environmental coverage")
    return LinearDrive(q1[:, 0], q1[:, 1]), LinearDrive(q1[:, 0], q1[:, 2]), q1


class RadialModes:
    """Bessel solution of constant diffusivity with finite Robin resistance.

u_t=diffusivity/r*(r*u_r)_r; -u_r(R)=Bi/R*(u_s-drive).
Expansion u=drive+sum b_n J0(lambda_n*r/R), radial r-weight projection.
"""
    def __init__(self, radius, diffusivity, bi, modes=64):
        if radius <= 0 or diffusivity <= 0 or bi < 0 or modes < 1:
            raise ValueError("Invalid radial mode parameters")
        self.radius, self.diffusivity, self.bi = radius, diffusivity, bi
        self.modes = int(modes)
        if bi == 0:
            self.roots = self.coefficients = self.beta = np.empty(0)
            return
        z0 = special.jn_zeros(0, modes)
        z1 = special.jn_zeros(1, modes-1) if modes > 1 else np.empty(0)
        left = np.r_[0., z1]
        f = lambda lam: lam*special.j1(lam)-bi*special.j0(lam)
        self.brackets = np.column_stack([left, z0])
        self.roots = np.array([brentq(f, lo, hi, xtol=5e-15,
                                     rtol=4*np.finfo(float).eps)
                               for lo, hi in self.brackets])
        if not np.all(np.diff(self.roots) > 0):
            raise ArithmeticError("Unsorted or repeated Robin root")
        lam = self.roots
        j0, j1 = special.j0(lam), special.j1(lam)
        self.coefficients = 2*j1/(lam*(j0*j0+j1*j1))
        self.beta = diffusivity*(lam/radius)**2
        self.root_residual = lam*j1-bi*j0

    def amplitudes(self, times, drive: LinearDrive, initial):
        ts = np.asarray(times, dtype=float)
        if (ts.ndim != 1 or np.any(ts < 0) or np.any(np.diff(ts) < 0)
                or drive.times[0] != 0):
            raise ValueError("Queries must be ordered and start within a drive from zero")
        drive(ts)  # validates domain
        if self.bi == 0:
            return np.zeros((len(ts), 0))
        b = self.coefficients*(initial-drive(0.))
        now = 0.
        out = []
        for target in ts:
            while now < target:
                endpoint = min(float(target), drive.next_knot(now))
                j = min(np.searchsorted(drive.times, now, side="right")-1,
                        len(drive.times)-2)
                slope = ((drive.values[j+1]-drive.values[j]) /
                         (drive.times[j+1]-drive.times[j]))
                dt = endpoint-now
                decay = np.exp(-self.beta*dt)
                integ = -np.expm1(-self.beta*dt)/self.beta
                b = b*decay-self.coefficients*slope*integ
                now = endpoint
            out.append(b.copy())
        return np.asarray(out)

    def values(self, times, radii, drive: LinearDrive, initial):
        ts, rr = np.asarray(times, float), np.asarray(radii, float)
        if np.any(rr < 0) or np.any(rr > self.radius):
            raise ValueError("Radius outside the cylinder")
        if self.bi == 0:
            return np.full((len(ts), len(rr)), initial, dtype=float)
        b = self.amplitudes(ts, drive, initial)
        basis = special.j0(np.outer(self.roots, rr/self.radius))
        result = drive(ts)[:, None] + b @ basis
        # The initial condition is exact, including an incompatible corner.
        result[ts == 0, :] = initial
        return result

    def volume_mean(self, times, drive, initial):
        if self.bi == 0:
            return np.full(len(times), initial)
        avg_basis = 2*special.j1(self.roots)/self.roots
        ans = drive(times)+self.amplitudes(times, drive, initial)@avg_basis
        ans[np.asarray(times) == 0] = initial
        return ans


_GL_X, _GL_W = leggauss(12)


class DiffusionLaw:
    def __init__(self, prefactor=7e-9, exponent=.89):
        if prefactor <= 0 or exponent < 0:
            raise ValueError("Invalid diffusion law")
        self.prefactor, self.exponent = prefactor, exponent

    def D(self, concentration):
        c = np.asarray(concentration, dtype=float)
        if np.any(c < 0) or not np.all(np.isfinite(c)):
            raise ValueError("Nonphysical concentration passed to D; no clipping")
        ans = np.zeros_like(c)
        if self.exponent == 0:  # constant-D verification configuration only
            ans[...] = self.prefactor
        else:
            positive = c > 0
            ans[positive] = self.prefactor*np.exp(-self.exponent/c[positive])
        return float(ans) if ans.ndim == 0 else ans

    def phi(self, concentration):
        c = np.asarray(concentration, dtype=float)
        if np.any(c < 0) or not np.all(np.isfinite(c)):
            raise ValueError("Nonphysical concentration passed to Phi")
        ans = np.zeros_like(c)
        if self.exponent == 0:
            ans = self.prefactor*c
        else:
            positive = c > 0
            # Phi = d0*C*E2(a/C), exactly equivalent to the integral.
            ans[positive] = (self.prefactor*c[positive] *
                             special.expn(2, self.exponent/c[positive]))
        return float(ans) if ans.ndim == 0 else ans

    def difference(self, left, right):
        a, b = np.broadcast_arrays(np.asarray(left, float), np.asarray(right, float))
        if np.any(a < 0) or np.any(b < 0) or not np.all(np.isfinite(a+b)):
            raise ValueError("Nonphysical Phi difference arguments")
        if self.exponent == 0:
            return self.prefactor*(a-b)
        ans = np.asarray(self.phi(a)-self.phi(b))
        close = (np.abs(a-b) <= .2*np.minimum(a, b)) & (a != b)
        # Use quadrature only when its integrand is also gently varying.
        # At tiny C, relative proximity alone need not imply slowly varying D.
        viable = close & (a > self.exponent/700) & (b > self.exponent/700)
        gentle = np.zeros_like(close)
        gentle[viable] = np.abs(self.exponent/a[viable]-self.exponent/b[viable]) <= 1.
        close = viable & gentle
        if np.any(close):
            mid = (a[close]+b[close])/2
            half = (a[close]-b[close])/2
            args = mid[:, None]+half[:, None]*_GL_X
            ans[close] = half*(self.D(args)@_GL_W)
        return float(ans) if ans.ndim == 0 else ans


class RingGrid:
    def __init__(self, cells, radius=.02, length=.25):
        if cells < 3 or radius <= 0 or length <= 0:
            raise ValueError("Invalid ring geometry")
        self.cells, self.radius, self.length = int(cells), radius, length
        self.faces = np.linspace(0., radius, cells+1)
        self.centres = (self.faces[:-1]+self.faces[1:])/2
        self.volumes = np.pi*length*np.diff(self.faces**2)
        self.areas = 2*np.pi*length*self.faces
        self.tau = self.areas[1:-1]/np.diff(self.centres)
        self.delta = radius-self.centres[-1]

    def mean_of(self, fn, order=16):
        x, w = leggauss(order)
        lo, hi = self.faces[:-1]**2, self.faces[1:]**2
        radii = np.sqrt((lo+hi)[:, None]/2+(hi-lo)[:, None]/2*x)
        return fn(radii)@w/2

    def point_weights(self, radii):
        """Quadratic-in-r^2 local fit to 3 exact ring-mean moments.

        r=0 instead uses the approved 2-ring even reconstruction. The surface
        row is reserved for the independent Cs. No artificial clipping.
        """
        rr = np.asarray(radii, float)
        if np.any(rr < 0) or np.any(rr > self.radius):
            raise ValueError("Query outside radius")
        out = np.zeros((len(rr), self.cells))
        edges = self.faces**2
        for j, r in enumerate(rr):
            if r == self.radius:
                continue
            if r == 0:
                out[j, :2] = [1.25, -.25]
                continue
            idx = min(np.searchsorted(self.faces, r, side="right")-1,
                      self.cells-1)
            start = max(0, min(idx-1, self.cells-3))
            ids = np.arange(start, start+3)
            origin = (edges[idx]+edges[idx+1])/2
            scale = edges[start+3]-edges[start]
            lo, hi = (edges[ids]-origin)/scale, (edges[ids+1]-origin)/scale
            moments = np.column_stack([np.ones(3), (lo+hi)/2,
                                       (lo*lo+lo*hi+hi*hi)/3])
            s = (r*r-origin)/scale
            out[j, ids] = np.linalg.solve(moments.T, [1., s, s*s])
        return out

    def reconstruct(self, averages, radii, surface):
        ans = np.asarray(averages) @ self.point_weights(radii).T
        ans[..., np.asarray(radii) == self.radius] = np.asarray(surface)[..., None]
        return ans


@dataclass
class SurfaceState:
    """root_error_C denotes residual/derivative, a local error indicator only."""
    concentration: float
    outward_flux: float
    derivative_flux_last: float
    residual_flux: float
    root_error_C: float
    bracket_low: float
    bracket_high: float
    iterations: int


class MoistureFV:
    def __init__(self, grid: RingGrid, law: DiffusionLaw, transfer=8e-7,
                 root_xtol=5e-15, nonlinear_tol=2e-13):
        self.grid, self.law, self.transfer = grid, law, float(transfer)
        self.root_xtol, self.nonlinear_tol = root_xtol, nonlinear_tol
        if transfer < 0:
            raise ValueError("Negative transfer coefficient")

    def surface(self, last, external):
        if last < 0 or external < 0:
            raise ValueError("Negative state at surface")
        lo, hi = min(last, external), max(last, external)
        h, delta, law = self.transfer, self.grid.delta, self.law
        if h == 0 or last == external:
            return SurfaceState(last, 0., 0. if h == 0 else
                                h*law.D(last)/(law.D(last)+h*delta),
                                0., 0., lo, hi, 0)
        scale = law.prefactor
        def g(c):
            return law.difference(c, last)/scale + h*delta/scale*(c-external)
        cs, info = brentq(g, lo, hi, xtol=self.root_xtol,
                         rtol=4*np.finfo(float).eps, full_output=True)
        if not info.converged or not lo <= cs <= hi:
            raise ArithmeticError("Surface bracket solve failed")
        dcs = law.D(cs)
        flux = h*(cs-external)
        residual = law.difference(cs, last)/delta+flux
        deriv = h*law.D(last)/(dcs+h*delta)
        c_error = abs(residual)/(dcs/delta+h)
        if c_error > 10*self.root_xtol + 2e-14*abs(cs):
            raise ArithmeticError("Surface residual exceeds concentration budget")
        return SurfaceState(cs, flux, deriv, residual, c_error, lo, hi,
                            int(info.iterations))

    def fluxes_and_jacobian(self, state, external):
        """Returns divergence=(Fout-Fin)/V, its tridiagonal Jacobian."""
        grid, law = self.grid, self.law
        c = np.asarray(state)
        internal = grid.tau*law.difference(c[:-1], c[1:])
        surf = self.surface(float(c[-1]), float(external))
        flux = np.r_[0., internal, grid.areas[-1]*surf.outward_flux]
        divergence = np.diff(flux)/grid.volumes
        d = law.D(c)
        diagonal = np.zeros_like(c)
        diagonal[:-1] += grid.tau*d[:-1]
        diagonal[1:] += grid.tau*d[1:]
        diagonal[-1] += grid.areas[-1]*surf.derivative_flux_last
        diagonal /= grid.volumes
        lower = -grid.tau*d[:-1]/grid.volumes[1:]
        upper = -grid.tau*d[1:]/grid.volumes[:-1]
        return divergence, (lower, diagonal, upper), surf

    def solve_balance(self, old, dt, external, source=None, steady=False):
        """Damped Newton in concentration units; no after-the-fact clipping.

        Steady/source options are exclusively for manufactured verification.
        """
        old = np.asarray(old, float)
        c = old.copy()
        src = np.zeros_like(c) if source is None else np.asarray(source, float)
        weight = 0. if steady else 1.
        factor = 1. if steady else dt
        evaluations = 0
        for iteration in range(25):
            div, (lower, diagonal, upper), surf = self.fluxes_and_jacobian(c, external)
            evaluations += 1
            residual = weight*(c-old)+factor*(div-src)
            norm = float(np.max(np.abs(residual)))
            if norm <= self.nonlinear_tol:
                return c, surf, {"iterations": iteration, "residual_C": norm,
                                  "flux_evaluations": evaluations}
            ab = np.zeros((3, len(c)))
            ab[0, 1:] = factor*upper
            ab[1] = weight+factor*diagonal
            ab[2, :-1] = factor*lower
            change = solve_banded((1, 1), ab, -residual, check_finite=False)
            damping = 1.
            # Only compute a positivity bound when a full step can reach zero.
            # Dividing by harmless subnormal negative corrections can overflow.
            negative = change < -.99*c
            if np.any(negative):
                damping = min(1., float(np.min(-.99*c[negative]/change[negative])))
            for _ in range(22):
                candidate = c+damping*change
                if np.any(candidate < 0) or not np.all(np.isfinite(candidate)):
                    damping *= .5
                    continue
                new_div, _, _ = self.fluxes_and_jacobian(candidate, external)
                evaluations += 1
                new_res = weight*(candidate-old)+factor*(new_div-src)
                if np.max(np.abs(new_res)) <= (1.-1e-4*damping)*norm:
                    c = candidate
                    break
                damping *= .5
            else:
                raise ArithmeticError(f"Newton line search failed, residual={norm:.3e}")
        raise ArithmeticError(f"Newton iteration limit, residual={norm:.3e}")


@dataclass
class MoistureResult:
    times: np.ndarray
    ring_means: np.ndarray
    surface: np.ndarray
    cumulative_exchange: np.ndarray
    balance_residual: np.ndarray
    accepted_half_steps: list = field(default_factory=list)
    rejected_steps: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def integrate_moisture(model: MoistureFV, drive: LinearDrive, initial, query_times,
                       *, atol=1e-9, rtol=1e-7, initial_dt=.1, max_dt=1.,
                       min_dt=1e-10, timeout=300., checkpoint: Callable | None=None):
    """Adaptive BE, accepting two half steps and their actual exchange fluxes.

    max norm includes ring concentrations AND the surface root. The error
    estimate is local, first order; this is not a global accuracy certificate.
    Queries and all environmental knots are exact step endpoints.
    """
    targets = np.asarray(query_times, float)
    if (targets.ndim != 1 or targets[0] != 0 or np.any(np.diff(targets) <= 0)
            or max_dt <= 0 or initial_dt <= 0 or atol <= 0 or rtol < 0):
        raise ValueError("Invalid integration settings or output times")
    drive(targets)
    c = np.broadcast_to(initial, (model.grid.cells,)).astype(float).copy()
    if np.any(c < 0) or not np.all(np.isfinite(c)):
        raise ValueError("Invalid initial concentration")
    if not np.all(c == c[0]):
        raise ValueError("This integrator requires the approved uniform initial field; "
                         "nonuniform verification needs an explicit initial surface trace")
    state0 = c.copy()
    initial_integral = float(model.grid.volumes@c)
    exchange = 0.
    now, dt = 0., initial_dt
    rings, surfaces, exchanges, balances = [c.copy()], [float(c[-1])], [0.], [0.]
    accepted, rejected = [], []
    start_clock = time.perf_counter()
    max_nl = max_root = max_root_C = 0.
    max_iters = 0
    max_step_balance = 0.
    for target in targets[1:]:
        while now < target:
            if time.perf_counter()-start_clock > timeout:
                if checkpoint:
                    checkpoint(now, c.copy(), exchange, "timeout")
                raise TimeoutError(f"Short-run time budget exhausted at t={now:.9g}")
            endpoint = min(float(target), drive.next_knot(now), now+min(dt, max_dt))
            step = endpoint-now
            if step < min_dt and target-now > 2*min_dt:
                raise ArithmeticError("Adaptive step below minimum")
            mid = now+step/2
            try:
                coarse, sc, dc = model.solve_balance(c, step, drive(endpoint))
                half, sh, dh = model.solve_balance(c, step/2, drive(mid))
                fine, sf, df = model.solve_balance(half, step/2, drive(endpoint))
                scale = atol+rtol*np.maximum(np.abs(coarse), np.abs(fine))
                error = float(np.max(np.abs(fine-coarse)/scale))
                es = abs(sf.concentration-sc.concentration)/(atol+rtol*max(
                    abs(sf.concentration), abs(sc.concentration)))
                error = max(error, es)
                low_half = min(float(c.min()), drive(mid))
                high_half = max(float(c.max()), drive(mid))
                low_fine = min(float(half.min()), drive(endpoint))
                high_fine = max(float(half.max()), drive(endpoint))
                if (fine.min() < low_fine-2e-11 or fine.max() > high_fine+2e-11 or
                        half.min() < low_half-2e-11 or half.max() > high_half+2e-11):
                    raise ArithmeticError("Discrete range principle violated")
            except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
                rejected.append({"t": now, "dt": step, "reason": str(exc)})
                dt = step*.25
                if dt < min_dt:
                    raise ArithmeticError("Unrecoverable implicit step") from exc
                continue
            if not np.isfinite(error) or error > 1.:
                rejected.append({"t": now, "dt": step, "error_norm": error,
                                 "reason": "local step-doubling error"})
                dt = step*max(.2, .9/np.sqrt(max(error, 1.)))
                continue
            for state_before, state_after, surface_state, diagnostic, end in (
                    (c, half, sh, dh, mid), (half, fine, sf, df, endpoint)):
                exchanged = model.grid.areas[-1]*surface_state.outward_flux*step/2
                step_balance = float(model.grid.volumes@(state_after-state_before)+exchanged)
                exchange += exchanged
                max_step_balance = max(max_step_balance, abs(step_balance))
                max_nl = max(max_nl, diagnostic["residual_C"])
                max_iters = max(max_iters, diagnostic["iterations"])
                max_root = max(max_root, abs(surface_state.residual_flux))
                max_root_C = max(max_root_C, surface_state.root_error_C)
                accepted.append([end, step/2, surface_state.concentration,
                                 surface_state.outward_flux, diagnostic["residual_C"],
                                 surface_state.residual_flux, step_balance,
                                 surface_state.bracket_low, surface_state.bracket_high,
                                 surface_state.root_error_C])
            c, now = fine, endpoint
            dt = min(max_dt, step*min(2., max(.5, .9/np.sqrt(max(error, 1e-12)))))
        rings.append(c.copy())
        surfaces.append(sf.concentration)
        exchanges.append(exchange)
        balances.append(float(model.grid.volumes@(c-state0)+exchange))
        if checkpoint:
            checkpoint(now, c.copy(), exchange, "output")
    return MoistureResult(targets, np.array(rings), np.array(surfaces),
                          np.array(exchanges), np.array(balances), accepted, rejected,
                          {"wall_seconds": time.perf_counter()-start_clock,
                           "accepted_full_steps": len(accepted)//2,
                           "accepted_half_steps": len(accepted),
                           "rejected_steps": len(rejected),
                           "maximum_nonlinear_residual_C": max_nl,
                           "maximum_surface_flux_residual_m_s": max_root,
                           "maximum_surface_residual_equivalent_C": max_root_C,
                           "maximum_newton_iterations": max_iters,
                           "maximum_step_balance_m3_C": max_step_balance,
                           "initial_integral_m3_C": initial_integral,
                           "maximum_cumulative_balance_m3_C": float(np.max(np.abs(balances))),
                           "maximum_normalized_balance": (float(np.max(np.abs(balances)))/initial_integral
                                                          if initial_integral > 0 else None),
                           "range_ring_means": [float(np.min(rings)), float(np.max(rings))]})

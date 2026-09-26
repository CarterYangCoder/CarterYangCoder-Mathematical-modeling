"""Q2-R short-diagnostic implementation; no Q1 files are modified.

Unknowns are annular volume averages, interleaved T (degC), C (kg/kg dry).
The half-ring boundary states are separate algebraic unknowns.  Geometry and
moment-based point reconstruction reuse the frozen Q1 pure helper only.
No latent heat, Soret flux, moving domain, rho*C storage or extrapolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
from scipy import special
from scipy.optimize import brentq
from scipy.sparse import coo_matrix, bmat, csr_matrix
from scipy.integrate import solve_ivp

from q1_model import RingGrid, LinearDrive

GLX, GLW = np.polynomial.legendre.leggauss(12)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def harmonic(x, y):
    return 2*x*y/(x+y)


def harmonic_partials(x, y):
    return 2*y*y/(x+y)**2, 2*x*x/(x+y)**2


@dataclass(frozen=True)
class Properties:
    """Appendix 3 by default. q1 is an explicitly isolated verification case."""
    test_q1: bool = False

    @property
    def dstar(self):
        return 7e-9 if self.test_q1 else 2.4e-3

    @property
    def gamma(self):
        return .89 if self.test_q1 else .45

    def thermal(self, c):
        c = np.asarray(c, float)
        if self.test_q1:
            return (np.full_like(c, 820*2600.), np.full_like(c, .36),
                    np.zeros_like(c), np.zeros_like(c), np.zeros_like(c))
        rho, cp = 650+128*c, 1450+2736*c/(1+c)
        cpp = 2736/(1+c)**2
        return (rho*cp, .21+.38*c/(1+c), 128*cp+rho*cpp,
                .38/(1+c)**2, 256*cpp-2*rho*2736/(1+c)**3)

    def a(self, t):
        tk = np.asarray(t, float)+273.15
        if np.any(tk <= 0) or not np.all(np.isfinite(tk)):
            raise ValueError("Nonphysical absolute temperature; no clipping")
        if self.test_q1:
            return np.ones_like(tk), np.zeros_like(tk)
        a = np.exp(-3850/tk)
        return a, a*3850/tk**2

    def b(self, c):
        c = np.asarray(c, float)
        if np.any(c < 0) or not np.all(np.isfinite(c)):
            raise ValueError("Nonphysical C; no clipping")
        ans = np.zeros_like(c)
        positive = c > 0
        ans[positive] = np.exp(-self.gamma/c[positive])
        return ans

    def psi(self, c):
        c = np.asarray(c, float)
        if np.any(c < 0) or not np.all(np.isfinite(c)):
            raise ValueError("Nonphysical potential input")
        ans = np.zeros_like(c)
        positive = c > 0
        ans[positive] = c[positive]*special.expn(2, self.gamma/c[positive])
        return ans

    def psi_difference(self, left, right):
        """Integral from right to left; near cancellation uses 12-point GL.

        This is Psi(C_left)-Psi(C_right), NOT Phi(C_left,T_left)-Phi(...).
        """
        left, right = np.broadcast_arrays(np.asarray(left, float), np.asarray(right, float))
        if np.any(left < 0) or np.any(right < 0) or not np.all(np.isfinite(left+right)):
            raise ValueError("Nonphysical potential difference input")
        delta = left-right
        out = np.zeros_like(delta)
        unequal = delta != 0
        near = unequal & (abs(delta) <= .2*np.minimum(left, right))
        near &= (left > self.gamma/700) & (right > self.gamma/700)
        gentle = np.zeros_like(near)
        gentle[near] = abs(self.gamma/left[near]-self.gamma/right[near]) <= 1
        near &= gentle
        if np.any(near):
            points = (left[near]+right[near])[:, None]/2+delta[near, None]*GLX/2
            out[near] = delta[near]/2*(np.exp(-self.gamma/points)@GLW)
        far = unequal & ~near
        if np.any(far):
            out[far] = self.psi(left[far])-self.psi(right[far])
        return out

    def D(self, c, t):
        return self.dstar*self.a(t)[0]*self.b(c)


@dataclass
class Surface:
    T: float
    C: float
    flux_T: float
    flux_C: float
    derivatives: np.ndarray  # rows Ts,Cs; columns T_last,C_last
    heat_residual_W_m2: float
    moisture_residual_m_s: float
    root_equivalent_C: float
    bracket: tuple
    root_iterations: int


class CoupledFV:
    def __init__(self, cells, *, flux="potential", hT=25., hC=8e-7,
                 properties=None, root_xtol=5e-15):
        if flux not in ("potential", "fick"):
            raise ValueError("Unknown internal moisture flux")
        self.grid = RingGrid(cells)
        self.props = Properties() if properties is None else properties
        self.flux, self.hT, self.hC = flux, float(hT), float(hC)
        self.root_xtol = root_xtol
        self.boundary_calls = 0
        self.max_root_equiv = self.max_heat_residual = self.max_moisture_residual = 0.
        self.max_root_iterations = 0

    def surface(self, ti, ci, ta, ce):
        """Shared, consistent half-ring Robin closure for the two flux schemes.

        A common accurate surface treatment isolates the interior-flux comparison.
        'fick' is therefore a direct-Fick INTERNAL-flux baseline, not an
        independent boundary implementation. The node reference is independent.
        """
        if ci <= 0 or ce <= 0 or not all(map(math.isfinite, (ti, ci, ta, ce))):
            raise ValueError("Invalid boundary state; no clipping")
        p, delta = self.props, self.grid.delta
        _, ki, _, kip, _ = p.thermal(ci)
        ai, aip = p.a(ti)
        ds = p.dstar
        ht, hc = self.hT, self.hC
        # A local bracket-wide sufficient condition for a unique crossing.
        # This guards actual nonlinear trial states, not just a predeclared box.
        cmin = min(ci, ce)
        _, kmin, _, kpmax, _ = p.thermal(cmin)
        kbmin = harmonic(ki, kmin)
        kbprime_bound = 2*ki**2/(ki+kmin)**2*kpmax
        tsprime_bound = delta*ht*abs(ti-ta)*kbprime_bound/(kbmin+delta*ht)**2
        tkmin = min(ti, ta)+273.15
        if tkmin <= 0:
            raise ValueError("Nonphysical boundary temperature")
        eta = (0. if p.test_q1 else 3850/tkmin**2)*tsprime_bound*abs(ci-ce)
        if eta >= 1:
            raise ArithmeticError(f"Surface branch uniqueness guard failed: eta={eta:.17g}")

        def calc(cs, derivatives=False):
            _, ks, _, ksp, _ = p.thermal(cs)
            kb = harmonic(ki, ks)
            ts = (kb*ti+delta*ht*ta)/(kb+delta*ht)
            av, ap = p.a(ts)
            ab = harmonic(ai, av)
            dp = float(p.psi_difference(cs, ci))
            g = ds*ab*dp/delta+hc*(cs-ce)
            if not derivatives:
                return float(g)
            hki, hks = harmonic_partials(ki, ks)
            hai, has = harmonic_partials(ai, av)
            kbci, kbcs = hki*kip, hks*ksp
            abi, abs_ = hai*aip, has*ap
            # Independent 2x2 implicit differentiation of both Robin equations.
            j00 = kb/delta+ht
            j01 = kbcs*(ts-ti)/delta
            j10 = ds*abs_*dp/delta
            j11 = ds*ab*float(p.b(cs))/delta+hc
            x00 = -kb/delta
            x01 = kbci*(ts-ti)/delta
            x10 = ds*abi*dp/delta
            x11 = -ds*ab*float(p.b(ci))/delta
            determinant = j00*j11-j01*j10
            if not math.isfinite(determinant) or determinant <= 0:
                raise ArithmeticError("Surface implicit Jacobian lost the verified positive branch")
            dx = -np.array([[j11*x00-j01*x10, j11*x01-j01*x11],
                            [-j10*x00+j00*x10, -j10*x01+j00*x11]])/determinant
            root_derivative = j11-j10*j01/j00
            hr = kb*(ts-ti)/delta+ht*(ts-ta)
            return float(ts), dx, float(hr), float(g), float(root_derivative)

        if hc == 0 or ci == ce:
            cs, iterations = ci, 0
        else:
            lo, hi = sorted((ci, ce))
            # Scale m/s residual to a C-sized function; exact roots unchanged.
            scale = hc+float(p.D(ci, ti))/delta
            cs, info = brentq(lambda c: calc(c)/scale, lo, hi,
                             xtol=self.root_xtol, rtol=8.881784197001252e-16,
                             full_output=True, maxiter=80)
            if not info.converged:
                raise ArithmeticError("Bracketed surface solve failed")
            iterations = int(info.iterations)
        ts, dx, hr, cr, gp = calc(cs, True)
        equiv = abs(cr/gp)
        if equiv > 10*self.root_xtol+3e-14*abs(cs):
            raise ArithmeticError("Surface residual exceeds concentration-scaled tolerance")
        self.boundary_calls += 1
        self.max_root_equiv = max(self.max_root_equiv, equiv)
        self.max_heat_residual = max(self.max_heat_residual, abs(hr))
        self.max_moisture_residual = max(self.max_moisture_residual, abs(cr))
        self.max_root_iterations = max(self.max_root_iterations, iterations)
        return Surface(ts, cs, ht*(ts-ta), hc*(cs-ce), dx, hr, cr, equiv,
                       tuple(sorted((ci, ce))), iterations)

    def interior(self, t, c, *, jacobian=False):
        p, tau = self.props, self.grid.tau
        w, k, wp, kp, wpp = p.thermal(c)
        a, ap = p.a(t)
        b = p.b(c)
        kh = harmonic(k[:-1], k[1:])
        ft = tau*kh*(t[:-1]-t[1:])
        if self.flux == "potential":
            ah = harmonic(a[:-1], a[1:])
            diff = p.psi_difference(c[:-1], c[1:])
            fc = tau*p.dstar*ah*diff
        else:
            d = p.dstar*a*b
            dh = harmonic(d[:-1], d[1:])
            diff = c[:-1]-c[1:]
            fc = tau*dh*diff
        if not jacobian:
            return ft, fc
        n = len(c)
        deriv = np.zeros((n-1, 2, 2, 2))  # face, field, left/right, state T/C
        hx, hy = harmonic_partials(k[:-1], k[1:])
        deriv[:, 0, 0, 0] = tau*kh
        deriv[:, 0, 1, 0] = -tau*kh
        deriv[:, 0, 0, 1] = tau*hx*kp[:-1]*(t[:-1]-t[1:])
        deriv[:, 0, 1, 1] = tau*hy*kp[1:]*(t[:-1]-t[1:])
        if self.flux == "potential":
            hx, hy = harmonic_partials(a[:-1], a[1:])
            deriv[:, 1, 0, 0] = tau*p.dstar*hx*ap[:-1]*diff
            deriv[:, 1, 1, 0] = tau*p.dstar*hy*ap[1:]*diff
            deriv[:, 1, 0, 1] = tau*p.dstar*ah*b[:-1]
            deriv[:, 1, 1, 1] = -tau*p.dstar*ah*b[1:]
        else:
            hx, hy = harmonic_partials(d[:-1], d[1:])
            dc = d*p.gamma/c**2
            dt = p.dstar*ap*b
            deriv[:, 1, 0, 0] = tau*hx*dt[:-1]*diff
            deriv[:, 1, 1, 0] = tau*hy*dt[1:]*diff
            deriv[:, 1, 0, 1] = tau*(hx*dc[:-1]*diff+dh)
            deriv[:, 1, 1, 1] = tau*(hy*dc[1:]*diff-dh)
        return ft, fc, deriv

    def evaluate(self, state, ta, ce, *, jacobian=False, source=None):
        n, p = self.grid.cells, self.props
        yy = np.asarray(state[:2*n]).reshape(n, 2)
        t, c = yy[:, 0], yy[:, 1]
        if not np.all(np.isfinite(yy)) or np.any(c <= 0):
            raise ValueError("Nonphysical trial state; no clipping or reuse of old result")
        w, _, wp, _, wpp = p.thermal(c)
        v, area = self.grid.volumes, self.grid.areas[-1]
        sf = self.surface(float(t[-1]), float(c[-1]), ta, ce)
        inner = self.interior(t, c, jacobian=jacobian)
        ft, fc = np.r_[0., inner[0], area*sf.flux_T], np.r_[0., inner[1], area*sf.flux_C]
        rhs = np.column_stack((-np.diff(ft)/(v*w), -np.diff(fc)/v))
        # Manufactured tests alone supply volume-averaged heat W/m3 and C/s sources.
        if source is not None:
            rhs[:, 0] += source[:, 0]/w
            rhs[:, 1] += source[:, 1]
        if not jacobian:
            return rhs.ravel(), sf
        rows, cols, vals = [], [], []
        faces = np.arange(n-1)
        d = inner[2]
        for field in range(2):
            storage = v*w if field == 0 else v
            for receiver, sign in ((0, -1), (1, 1)):
                for neighbor in range(2):
                    for variable in range(2):
                        rows.append(2*(faces+receiver)+field)
                        cols.append(2*(faces+neighbor)+variable)
                        vals.append(sign*d[:, field, neighbor, variable]/storage[faces+receiver])
        for field, h in ((0, self.hT), (1, self.hC)):
            storage = v[-1]*(w[-1] if field == 0 else 1)
            for variable in range(2):
                rows.append(np.array([2*(n-1)+field]))
                cols.append(np.array([2*(n-1)+variable]))
                vals.append(np.array([-area*h*sf.derivatives[field, variable]/storage]))
        rows.append(2*np.arange(n)); cols.append(2*np.arange(n)+1)
        vals.append(-rhs[:, 0]*wp/w)
        mat = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(2*n, 2*n)).tocsc()
        return rhs.ravel(), sf, mat

    def augmented(self, state, ta, ce, *, jacobian=False, source=None):
        """Three passive integrals: C exchange, heat input, storage-chain term.

        Scales are mean C and mean initial-capacity-equivalent kelvins. These
        auxiliary states do not feed back into the physical solution.
        """
        n, p, v = self.grid.cells, self.props, self.grid.volumes
        total = float(v.sum())
        w0 = float(p.thermal(2.55)[0])
        scale = total*w0
        result = self.evaluate(state, ta, ce, jacobian=jacobian, source=source)
        dy, sf = result[:2]
        t, c, dc = state[:2*n:2], state[1:2*n:2], dy[1::2]
        _, _, wp, _, wpp = p.thermal(c)
        coeff = v*wp*(t-28)/scale
        extras = np.array([self.grid.areas[-1]*sf.flux_C/total,
                           -self.grid.areas[-1]*sf.flux_T/scale, coeff@dc])
        if not jacobian:
            return np.r_[dy, extras]
        mat = result[2]
        bottom = np.zeros((3, 2*n))
        bottom[0, -2:] = self.grid.areas[-1]*self.hC*sf.derivatives[1]/total
        bottom[1, -2:] = -self.grid.areas[-1]*self.hT*sf.derivatives[0]/scale
        bottom[2] = np.asarray(coeff@mat[1::2]).ravel()
        bottom[2, 0::2] += v*wp*dc/scale
        bottom[2, 1::2] += v*wpp*(t-28)*dc/scale
        return bmat([[mat, None], [csr_matrix(bottom), csr_matrix((3, 3))]], format="csc")


def read_environment(root, config):
    import openpyxl
    spec = config["inputs"]["environment"]
    path = Path(root)/spec["path"]
    if sha256(path) != spec["sha256"]:
        raise ValueError("Environmental source fingerprint changed")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = np.array([r[:3] for r in wb[spec["sheet"]].iter_rows(min_row=2, values_only=True)], float)
    wb.close()
    if rows.shape != (241, 3) or not np.all(np.isfinite(rows)):
        raise ValueError("Missing/non-numeric input; stop rather than fill")
    if not np.array_equal(rows[:, 0], np.arange(241)*60):
        raise ValueError("Original sampling differs from the audited 60 s nodes")
    return LinearDrive(rows[:, 0], rows[:, 1]), LinearDrive(rows[:, 0], rows[:, 2]), rows


def run_short(model, drives, times, radii, *, method="Radau", rtol=1e-9,
              atol_T=1e-10, atol_C=1e-12, max_step=1., first_step_factor=.01,
              checkpoint_dir=None, timeout_per_segment=600., verbose=True):
    times, radii = np.asarray(times, float), np.asarray(radii, float)
    if times[0] != 0 or times[-1] > 121 or np.any(np.diff(times) <= 0):
        raise ValueError("This entry is authorized only for 0--121 s diagnostics")
    td, cd = drives
    n, grid, p = model.grid.cells, model.grid, model.props
    y = np.r_[np.tile([28., 2.55], n), np.zeros(3)]
    weights = grid.point_weights(radii)
    T = np.full((len(times), len(radii)), np.nan); T[0] = 28.
    C = np.full_like(T, np.nan); C[0] = 2.55
    exchange = np.full((len(times), 3), np.nan); exchange[0] = 0
    balance = np.full((len(times), 2), np.nan); balance[0] = 0
    atols = np.r_[np.tile([atol_T, atol_C], n), atol_C, atol_T, atol_T]
    w0 = float(p.thermal(2.55)[0]); volume = float(grid.volumes.sum())
    alpha = float(p.thermal(2.55)[1])/w0
    first = min(max_step, first_step_factor*np.min(np.diff(grid.faces))**2/alpha)
    now, segments = 0., []
    accepted_range = [28., 28., 2.55, 2.55]
    clock = time.perf_counter()
    for end in np.r_[td.times[(td.times > 0)&(td.times < times[-1])], times[-1]]:
        started = time.perf_counter()
        def fun(t, z):
            if time.perf_counter()-started > timeout_per_segment:
                raise TimeoutError(f"Q2 segment deadline at t={t:.17g}")
            return model.augmented(z, td(t), cd(t))
        def jac(t, z):
            return model.augmented(z, td(t), cd(t), jacobian=True)
        sol = solve_ivp(fun, (now, float(end)), y, method=method, jac=jac,
                        rtol=rtol, atol=atols, max_step=max_step,
                        first_step=min(first, float(end)-now), dense_output=True)
        if not sol.success:
            raise ArithmeticError(sol.message)
        select = (times > now)&(times <= end)
        state = sol.sol(times[select]).T
        if not np.all(np.isfinite(state)) or np.any(state[:, 1:2*n:2] <= 0):
            raise ArithmeticError("Nonpositive or nonfinite accepted/dense-output state")
        surfaces = [model.surface(z[-5], z[-4], td(t), cd(t))
                    for t, z in zip(times[select], state)]
        T[select] = state[:, :2*n:2]@weights.T
        C[select] = state[:, 1:2*n:2]@weights.T
        surf_mask = radii == grid.radius
        T[np.ix_(select, surf_mask)] = np.array([s.T for s in surfaces])[:, None]
        C[np.ix_(select, surf_mask)] = np.array([s.C for s in surfaces])[:, None]
        exchange[select] = state[:, -3:]
        balance[select, 0] = (state[:, 1:2*n:2]-2.55)@grid.volumes+volume*state[:, -3]
        w = p.thermal(state[:, 1:2*n:2])[0]
        balance[select, 1] = (w*(state[:, :2*n:2]-28))@grid.volumes-volume*w0*(state[:, -2]+state[:, -1])
        accepted_range[0] = min(accepted_range[0], float(sol.y[:2*n:2].min()))
        accepted_range[1] = max(accepted_range[1], float(sol.y[:2*n:2].max()))
        accepted_range[2] = min(accepted_range[2], float(sol.y[1:2*n:2].min()))
        accepted_range[3] = max(accepted_range[3], float(sol.y[1:2*n:2].max()))
        y, now = sol.y[:, -1], float(end)
        info = dict(end_s=now, accepted_steps=len(sol.t)-1, nfev=sol.nfev,
                    njev=sol.njev, nlu=sol.nlu, wall_s=time.perf_counter()-started,
                    rejected_steps="not exposed by solve_ivp; not reported as zero")
        segments.append(info)
        if checkpoint_dir:
            out = Path(checkpoint_dir); out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out/f"checkpoint-{now:g}.npz", time_s=now,
                                state=y, faces_m=grid.faces, config_method=method,
                                code_sha256=sha256(__file__))
        if verbose:
            print(json.dumps({"Q2_progress": info, "cells": n, "flux": model.flux,
                              "method": method}, ensure_ascii=False), flush=True)
    if np.any(~np.isfinite(T)) or np.any(~np.isfinite(C)):
        raise ArithmeticError("Missing diagnostic output")
    stats = dict(cells=n, flux=model.flux, method=method, rtol=rtol, atol_T=atol_T,
                 atol_C=atol_C, max_step_s=max_step, first_step_s=first,
                 first_step_factor=first_step_factor, segments=segments,
                 wall_s=time.perf_counter()-clock, full_final_state_saved=bool(checkpoint_dir),
                 root_calls=model.boundary_calls, root_max_equivalent_C=model.max_root_equiv,
                 root_max_heat_residual_W_m2=model.max_heat_residual,
                 root_max_moisture_residual_m_s=model.max_moisture_residual,
                 root_max_iterations=model.max_root_iterations,
                 maximum_C_balance_m3_C=float(np.max(abs(balance[:, 0]))),
                 maximum_C_balance_normalized=float(np.max(abs(balance[:, 0]))/(volume*2.55)),
                 maximum_heat_chain_balance_J=float(np.max(abs(balance[:, 1]))),
                 maximum_heat_chain_balance_normalized=float(np.max(abs(balance[:, 1]))/(volume*w0)),
                 accepted_ring_range_T_C=accepted_range[:2],
                 accepted_ring_range_C_kg_kg=accepted_range[2:],
                 diagnostic_only=True, original_t0_surface_preserved=True)
    return dict(time_s=times, radius_m=radii, T_C=T, C_kg_kg=C,
                exchanges=exchange, balances=balance, final_state=y, stats=stats)

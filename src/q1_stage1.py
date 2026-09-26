"""Bounded Q1 stage 1 trials and evidence, never a formal result1.xlsx export."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import quad

from q1_model import (Parameters, RadialModes, DiffusionLaw, RingGrid, MoistureFV,
                      integrate_moisture, read_drives, sha256)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tests"))
from q1_heat_reference import solve_heat_reference, _self_check


def write_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def difference_summary(a,b,times,radii,mask=None):
    select=np.ones(len(times),bool) if mask is None else mask
    errors=np.abs(a[select]-b[select])
    idx=np.unravel_index(np.argmax(errors),errors.shape)
    original=np.flatnonzero(select)[idx[0]]
    return {"max_absolute_difference":float(errors[idx]),"time_s":float(times[original]),
            "radius_m":float(radii[idx[1]]),"first_value":float(a[original,idx[1]]),
            "second_value":float(b[original,idx[1]]),
            "rms_difference":float(np.sqrt(np.mean(errors**2)))}


def run(root, config, config_path, out):
    start=time.perf_counter()
    files=[root/config["inputs"]["statement"],root/config["inputs"]["environment"]]
    files+=sorted((root/"A题/附件/附件3").glob("*.xlsx"))
    inputs_before={str(f.relative_to(root)):sha256(f) for f in files}
    p=Parameters(**config["parameters"])
    ta,ce,nodes=read_drives(root,config)
    stop=config["stage1"]["stop_time_s"]
    if stop>121 or config["stage1"]["formal_export_authorized"]:
        raise ValueError("This executable is restricted to stage 1, at most 121 s")
    times=np.unique(np.r_[np.arange(0,stop+1),config["stage1"]["extra_query_times_s"]])
    times=times[times<=stop]
    radii=np.arange(21,dtype=float)*.001
    integer_mask=times==np.floor(times)
    implementation_files=[root/"src/q1_model.py",Path(__file__),root/"tests/test_q1_components.py",
                          root/"tests/q1_heat_reference.py",config_path]
    hashes={f.relative_to(root).as_posix():sha256(f) for f in implementation_files}
    write_json(out/"config-used.json",config)
    report={"status":"RUNNING","formal_answer":False,"full_1800_s_solved":False,
            "started_utc":datetime.now(timezone.utc).isoformat(),
            "command":[sys.executable,*sys.argv],"python_executable":sys.executable,
            "python_version":sys.version,"versions":{k:importlib.metadata.version(k) for k in
                    ["numpy","scipy","mpmath","openpyxl"]},
            "random_seed":None,"workers":1,"hashes":hashes,"inputs_before":inputs_before,
            "scales":{"alpha_m2_s":p.alpha,"Bi_T":p.bi_heat,
                      "D_initial_m2_s":DiffusionLaw().D(2.55),
                      "initial_moisture_boundary_residual_m_s":p.moisture_transfer_m_s*(2.55-ce(0))},
            "initial_conditions":{"T":28.,"C":2.55,"T_compatible_at_zero":ta(0)==28.,
                "C_compatible_at_zero":ce(0)==2.55,
                "handling":"Retain uniform prescribed t=0 point values; exchange boundary for t>0"},
            "query_times_s":times.tolist(),"radii_m":radii.tolist()}
    write_json(out/"run.json",report)
    # All executable component records must correspond to THIS implementation.
    component_dir=root/config["stage1"]["output_folder"]
    components={}
    for name in ["potential","geometry","manufactured","limits_time","constant_D","modes"]:
        file=component_dir/f"component-{name}.json"
        rec=json.loads(file.read_text(encoding="utf-8"))
        if (rec["status"]!="PASS" or rec["implementation_sha256"]!=hashes["src/q1_model.py"] or
                rec["test_sha256"]!=hashes["tests/test_q1_components.py"]):
            raise ValueError(f"Component {name} is failed or stale; rerun current tests")
        components[name]={"path":str(file.relative_to(root)),"sha256":sha256(file),"status":"PASS"}
    report["component_checks"]=components
    write_json(out/"independent-heat-components.json",_self_check(None))
    print("Current component versions verified; independent heat self-check PASS",flush=True)

    heat={}
    for count in [16,32,64,128,256]:
        mode=RadialModes(p.radius_m,p.alpha,p.bi_heat,count)
        heat[count]=mode.values(times,radii,ta,p.initial_temperature_C)
        np.savez_compressed(out/f"thermal-M{count}.npz",time_s=times,radius_m=radii,
                            T_C=heat[count],roots=mode.roots,coefficients=mode.coefficients,
                            root_residuals=mode.root_residual)
    report["thermal_mode_comparisons"]=[{"from":a,"to":b,
                  **difference_summary(heat[a],heat[b],times,radii)}
                  for a,b in [(16,32),(32,64),(64,128),(128,256)]]
    references={}
    report["independent_heat"]=[]
    for n in [512,1024]:
        ref=solve_heat_reference(times,ta.times,ta.values,cells=n)
        refs=ref.temperatures_at(radii)
        references[n]=refs
        np.savez_compressed(out/f"thermal-independent-N{n}.npz",time_s=times,
                            radius_m=radii,T_C=refs,
                            energy_balance_J=ref.energy_balance_residual_J_m*p.length_m)
        report["independent_heat"].append({"cells":n,
            "versus_Bessel_M256":difference_summary(refs,heat[256],times,radii),
            "maximum_energy_balance_J_for_L_0_25":float(np.max(abs(ref.energy_balance_residual_J_m)))*p.length_m})
    report["independent_heat_refinement"]=difference_summary(references[512],references[1024],times,radii)
    # Independently quadrature-integrate the Bessel surface exchange per input segment.
    final_mode=RadialModes(p.radius_m,p.alpha,p.bi_heat,256)
    total_heat=0.
    segments=np.r_[ta.times[ta.times<stop],stop]
    quadrature_error=0.
    for lo,hi in zip(segments[:-1],segments[1:]):
        def input_power(t):
            ts=final_mode.values([t],[p.radius_m],ta,p.initial_temperature_C)[0,0]
            return 2*np.pi*p.length_m*p.radius_m*p.heat_transfer_W_m2K*(ta(t)-ts)
        val,err=quad(input_power,lo,hi,epsabs=2e-10,epsrel=1e-10,limit=120)
        total_heat+=val; quadrature_error+=err
    stored_heat=float(p.density_kg_m3*p.heat_capacity_J_kgK*np.pi*p.length_m*p.radius_m**2*
                      (final_mode.volume_mean([stop],ta,p.initial_temperature_C)[0]-p.initial_temperature_C))
    report["Bessel_heat_balance"]={"stored_heat_change_J":stored_heat,
            "independent_surface_heat_integral_J":total_heat,"residual_J":stored_heat-total_heat,
            "quadrature_error_estimate_J":quadrature_error,
            "normalization":"abs(stored heat change), finite here",
            "normalized_residual":abs(stored_heat-total_heat)/abs(stored_heat)}
    print("Thermal modes and independent space reference complete",flush=True)
    write_json(out/"run.json",report)

    results={}
    settings=config["numerics"]
    cases=[(n,settings["atol_C"],settings["rtol_C"],f"N{n}") for n in settings["stage1_cells"]]
    cases.append((256,1e-11,1e-9,"N256-time-tight"))
    report["moisture_cases"]={}
    for n,atol,rtol,key in cases:
        grid=RingGrid(n,p.radius_m,p.length_m)
        model=MoistureFV(grid,DiffusionLaw(p.diffusivity_prefactor_m2_s,p.diffusivity_exponent_kg_kg),
                         p.moisture_transfer_m_s,settings["surface_root_xtol_C"],
                         settings["nonlinear_residual_tolerance_C"])
        last_checkpoint=[-np.inf]
        def checkpoint(t,c,exchange,reason):
            clock_now=time.perf_counter()
            if clock_now-last_checkpoint[0]>15 or reason=="timeout" or t==stop:
                np.savez_compressed(out/f"checkpoint-{key}.npz",t_s=t,ring_means_C=c,
                                    cumulative_exchange_m3_C=exchange,reason=reason)
                print(f"{key}: t={t:g} s / {stop:g} s; checkpoint {reason}",flush=True)
                last_checkpoint[0]=clock_now
        sol=integrate_moisture(model,ce,p.initial_moisture_kg_kg,times,
                             atol=atol,rtol=rtol,initial_dt=settings["initial_dt_s"],
                             max_dt=settings["maximum_dt_s"],min_dt=settings["minimum_dt_s"],
                             timeout=settings["case_timeout_s"],checkpoint=checkpoint)
        points=grid.reconstruct(sol.ring_means,radii,sol.surface)
        # Known exact initial field, including the incompatible surface corner.
        assert np.array_equal(sol.ring_means[0],np.full(n,p.initial_moisture_kg_kg))
        assert sol.surface[0] == p.initial_moisture_kg_kg
        # Avoid roundoff from multiplying constant initial data by reconstruction
        # weights. This imposes the prescribed t=0 field, not a positivity clip.
        points[0,:]=p.initial_moisture_kg_kg
        accepted=np.asarray(sol.accepted_half_steps)
        low=min(p.initial_moisture_kg_kg,ce.values[ce.times<=stop].min())
        high=max(p.initial_moisture_kg_kg,ce.values[ce.times<=stop].max())
        overshoot=np.maximum(points-high,low-points)
        violating=np.argwhere(overshoot>5e-11)
        # Reconstruct-and-integrate in r with independent Gauss quadrature per ring.
        x,w=np.polynomial.legendre.leggauss(8)
        rq=(grid.faces[:-1,None]+grid.faces[1:,None])/2+np.diff(grid.faces)[:,None]*x/2
        cq=grid.reconstruct(sol.ring_means[-1],rq.ravel(),sol.surface[-1]).reshape(n,-1)
        reconstructed_integral=float(np.sum((cq*rq)@w*np.diff(grid.faces)/2)*2*np.pi*p.length_m)
        ring_integral=float(grid.volumes@sol.ring_means[-1])
        # Separate continuous-time trapezoidal integral from BE endpoint quadrature.
        trap_exchange=float(np.trapezoid(np.r_[p.moisture_transfer_m_s*(2.55-ce(0)),accepted[:,3]],
                          np.r_[0.,accepted[:,0]])*grid.areas[-1])
        info={"cells":n,"atol_C":atol,"rtol_C":rtol,**sol.diagnostics,
              "maximum_point_range_overshoot":float(max(0.,overshoot.max())),
              "point_range_violations_over_5e_11":[{"t":float(times[i]),"r_m":float(radii[j]),
                     "C":float(points[i,j]),"overshoot":float(overshoot[i,j])} for i,j in violating],
              "reconstructed_minus_ring_integral_m3_C":reconstructed_integral-ring_integral,
              "independent_time_trapezoid_exchange_m3_C":trap_exchange,
              "accepted_BE_exchange_m3_C":float(sol.cumulative_exchange[-1]),
              "time_trapezoid_balance_m3_C":ring_integral-float(grid.volumes.sum()*2.55)+trap_exchange,
              "time_trapezoid_note":"Independent quadrature has time error and a startup corner; not expected algebraic zero"}
        np.savez_compressed(out/f"moisture-{key}.npz",time_s=times,radius_m=radii,C_kg_kg=points,
                            ring_means_C=sol.ring_means,ring_faces_m=grid.faces,surface_C=sol.surface,
                            cumulative_exchange_m3_C=sol.cumulative_exchange,
                            balance_m3_C=sol.balance_residual,accepted_half_steps=accepted)
        write_json(out/f"moisture-{key}-diagnostics.json",info)
        write_json(out/f"moisture-{key}-rejected-steps.json",sol.rejected_steps)
        results[key]=points
        report["moisture_cases"][key]=info
        write_json(out/"run.json",report)
        print(f"{key}: complete in {sol.diagnostics['wall_seconds']:.2f}s; "
              f"max normalized BE balance={sol.diagnostics['maximum_normalized_balance']:.3e}",flush=True)
    report["moisture_space_comparisons"]=[{"from":a,"to":b,
             "all_probes":difference_summary(results[a],results[b],times,radii),
             "integer_seconds":difference_summary(results[a],results[b],times,radii,integer_mask)}
             for a,b in [("N64","N128"),("N128","N256")]]
    report["moisture_time_comparison"]={"same_cells":256,
             "all_probes":difference_summary(results["N256"],results["N256-time-tight"],times,radii),
             "integer_seconds":difference_summary(results["N256"],results["N256-time-tight"],times,radii,integer_mask),
             "interpretation":"Two local tolerance levels only; global time convergence is not yet accepted"}
    # Trial authority, explicitly not a fully validated production authority.
    np.savez_compressed(out/"trial-fullprecision.npz",time_s=times,radius_m=radii,
                        T_C=heat[256],C_kg_kg=results["N256-time-tight"],
                        status="STAGE1_TRIAL_NOT_FORMAL_NOT_ACCURACY_ACCEPTED")
    with (out/"short-trial.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["time_s","ambient_T_C","effective_Ce_kg_kg","centre_T_C",
                         "surface_T_C","centre_C_kg_kg","surface_C_kg_kg"])
        for t in [0.,.01,.1,1.,59.9,60.,60.1,119.9,120.,120.1,121.]:
            i=int(np.flatnonzero(times==t)[0])
            writer.writerow([t,ta(t),ce(t),heat[256][i,0],heat[256][i,-1],
                             results["N256-time-tight"][i,0],results["N256-time-tight"][i,-1]])
    report["inputs_after"]={str(f.relative_to(root)):sha256(f) for f in files}
    if report["inputs_after"]!=inputs_before:
        raise RuntimeError("Original source file changed during run")
    report["all_values_finite"]=bool(all(np.all(np.isfinite(x)) for x in [*heat.values(),*results.values()]))
    report["status"]="STAGE1_COMPLETE_NUMERICAL_ACCEPTANCE_PENDING"
    report["wall_seconds"]=time.perf_counter()-start
    report["stopped_pending_user"]="继续正式计算"
    report["limitations"]=[
        "No 0–1800 s solution or formal export was executed",
        "Moisture short-time spatial accuracy and four-decimal stability are not accepted",
        "Range checks apply separately to means and reconstructed points; no clipping was applied",
        "Physical boundary mapping and neglected latent cooling uncertainties remain unquantified",
        "AI independent numerical checking is not teammate manual verification"]
    report["output_hashes"]={str(f.relative_to(out)):sha256(f) for f in sorted(out.iterdir())
                               if f.is_file() and f.name!="run.json"}
    write_json(out/"run.json",report)
    write_json(component_dir/"latest-run.json",{"path":str(out.relative_to(root)),
                 "manifest_sha256":sha256(out/"run.json"),"status":report["status"]})
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="config/q1.json")
    args=parser.parse_args()
    config_path=ROOT/args.config
    config=json.loads(config_path.read_text(encoding="utf-8"))
    out=ROOT/config["stage1"]["output_folder"]/datetime.now().strftime("run-%Y%m%d-%H%M%S")
    out.mkdir(parents=True,exist_ok=False)
    try:
        report=run(ROOT,config,config_path,out)
        print(json.dumps({"status":report["status"],"out":str(out),
                          "wall_seconds":report["wall_seconds"]},ensure_ascii=False),flush=True)
    except Exception:
        import traceback
        (out/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()

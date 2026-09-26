"""Run one bounded moisture diagnostic case; append-only case artifacts."""
from pathlib import Path
import argparse,json,sys,time
from datetime import datetime,timezone
import importlib.metadata
import numpy as np
from q1_model import MoistureFV,RingGrid,read_drives,integrate_moisture,sha256
from q1_moisture_diagnostic import FastDiffusionLaw,integrate_reference

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cells",type=int,required=True)
    parser.add_argument("--method",choices=["BE","BDF","Radau"],required=True)
    parser.add_argument("--rtol",type=float,required=True)
    parser.add_argument("--stop",type=float,default=121.)
    parser.add_argument("--max-step",type=float,default=.5)
    parser.add_argument("--nonlinear-tol",type=float,default=2e-13)
    parser.add_argument("--root-tol",type=float,default=5e-15)
    parser.add_argument("--label",default="")
    args=parser.parse_args()
    if args.stop>121:raise ValueError("Full time range not authorized")
    cfgpath=ROOT/"config/q1-short-diagnostic.json"
    cfg=json.loads(cfgpath.read_text(encoding="utf-8"))
    basepath=ROOT/cfg["base_configuration"]
    base=json.loads(basepath.read_text(encoding="utf-8"))
    kernel=json.loads((ROOT/cfg["output_folder"]/"kernel-checks.json").read_text(encoding="utf-8"))
    assert kernel["status"]=="PASS"
    assert kernel["new_diagnostic_sha256"]==sha256(ROOT/"src/q1_moisture_diagnostic.py")
    out=ROOT/cfg["output_folder"]
    key=f"{args.method}-N{args.cells}-r{args.rtol:g}-t{args.stop:g}{args.label}"
    case=out/key
    case.mkdir(exist_ok=False,parents=True)
    _,drive,nodes=read_drives(ROOT,base)
    times=np.unique(np.r_[np.arange(int(args.stop)+1),cfg["queries"]["extra_seconds"],args.stop])
    times=times[times<=args.stop]
    radii=np.arange(21)*.001
    grid=RingGrid(args.cells)
    model=MoistureFV(grid,FastDiffusionLaw(),root_xtol=args.root_tol,nonlinear_tol=args.nonlinear_tol)
    wall=time.perf_counter()
    meta={"status":"RUNNING","case":key,"args":vars(args),"command":[sys.executable,*sys.argv],
          "started_utc":datetime.now(timezone.utc).isoformat(),
          "hashes":{p.relative_to(ROOT).as_posix():sha256(p) for p in
            [ROOT/"src/q1_model.py",ROOT/"src/q1_moisture_diagnostic.py",Path(__file__),cfgpath,basepath]},
          "environment_sha256":sha256(ROOT/base["inputs"]["environment"]),
          "versions":{k:importlib.metadata.version(k) for k in ["numpy","scipy"]},
          "workers":1,"random_seed":None,"physical_model_changes":[]}
    def save_meta():
        (case/"case.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    last=[-np.inf]
    def progress(t,c,q,reason):
        if time.perf_counter()-last[0]>15 or t==args.stop or reason=="timeout":
            np.savez_compressed(case/"checkpoint.npz",t_s=t,ring_C=c,exchange=q,reason=reason)
            print(f"{key} t={t:g}/{args.stop:g} wall={time.perf_counter()-wall:.1f}s",flush=True)
            last[0]=time.perf_counter()
    save_meta()
    try:
        if args.method=="BE":
            result=integrate_moisture(model,drive,2.55,times,atol=args.rtol*.01,rtol=args.rtol,
                    initial_dt=.1,max_dt=args.max_step,timeout=cfg["budget"]["per_case_wall_s"],checkpoint=progress)
            means,surf=result.ring_means,result.surface
            extra={"exchange_m3_C":result.cumulative_exchange,"balance_m3_C":result.balance_residual,
                   "accepted_half_steps":np.asarray(result.accepted_half_steps)}
            (case/"rejections.json").write_text(json.dumps(result.rejected_steps,indent=2),encoding="utf-8")
        else:
            result=integrate_reference(model,drive,times,method=args.method,rtol=args.rtol,atol=args.rtol*.01,
                    max_step=args.max_step,timeout=cfg["budget"]["per_case_wall_s"],progress=progress)
            means,surf=result.means,result.surface
            extra={"exchange_m3_C":result.exchange_mean_C*grid.volumes.sum()}
        points=grid.reconstruct(means,radii,surf);points[0,:]=2.55
        assert np.all(np.isfinite(points))
        np.savez_compressed(case/"solution.npz",time_s=times,radius_m=radii,C=points,ring_C=means,
            surface_C=surf,faces_m=grid.faces,**extra)
        meta.update({"status":"COMPLETE_DIAGNOSTIC_NOT_FORMAL","diagnostics":result.diagnostics,
            "wall_seconds":time.perf_counter()-wall,"solution_sha256":sha256(case/"solution.npz"),
            "surface_t1":float(surf[np.flatnonzero(times==1)[0]]) if args.stop>=1 else None,
            "surface_final":float(surf[-1]),"point_maximum_above_initial":float(max(0.,points.max()-2.55)),
            "delta_m":grid.delta,"minimum_ring_volume_m3":float(grid.volumes.min()),
            "total_volume_error_m3":float(grid.volumes.sum()-np.pi*.25*.02**2)})
        save_meta();print(json.dumps({k:meta[k] for k in ["case","status","wall_seconds","surface_t1","surface_final"]}),flush=True)
    except Exception:
        import traceback
        meta.update({"status":"FAILED","traceback":traceback.format_exc(),"wall_seconds":time.perf_counter()-wall})
        save_meta();raise
if __name__=="__main__":main()

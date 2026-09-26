"""Bounded full-period Q1 moisture runs; original inputs are read only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.metadata
import json
import os
import sys
import time
import numpy as np

from q1_model import RingGrid, MoistureFV, read_drives, sha256
from q1_moisture_diagnostic import FastDiffusionLaw
from q1_full_integrators import be_segment, fv_time_reference

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "records/q1-full-20260911"


def queries(stop=1800.):
    extra = [.001,.01,.1,.25,.5,2.5]
    knots = np.arange(60.,1800.,60.)
    t = np.unique(np.r_[np.arange(int(stop)+1,dtype=float),extra,knots-.1,knots+.1,stop])
    return t[t<=stop]


def write_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


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
    args = ap.parse_args()
    assert 0<args.stop<=1800
    if args.atol is None: args.atol=float(f"{args.rtol*.01:.15g}")
    config = json.loads((ROOT/"config/q1-full.json").read_text(encoding="utf-8"))
    physical = json.loads((ROOT/config["base_configuration"]).read_text(encoding="utf-8"))
    assert not config["physical_model_changes"]
    _,drive,nodes = read_drives(ROOT,physical)
    ts,rr = queries(args.stop),np.arange(21)*.001
    key = f"{args.method}-N{args.cells}-r{args.rtol:g}-t{args.stop:g}{args.label}"
    out = BASE/key;out.mkdir(parents=True,exist_ok=False)
    grid = RingGrid(args.cells)
    model = MoistureFV(grid,FastDiffusionLaw(),root_xtol=args.root_tol,nonlinear_tol=args.nonlinear_tol)
    start_clock = time.perf_counter()
    record = {"status":"RUNNING_NOT_ACCEPTED","case":key,"args":vars(args),
        "command":[sys.executable,*sys.argv],"started_utc":datetime.now(timezone.utc).isoformat(),
        "physical_model_changes":[],"workers":1,"random_seed":None,
        "thread_environment":{k:os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]},
        "python":sys.version,"versions":{k:importlib.metadata.version(k) for k in ["numpy","scipy","openpyxl"]},
        "source_sha256":{p:sha256(ROOT/p) for p in ["src/q1_model.py","src/q1_moisture_diagnostic.py",
            "src/q1_full_integrators.py","src/q1_full_compute.py","config/q1.json","config/q1-full.json"]},
        "environment_sha256":sha256(ROOT/physical["inputs"]["environment"]),"segments":[]}
    write_json(out/"case.json",record)
    try:
        if args.method == "BE":
            prefix = ROOT/config["continuation"]["prefix"]
            can_reuse = (not args.fresh and args.cells==8192 and args.rtol==1e-11 and
                         args.atol==1e-13 and args.nonlinear_tol==2e-13 and args.root_tol==5e-15 and args.stop>121)
            all_t=[];all_c=[];all_s=[];all_e=[];all_b=[];all_steps=[]
            if can_reuse:
                pm=json.loads((prefix/"case.json").read_text(encoding="utf-8"))
                assert sha256(prefix/"solution.npz")==config["continuation"]["prefix_solution_sha256"]
                for p,h in pm["hashes"].items(): assert sha256(ROOT/p)==h, p
                with np.load(prefix/"solution.npz",allow_pickle=False) as z:
                    initial=z["ring_C"][-1].copy();exchange=float(z["exchange_m3_C"][-1])
                    begin=float(z["time_s"][-1]);dt=2*float(z["accepted_half_steps"][-1,1])
                    steps=np.column_stack([z["accepted_half_steps"],np.full(len(z["accepted_half_steps"]),-1.)])
                    # Old per-step Newton count was not stored; do not invent it.
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
                            model_sha256=record["source_sha256"]["src/q1_model.py"],reason=reason)
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
            # Prefix contains exactly the same early query points as the full grid.
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


if __name__=="__main__":main()

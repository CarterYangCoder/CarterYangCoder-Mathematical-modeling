"""One additional 65536-ring space refinement from t=0; preserve delivery."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from q1_model import Parameters, RingGrid, MoistureFV, read_drives, sha256
from q1_moisture_diagnostic import FastDiffusionLaw
from q1_full_integrators import fv_time_reference

ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    out = ROOT / "records/q1-reaudit-20260911/BDF-N65536"
    out.mkdir(parents=True, exist_ok=False)
    p = Parameters()
    cfg = json.loads((ROOT / "config/q1.json").read_text(encoding="utf-8"))
    assert cfg["parameters"] == p.__dict__
    _, drive, _ = read_drives(ROOT, cfg)
    g = RingGrid(65536, p.radius_m, p.length_m)
    geometry = {"positive_volumes": bool(np.all(g.volumes > 0)),
        "sum_volume_m3": float(g.volumes.sum()), "exact_volume_m3": float(np.pi*p.length_m*p.radius_m**2),
        "delta_m": float(p.radius_m-g.centres[-1]), "expected_delta_m": p.radius_m/(2*g.cells)}
    assert geometry["positive_volumes"] and np.isclose(geometry["sum_volume_m3"], geometry["exact_volume_m3"], atol=1e-18, rtol=0)
    assert np.isclose(geometry["delta_m"], geometry["expected_delta_m"], atol=1e-17, rtol=0)
    ts, rr = np.arange(1801, dtype=float), np.arange(21)*.001
    averages = .5*(g.faces[:-1]**2 + g.faces[1:]**2)
    recovered = g.reconstruct(averages[None, :], rr, np.array([p.radius_m**2]))[0]
    geometry["r_squared_mean_to_point_max_error_m2"] = float(np.max(abs(recovered-rr**2)))
    assert geometry["r_squared_mean_to_point_max_error_m2"] < 2e-18
    model = MoistureFV(g, FastDiffusionLaw(), p.moisture_transfer_m_s, root_xtol=5e-15)
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in [
        Path(__file__), ROOT/"config/q1.json", ROOT/"src/q1_model.py",
        ROOT/"src/q1_moisture_diagnostic.py", ROOT/"src/q1_full_integrators.py",
        ROOT/cfg["inputs"]["environment"], ROOT/"results/q1/q1-authority.npz", ROOT/"results/q1/result1.xlsx"]}
    record = {"status": "RUNNING", "command": [sys.executable, *sys.argv],
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_input_output_hashes": hashes,
        "reason": "Additional spatial evidence for least-margin 1696s/1.1cm rounding; no authority change.",
        "method": "BDF", "cells": 65536, "rtol": 1e-12, "atol": 1e-14, "max_step_s": .5,
        "initial_state": "Original uniform C=2.55 at t=0; not restarted from coarser mesh",
        "geometry_checks": geometry, "seed": None, "workers": 1,
        "time_limit_per_60s_segment_wall_s": 600, "total_wall_budget_s": 2400,
        "independence": "Same accepted ring FV; additional refinement, not independent spatial method"}
    save(out/"run.json", record)
    start = time.perf_counter()
    def checkpoint(t, state, exchange, values, exchanges, balances, segments):
        np.savez_compressed(out/f"checkpoint-{t:g}.npz", time_s=t, ring_C=state,
                            exchange_m3_C=exchange, faces_m=g.faces)
        record.update(last_completed_time_s=t, elapsed_s=time.perf_counter()-start, segments=segments)
        save(out/"run.json", record)
        print(f"Q1 additional BDF65536 t={t:g}/1800 wall={record['elapsed_s']:.1f}s", flush=True)
        if record["elapsed_s"] > record["total_wall_budget_s"]:
            raise TimeoutError("Declared total wall budget reached; checkpoint preserved")
    try:
        with threadpool_limits(limits=24):
            record["threadpools"] = threadpool_info()
            C, exchange, balance, diagnostics = fv_time_reference(model, drive, ts, rr,
                method="BDF", rtol=1e-12, atol=1e-14, max_step=.5,
                timeout_per_segment=600, checkpoint=checkpoint)
        np.savez_compressed(out/"solution.npz", time_s=ts, radius_m=rr, C=C,
                            exchange_m3_C=exchange, balance_m3_C=balance, faces_m=g.faces)
        for path, expected in hashes.items():
            assert sha256(ROOT/path) == expected, path
        record.update(status="COMPUTATION_COMPLETE_REVIEW_PENDING", diagnostics=diagnostics,
            elapsed_s=time.perf_counter()-start, solution_sha256=sha256(out/"solution.npz"),
            previous_inputs_and_outputs_unchanged=True)
        save(out/"run.json", record)
        print(json.dumps({k: record[k] for k in ["status", "elapsed_s", "solution_sha256"]}), flush=True)
    except Exception as error:
        record.update(status="FAILED_RETAINED", error=repr(error), elapsed_s=time.perf_counter()-start)
        save(out/"run.json", record)
        raise


if __name__ == "__main__":
    main()

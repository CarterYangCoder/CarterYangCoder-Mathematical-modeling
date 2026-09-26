"""Recompute approved Q1 in a new version directory; never overwrite delivery.

Default: read-only delivery check. --run: original t=0 through 1800 s, approved
Radau/FV and Bessel algorithms, full-precision comparison with accepted authority.
"""
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from q1_model import Parameters, RingGrid, MoistureFV, RadialModes, read_drives, sha256
from q1_moisture_diagnostic import FastDiffusionLaw
from q1_full_integrators import fv_time_reference
from q1_export import check_numerical_gate, load_authority, round_half_up
from q1_verify_delivery import main as verify_delivery

ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        verify_delivery()
        return
    verify_delivery()  # Includes the configured-runtime and all current evidence gates.
    authority = ROOT/'results/q1/q1-authority.npz'
    acceptance = json.loads((ROOT/'records/q1-full-20260911/numerical-acceptance.json').read_text(encoding='utf-8'))
    check_numerical_gate(acceptance, authority, False)
    ts, rr, old = load_authority(authority)
    cfg = json.loads((ROOT/'config/q1-accepted.json').read_text(encoding='utf-8'))
    physical = json.loads((ROOT/'config/q1.json').read_text(encoding='utf-8'))
    p = Parameters(**cfg['physical_parameters'])
    assert asdict(p) == physical['parameters']
    temperature_drive, moisture_drive, _ = read_drives(ROOT, physical)
    out = ROOT/'records/q1-full-20260911/reproductions'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    record = {'status':'RUNNING', 'command':[sys.executable,*sys.argv],
        'authority_sha256':sha256(authority), 'accepted_config_sha256':sha256(ROOT/'config/q1-accepted.json'),
        'source_sha256':{str(path.relative_to(ROOT)):sha256(path) for path in [
            Path(__file__),ROOT/'src/q1_model.py',ROOT/'src/q1_moisture_diagnostic.py',ROOT/'src/q1_full_integrators.py']},
        'seed':None,'workers':1,'existing_outputs_policy':'Preserve; verify all bindings again at completion',
        'comparison_policy':{'T_absolute_tolerance_C':1e-11,'C_absolute_tolerance_kg_kg':1e-10,
                             'all_75600_rounded_values_must_match':True}}
    save(out/'run.json',record)
    try:
        # The accepted thermal run used one BLAS thread, moisture used 24.
        with threadpool_limits(limits=1):
            record['thermal_threadpools']=threadpool_info()
            T=RadialModes(p.radius_m,p.alpha,p.bi_heat,1024).values(ts,rr,temperature_drive,28.)
        g=RingGrid(32768,p.radius_m,p.length_m)
        model=MoistureFV(g,FastDiffusionLaw(),p.moisture_transfer_m_s,root_xtol=5e-15)
        def checkpoint(t,state,exchange,values,exchanges,balances,segments):
            np.savez_compressed(out/f'checkpoint-{t:g}.npz',time_s=t,ring_C=state,
                                exchange_m3_C=exchange,faces_m=g.faces)
            print(f'Q1 reproduction t={t:g}/1800 wall={time.perf_counter()-start:.1f}s',flush=True)
        with threadpool_limits(limits=24):
            record['moisture_threadpools']=threadpool_info()
            C,exchange,balance,diagnostics=fv_time_reference(model,moisture_drive,ts,rr,
                method='Radau',rtol=1e-12,atol=1e-14,max_step=.5,
                timeout_per_segment=180,checkpoint=checkpoint)
        np.savez_compressed(out/'reproduced.npz',time_s=ts,radius_m=rr,T_C=T,C_kg_kg=C,
                            exchange_m3_C=exchange,balance_m3_C=balance)
        differences=[float(abs(a-b).max()) for a,b in zip([T,C],old)]
        mismatch=sum(int((round_half_up(a[1:])!=round_half_up(b[1:])).sum()) for a,b in zip([T,C],old))
        passed=differences[0]<=1e-11 and differences[1]<=1e-10 and mismatch==0
        check_numerical_gate(acceptance, authority, False)
        verify_delivery()
        record.update(status='REPRODUCTION_COMPARISON_PASS' if passed else 'REPRODUCTION_DIFFERENCE_REQUIRES_REVIEW',
            maximum_abs_difference_T_C=differences[0],maximum_abs_difference_C_kg_kg=differences[1],
            four_decimal_mismatch_count=mismatch,diagnostics=diagnostics,
            reproduced_sha256=sha256(out/'reproduced.npz'),old_outputs_modified=False,
            elapsed_s=time.perf_counter()-start)
        save(out/'run.json',record)
        print(json.dumps(record,ensure_ascii=False,indent=2))
        if not passed:
            raise RuntimeError('Reproduction exceeds declared tolerance; existing delivery retained unchanged')
    except Exception as exc:
        record.update(status='REPRODUCTION_FAILED_RECORDED',error=str(exc),elapsed_s=time.perf_counter()-start)
        save(out/'run.json',record)
        raise


if __name__=='__main__':
    main()

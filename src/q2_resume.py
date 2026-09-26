"""Safely resume an existing full driver at a completed environment knot.

No checkpoint before 60 s: use a fresh label and original initial conditions.
This avoids duplicate initial-segment audit rows in the frozen v1 driver.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
from q2_model import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--check-only',action='store_true')
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    if Path(args.label).name!=args.label:raise ValueError('A run label, not a path, is required')
    folder=root/'records/q2-full-20260911/runs'/args.label
    record=json.loads((folder/'run.json').read_text(encoding='utf-8'))
    if record['status']=='COMPLETED':raise ValueError('This run is complete; do not resume it')
    segments=record.get('segments',[])
    if not segments:
        raise ValueError('No complete environment checkpoint: rerun from original initial state with a new label')
    if record['status']!='FAILED':
        raise ValueError('Run may still be active; confirm its process has ended before any resume')
    if 'budget exceeded' in record.get('error',''):
        raise ValueError('Recorded total budget reached; do not silently increase it in the bound run')
    driver=root/'src/q2_full_run.py'
    if sha256(driver)!=record['signature']['driver_sha256']:raise ValueError('Driver version changed')
    for seg in segments:
        if sha256(folder/seg['file'])!=seg['sha256']:raise ValueError('Checkpoint changed')
    if 'checkpoint_stats' not in record:raise ValueError('Missing monitor restart information')
    command=[sys.executable,'-B','-X','utf8',str(driver),'--label',args.label,'--resume']
    for key,value in record['signature']['settings'].items():
        flag='--'+key.replace('_','-')
        if key=='atol_T':flag='--atol-T'
        if key=='atol_C':flag='--atol-C'
        if isinstance(value,bool):
            if value:command.append(flag)
        else:command.extend([flag,str(value)])
    print(json.dumps({'resume_knot_s':segments[-1]['end_s'],'command':command,'check_only':args.check_only},ensure_ascii=False))
    if not args.check_only:
        (folder/f'failure-before-resume-{int(time.time())}.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
        raise SystemExit(subprocess.run(command,cwd=root,check=False).returncode)


if __name__=='__main__':main()

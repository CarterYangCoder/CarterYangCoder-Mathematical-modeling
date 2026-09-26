"""Explicit model job execution with a verifiable input/config/result manifest."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import digest, local_path, write_json


def run(root):
    root = Path(root).resolve()
    job_path = root/'model-job.json'
    if not job_path.is_file():
        raise ValueError('No model-job.json: define the actual model before running production.')
    job = json.loads(job_path.read_text(encoding='utf-8'))
    for field in ['entry','inputs','outputs','parameters','units','seed','workers','updating']:
        if field not in job:
            raise ValueError('Missing job field: '+field)
    entry = local_path(root,job['entry'])
    if not entry.is_file() or not entry.is_relative_to(root/'src'):
        raise ValueError('Entry must be an existing source file inside src.')
    if not job['inputs'] or not job['outputs']:
        raise ValueError('Explicit nonempty input and output lists are required.')
    inputs = {p:digest(local_path(root,p)) for p in job['inputs']}
    record = {'status':'running','entry':job['entry'],'job':job,'job_sha256':digest(job_path),
              'input_sha256':inputs,'source_sha256':{p.relative_to(root).as_posix():digest(p) for p in (root/'src').rglob('*.py')},
              'command':'python tools/project.py run','python':sys.version.split()[0],
              'requirements_sha256':digest(root/'requirements-lock.txt'),
              'versions':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
    write_json(root/'records/run-manifest.json',record)
    # Declared output files must be refreshed, not mistaken for old successful results.
    for relative in job['outputs']:
        path = local_path(root,relative)
        if not any(path.is_relative_to(root/p) for p in ['results','figures','paper/assets']):
            raise ValueError('Outputs must be inside results, figures or paper/assets.')
    env = os.environ.copy()
    env['PYTHONUTF8']='1'
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['PYTHONHASHSEED']=str(job['seed'])
    previous_times={p:local_path(root,p).stat().st_mtime_ns if local_path(root,p).exists() else None for p in job['outputs']}
    try:
        result = subprocess.run([sys.executable,str(entry),'--job',str(job_path)],cwd=root,env=env,
                                capture_output=True,text=True,encoding='utf-8',timeout=job.get('timeout_seconds',600))
        (root/'.local').mkdir(exist_ok=True)
        (root/'.local/last-run.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:
            raise RuntimeError('Model failed; inspect .local/last-run.log')
        if {p:digest(local_path(root,p)) for p in job['inputs']} != inputs:
            raise RuntimeError('Original input changed during computation.')
        for relative in job['outputs']:
            if not local_path(root,relative).is_file():
                raise RuntimeError('Declared output is missing: '+relative)
            if previous_times[relative] is not None and local_path(root,relative).stat().st_mtime_ns==previous_times[relative]:
                raise RuntimeError('Declared output was not refreshed: '+relative)
        record['output_sha256']={p:digest(local_path(root,p)) for p in job['outputs']}
        record['status']='success'
    except Exception as exc:
        record.update(status='failed',error=str(exc))
        write_json(root/'records/run-manifest.json',record)
        raise
    write_json(root/'records/run-manifest.json',record)
    return record


def compare(first,second,rtol=1e-9,atol=1e-12):
    """Compare numeric evidence, not PDF/XLSX container timestamps."""
    import numpy as np
    def same(a,b):
        if isinstance(a,bool) or isinstance(b,bool):
            return type(a)==type(b) and a==b
        if isinstance(a,(float,int)) and isinstance(b,(float,int)):
            return bool(np.isfinite(a) and np.isfinite(b) and np.isclose(a,b,rtol=rtol,atol=atol))
        if isinstance(a,dict) and isinstance(b,dict):
            return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,list) and isinstance(b,list):
            return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    return same(first,second)

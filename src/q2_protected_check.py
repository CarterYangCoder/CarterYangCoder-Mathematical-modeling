"""Read-only hash verification of protected Q1, original inputs and short Q2 evidence."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def run(output):
    manifests={
        'Q1':'records/q2-stage1-20260911/input-and-frozen-check.json',
        'Q2_short':'records/q2-stage1-20260911/artifact-manifest.json',
        'raw':'records/q2-plan-20260911/input-review.json'}
    source={k:json.loads((ROOT/v).read_text(encoding='utf-8')) for k,v in manifests.items()}
    groups={
        'Q1':source['Q1']['protected_Q1_sha256'],
        'Q2_short':{x['path']:x['sha256'] for x in source['Q2_short']['files']},
        'raw':{k:v for k,v in source['raw']['protected_sha256_before'].items() if k.startswith('A题')}}
    checked={};errors=[]
    for group,expected in groups.items():
        values={}
        for name,digest in expected.items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT):raise ValueError('Manifest path outside project')
            actual=sha(path) if path.is_file() else None
            values[name]=dict(expected_sha256=digest,actual_sha256=actual,unchanged=actual==digest)
            if actual!=digest:errors.append(dict(group=group,path=name,expected=digest,actual=actual))
        checked[group]=values
    report=dict(status='PROTECTED_INPUTS_Q1_AND_SHORT_Q2_UNCHANGED' if not errors else 'PROTECTED_FILE_MISMATCH',
        checked_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),
        source_manifests={v:sha(ROOT/v) for v in manifests.values()},
        checked_counts={g:len(v) for g,v in checked.items()},checked=checked,errors=errors,
        scope='Byte identity of existing protected artifacts; no claim of physical truth or new computation.')
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'records/q2-full-20260911'):raise ValueError('Report must stay in current records')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ['status','checked_counts','errors']},ensure_ascii=False))
    if errors:raise RuntimeError('Protected files differ; see saved report')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    run(p.parse_args().output)

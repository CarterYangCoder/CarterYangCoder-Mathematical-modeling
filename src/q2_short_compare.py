"""Compare saved Q2 trajectories at identical physical coordinates."""
from pathlib import Path
import json
import argparse
import numpy as np
from q2_model import sha256

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'records/q2-stage1-20260911/runs'


def load(label):
    meta=json.loads((RUNS/label/'run.json').read_text(encoding='utf-8'))
    if meta['status']!='COMPUTATION_COMPLETED_NOT_FORMAL_ACCEPTANCE':
        raise ValueError(f'Cannot compare an incomplete or failed run: {label}')
    for rel,key in [('src/q2_model.py','code_sha256'),('src/q1_model.py','helper_sha256')]:
        if sha256(ROOT/rel)!=meta[key]:
            raise ValueError(f'Code version changed; invalidate dependent Q2 report: {label}, {rel}')
    if sha256(ROOT/'config/q2.json')!=meta['config_sha256']:
        raise ValueError(f'Physical configuration changed; invalidate dependent report: {label}')
    if sha256(RUNS/label/'solution.npz')!=meta['solution_sha256']:
        raise ValueError(f'Saved Q2 result changed: {label}')
    with np.load(RUNS/label/'solution.npz') as d:
        return {k:d[k] for k in d.files}


def compare(a, b, *, minimum_time=1):
    assert np.array_equal(a['time_s'], b['time_s'])
    assert np.array_equal(a['radius_m'], b['radius_m'])
    sel = a['time_s'] >= minimum_time
    result = {}
    for key in ('T_C','C_kg_kg'):
        diff = abs(a[key][sel]-b[key][sel])
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        it = np.flatnonzero(sel)[idx[0]]; ir = idx[1]
        result[key] = dict(max_abs=float(diff[idx]), time_s=float(a['time_s'][it]),
                           radius_m=float(a['radius_m'][ir]), value_a=float(a[key][it,ir]),
                           value_b=float(b[key][it,ir]),
                           four_decimal_differences=int(np.count_nonzero(np.round(a[key][sel],4)!=np.round(b[key][sel],4))))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('labels', nargs='+'); args=ap.parse_args()
    for aa,bb in zip(args.labels[:-1],args.labels[1:]):
        a,b = load(aa),load(bb)
        print(json.dumps(dict(pair=[aa,bb], output_from_1_s=compare(a,b),
                              including_subseconds=compare(a,b,minimum_time=0)), ensure_ascii=False))

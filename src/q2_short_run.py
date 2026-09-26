"""Bounded Q2 short-time run, saved full precision and provenance; no Excel."""
from pathlib import Path
import argparse
import json
import sys
import time
import platform
import numpy as np
import scipy
from threadpoolctl import threadpool_limits, threadpool_info
from q2_model import CoupledFV, Properties, read_environment, run_short, sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', type=int, required=True)
    ap.add_argument('--flux', choices=['potential','fick'], default='potential')
    ap.add_argument('--method', choices=['Radau','BDF'], default='Radau')
    ap.add_argument('--rtol', type=float, default=1e-9)
    ap.add_argument('--atol-T', type=float, default=1e-10)
    ap.add_argument('--atol-C', type=float, default=1e-12)
    ap.add_argument('--max-step', type=float, default=1.)
    ap.add_argument('--first-step-factor', type=float, default=.01)
    ap.add_argument('--stop', type=float, default=121.)
    ap.add_argument('--test-q1', action='store_true')
    ap.add_argument('--label', required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'config/q2.json').read_text(encoding='utf-8'))
    if not config['short_trials_authorized']:
        raise ValueError('Missing Q2 short trial authorization')
    dest = root/'records/q2-stage1-20260911/runs'/args.label
    dest.mkdir(parents=True, exist_ok=False)
    metadata = dict(command=sys.argv, interpreter=sys.executable, python=platform.python_version(),
                    numpy=np.__version__, scipy=scipy.__version__, settings=vars(args),
                    inputs=config['inputs'], config_sha256=sha256(root/'config/q2.json'),
                    code_sha256=sha256(root/'src/q2_model.py'), entry_sha256=sha256(__file__),
                    helper_sha256=sha256(root/'src/q1_model.py'), status='RUNNING',
                    test_configuration=args.test_q1, threadpools=threadpool_info())
    metadata_path = dest/'run.json'
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    started = time.perf_counter()
    try:
        td, cd, _ = read_environment(root, config)
        targets = np.unique(np.r_[0., .001, .01, .1, .25, .5,
                                 np.arange(1., np.floor(args.stop)+1),
                                 59.9, 60., 60.1, 119.9, 120., 120.1, args.stop])
        targets = targets[targets <= args.stop]
        radii = np.arange(21)*.001
        model = CoupledFV(args.cells, flux=args.flux, properties=Properties(args.test_q1))
        result = run_short(model, (td, cd), targets, radii, method=args.method,
                           rtol=args.rtol, atol_T=args.atol_T, atol_C=args.atol_C,
                           max_step=args.max_step, first_step_factor=args.first_step_factor,
                           checkpoint_dir=dest)
        metadata.update(status='COMPUTATION_COMPLETED_NOT_FORMAL_ACCEPTANCE', stats=result.pop('stats'))
        np.savez_compressed(dest/'solution.npz', **result)
        metadata['solution_sha256'] = sha256(dest/'solution.npz')
    except Exception as exc:
        metadata.update(status='FAILED', error=repr(exc))
        raise
    finally:
        metadata['total_wall_s'] = time.perf_counter()-started
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(dict(status=metadata['status'], label=args.label,
                              wall_s=metadata['total_wall_s'], report=str(metadata_path)),
                         ensure_ascii=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()

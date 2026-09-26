"""Build a local LaTeX entrypoint without shell escape. No files are uploaded."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import hashlib


def build(main, tex_bin=None):
    main = Path(main).resolve()
    if not main.is_file() or main.suffix.lower() != '.tex':
        raise ValueError(f'Missing .tex entrypoint: {main}')
    env = os.environ.copy()
    if tex_bin:
        env['PATH'] = str(Path(tex_bin).resolve()) + os.pathsep + env.get('PATH', '')
    latexmk = shutil.which('latexmk', path=env.get('PATH'))
    if not latexmk:
        raise RuntimeError('latexmk not found; configure tex_bin or install a TeX distribution.')
    out = main.parent / 'build'
    out.mkdir(exist_ok=True)
    report_path = out / (main.stem + '.build-report.json')
    report_path.write_text(json.dumps({'main': str(main), 'success': False, 'status': 'running'}), encoding='utf-8')
    command = [latexmk, '-norc', '-xelatex', '-interaction=nonstopmode',
               '-halt-on-error', '-file-line-error', '-outdir=build',
               '-latexoption=-no-shell-escape', main.name]
    try:
        result = subprocess.run(command, cwd=main.parent, env=env, capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=240)
    except (subprocess.TimeoutExpired, OSError) as exc:
        captured = getattr(exc, 'stdout', '') or ''
        if isinstance(captured, bytes):
            captured = captured.decode('utf-8', errors='replace')
        (out / (main.stem + '.compiler-output.txt')).write_text(captured + '\n' + str(exc), encoding='utf-8')
        report_path.write_text(json.dumps({'main': str(main), 'success': False, 'status': 'failed', 'error': str(exc)}), encoding='utf-8')
        raise
    (out / (main.stem + '.compiler-output.txt')).write_text(result.stdout + result.stderr, encoding='utf-8')
    log = out / (main.stem + '.log')
    contents = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
    warnings = [line.strip() for line in contents.splitlines() if re.search(
        r'Warning:|Overfull|Missing character|undefined', line, re.I)]
    pdf = out / (main.stem + '.pdf')
    report = {'main': str(main), 'pdf': str(pdf), 'returncode': result.returncode,
              'command': command, 'warnings': warnings,
              'success': result.returncode == 0 and pdf.is_file()}
    if report['success']:
        report['pdf_sha256'] = hashlib.sha256(pdf.read_bytes()).hexdigest()
        report['source_sha256'] = {p.relative_to(main.parent).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in main.parent.rglob('*') if p.is_file() and 'build' not in p.relative_to(main.parent).parts}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if not report['success']:
        raise RuntimeError(f'Build failed; see {out / "compiler-output.txt"}\n{result.stdout[-3500:]}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main')
    parser.add_argument('--tex-bin')
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.main, args.tex_bin), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

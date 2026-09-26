"""Read-only Q1 delivery verification; no PDE rerun, overwrite or submission."""
from pathlib import Path
import csv
import json
import os
import sys
import numpy as np
from q1_export import (check_numerical_gate, load_authority, round_half_up,
                       template_styles, verify_xlsx, digest, TABLE_TIMES,
                       TABLE_RADIAL_INDEX, project_path)

ROOT = Path(__file__).resolve().parents[1]


def main():
    runtime_cfg = json.loads((ROOT/'project_config.json').read_text(encoding='utf-8'))
    expected_python = Path(runtime_cfg['python'].replace('${USERPROFILE}', os.environ['USERPROFILE'])).resolve()
    assert Path(sys.executable).resolve() == expected_python, 'Use project_config.json Python'
    directory = ROOT/'results/q1'
    authority = directory/'q1-authority.npz'
    acceptance_path = ROOT/'records/q1-full-20260911/numerical-acceptance.json'
    acceptance = json.loads(acceptance_path.read_text(encoding='utf-8'))
    check_numerical_gate(acceptance, authority, False)
    _, _, fields = load_authority(authority)
    rounded = [round_half_up(field[1:]) for field in fields]
    manifest = json.loads((directory/'q1-export-manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'Q1_NUMERICAL_EXPORT_READBACK_PASS'
    assert digest(acceptance_path) == manifest['acceptance_sha256']
    for group in ['source_sha256', 'original_templates', 'output_sha256']:
        for relative, expected in manifest[group].items():
            assert digest(project_path(relative)) == expected, relative
    assert digest(ROOT/'project_config.json') == manifest['project_config_sha256']
    verified = verify_xlsx(directory/'result1.xlsx', rounded, template_styles())
    for i, field in enumerate(rounded, 1):
        with (directory/f'table{i}.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.reader(f))
        assert [int(row[0]) for row in rows[1:]] == list(TABLE_TIMES)
        actual = np.array([[float(x) for x in row[1:]] for row in rows[1:]])
        expected = field[np.array(TABLE_TIMES)-1][:, TABLE_RADIAL_INDEX]
        assert np.array_equal(actual, expected)
    print(json.dumps({'status':'Q1_CURRENT_DELIVERY_VERIFY_PASS',
        'checked_numeric_results':verified['checked_numeric_results'],
        'checked_paper_values':verified['checked_paper_table_links'],
        'authority_sha256':digest(authority), 'workbook_sha256':digest(directory/'result1.xlsx'),
        'numerical_acceptance_and_evidence_current':True,
        'unresolved_4dp':acceptance['unresolved_rounding_count'],
        'human_review_performed':False, 'full_contest_submission_ready':False}, ensure_ascii=False))


if __name__ == '__main__':
    main()

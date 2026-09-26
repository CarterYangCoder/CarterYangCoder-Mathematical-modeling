"""Run frozen artifact authoring, restore original template styles, then verify.

The artifact exporter normalizes A1 font/border, which the strict template gate
rightly rejected. This adapter restores styles from the original workbook;
all cell values are checked unchanged and the existing 75,600-value gate reruns.
"""
from copy import copy
from pathlib import Path
import sys
import openpyxl
import q1_export as base

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
SCRIPT_SHA=base.digest(SCRIPT)
original_verify=base.verify_xlsx
original_save=base.save_json


def audit_directory():
    if '--audit-dir' in sys.argv:
        return base.project_path(sys.argv[sys.argv.index('--audit-dir')+1])
    if '--job' in sys.argv:
        raise ValueError('Style adapter uses direct explicit arguments, not --job')
    return ROOT/'records/q1-full-20260911/export-v2'


def restore_then_verify(path,rounded,styles):
    audit=audit_directory()
    audit.mkdir(parents=True,exist_ok=True)
    preserved=audit/'artifact-before-style-restoration.xlsx'
    if preserved.exists():
        raise FileExistsError('Keep each failed or restored authoring version separately')
    preserved.write_bytes(path.read_bytes())
    if base.digest(base.TEMPLATE)!=base.TEMPLATE_SHA:
        raise ValueError('Original template changed')
    reference=openpyxl.load_workbook(base.TEMPLATE)
    target=openpyxl.load_workbook(path)
    before={s.title:tuple(tuple(cell.value for cell in row) for row in s) for s in target}
    # Style restoration changes no numeric value and preserves theme semantics.
    target.loaded_theme=reference.loaded_theme
    for name in base.NAMES:
        original=reference[name]
        sheet=target[name]
        sheet['A1']._style=copy(original['A1']._style)
        # Style IDs belong to their workbook, so copy actual style properties.
        for attr in ('font','fill','border','alignment','protection','number_format'):
            setattr(sheet['A1'],attr,copy(getattr(original['A1'],attr)))
        sheet.column_dimensions['A'].width=original.column_dimensions['A'].width
        sheet.row_dimensions[1].height=original.row_dimensions[1].height
        for row in sheet.iter_rows(min_row=1,max_row=1801,max_col=22):
            for cell in row:
                if cell.coordinate=='A1':
                    continue
                style_source=original['B1'] if cell.row==1 else original['A2'] if cell.column==1 else original['B2']
                for attr in ('font','fill','border','alignment','protection'):
                    setattr(cell,attr,copy(getattr(style_source,attr)))
                cell.number_format='0.0' if cell.row==1 else '0' if cell.column==1 else '0.0000'
    after={s.title:tuple(tuple(cell.value for cell in row) for row in s) for s in target}
    if before!=after:
        raise AssertionError('Style adapter unexpectedly changed values')
    target.save(path)
    target.close();reference.close()
    reopened=openpyxl.load_workbook(path,read_only=True,data_only=False)
    actual={s.title:tuple(tuple(row) for row in s.iter_rows(values_only=True)) for s in reopened}
    reopened.close()
    if before!=actual:
        raise AssertionError('Values changed after style-restoration serialization')
    result=original_verify(path,rounded,styles)
    original_save(audit/'template-style-restoration.json',{
        'status':'ORIGINAL_TEMPLATE_STYLES_RESTORED_VALUES_UNCHANGED',
        'template_sha256':base.TEMPLATE_SHA,'artifact_before_sha256':base.digest(preserved),
        'styled_after_sha256':base.digest(path),'all_cell_values_identical':True,
        'style_reference':'Original A1, repeated B1/A2/B2 template styles; numeric formats remain specified',
        'methods':'artifact-tool authoring and rendering; openpyxl original-style restoration and independent readback',
        'adapter_sha256':SCRIPT_SHA})
    return result


def save_with_adapter_source(path,data):
    if 'source_sha256' in data and 'export_parameters' in data:
        data['source_sha256'][str(SCRIPT.relative_to(ROOT))]=SCRIPT_SHA
        data['style_restoration_adapter']=str(SCRIPT.relative_to(ROOT))
    return original_save(path,data)


if __name__=='__main__':
    if '--audit-dir' not in sys.argv:
        sys.argv += ['--audit-dir','records/q1-full-20260911/export-v2']
    base.verify_xlsx=restore_then_verify
    base.save_json=save_with_adapter_source
    sys.exit(base.main())

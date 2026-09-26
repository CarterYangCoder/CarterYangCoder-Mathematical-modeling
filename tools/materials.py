"""Explicit support staging, source appendix, truthful AI log PDF, source bundle."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import zipfile
from common import local_path,write_json,digest
from build_latex import build
from precheck import scan_file,check,identity_terms


def escape(text):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(replacements.get(c,c) for c in str(text))


def ai_details(root,tex_bin):
    root=Path(root)
    cfg=json.loads((root/'submission-manifest.json').read_text(encoding='utf-8'))
    rows=list(csv.DictReader((root/'records/AI使用日志.csv').open(encoding='utf-8-sig',newline='')))
    rows=[r for r in rows if r.get('scope')==cfg.get('ai_log_scope','contest')]
    if not rows: raise ValueError('No actual AI log entries for the selected work scope')
    with tempfile.TemporaryDirectory(prefix='cumcm-ai-') as temporary:
        folder=Path(temporary)
        parts=[r'\documentclass[UTF8,a4paper,zihao=-4]{ctexart}',r'\usepackage[margin=2.5cm]{geometry}',r'\begin{document}',r'\section*{AI工具使用详情}']
        for i,row in enumerate(rows,1):
            parts.append(r'\subsection*{记录 '+str(i)+'}')
            for k in ['time','tool_model','purpose','prompt_process','adopted_modified','human_verification']:
                if not row.get(k): raise ValueError('AI log field is empty: '+k)
                parts.append(r'\noindent\textbf{'+escape(k)+r'}：'+escape(row[k])+r'\par')
        parts.append(r'\end{document}')
        (folder/'main.tex').write_text('\n'.join(parts),encoding='utf-8')
        result=build(folder/'main.tex',tex_bin)
        if result['warnings']: raise RuntimeError('AI details compilation warnings: '+str(result['warnings']))
        shutil.copy2(folder/'build/main.pdf',root/'AI工具使用详情.pdf')
    write_json(root/'records/ai-details-manifest.json',{'log_sha256':digest(root/'records/AI使用日志.csv'),
        'scope':cfg.get('ai_log_scope','contest'),'pdf_sha256':digest(root/'AI工具使用详情.pdf'),'records':len(rows)})


def appendix(root):
    root=Path(root)
    cfg=json.loads((root/'submission-manifest.json').read_text(encoding='utf-8'))
    names=cfg.get('support_files',[])
    if not names or len(names)!=len(set(names)): raise ValueError('Explicit unique support list required')
    generated=root/'paper/generated'
    generated.mkdir(exist_ok=True)
    lines=[r'\section{支撑材料文件列表}',r'\begin{itemize}']
    for name in names:
        if not local_path(root,name).is_file(): raise ValueError('Missing support file: '+name)
        lines.append(r'\item '+escape(name))
    lines += [r'\end{itemize}',r'\section{完整源程序}']
    code={}
    for i,name in enumerate(names):
        path=local_path(root,name)
        if path.suffix.lower() in {'.py','.m','.r','.cpp','.c','.h','.jl'}:
            target=generated/f'code-{i}{path.suffix}'
            shutil.copy2(path,target)
            code[name]={'copy':target.relative_to(root/'paper').as_posix(),'sha256':digest(path)}
            lines += [r'\subsection*{'+escape(name)+'}',r'\VerbatimInput[breaklines=true,breakanywhere=true,fontsize=\scriptsize]{generated/'+target.name+'}']
    (generated/'appendix.tex').write_text('\n'.join(lines),encoding='utf-8')
    write_json(generated/'support-index.json',{'files':names,'code':code})


def stage(root):
    root=Path(root)
    cfg=json.loads((root/'submission-manifest.json').read_text(encoding='utf-8'))
    names=cfg.get('support_files',[])
    if not names: raise ValueError('No support files selected')
    destination=root/'.local/candidate/support.zip'
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            path=local_path(root,name)
            if not path.is_file() or path.is_symlink(): raise ValueError('Invalid support file: '+name)
            archive.write(path,name)
    return destination


def package(root,tex_bin):
    root=Path(root)
    source=root/'paper'
    files=[p for p in source.rglob('*') if p.is_file() and 'build' not in p.relative_to(source).parts and '__pycache__' not in p.parts]
    with tempfile.TemporaryDirectory(prefix='cumcm-overleaf-') as temporary:
        folder=Path(temporary)
        for path in files:
            relative=path.relative_to(source)
            if not path.resolve().is_relative_to(source.resolve()) or path.is_symlink(): raise ValueError('Unsafe paper path')
            problems=scan_file(relative.as_posix(),path.read_bytes(),identity_terms(root))
            if problems: raise ValueError('Source bundle scan failed: '+str(problems))
            target=folder/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
        report=build(folder/'main.tex',tex_bin)
        if report['warnings']: raise ValueError('Source bundle build warnings: '+str(report['warnings']))
        destination=root/'submission/overleaf-source.zip'
        destination.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in files: archive.write(path,path.relative_to(source).as_posix())
    write_json(root/'records/package-check.json',{'relocated_build':True,'cloud_build':False,'files':[p.relative_to(source).as_posix() for p in files]})
    return {'source_bundle':'submission/overleaf-source.zip','relocated_build':True,'purpose':'editing only'}


def release(root):
    root=Path(root)
    report=check(root)
    if not report['release_ready']: raise ValueError('Release refused; inspect records/precheck.json')
    destination=root/'submission'
    destination.mkdir(exist_ok=True)
    shutil.copy2(root/'paper/build/main.pdf',destination/'paper.pdf')
    shutil.copy2(root/'.local/candidate/support.zip',destination/'support.zip')
    return {'paper':'submission/paper.pdf','support':'submission/support.zip','note':'Generic local filenames; no upload performed'}

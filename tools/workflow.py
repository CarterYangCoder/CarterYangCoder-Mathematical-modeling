import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from common import settings,write_json
from build_latex import build


def main():
    parser=argparse.ArgumentParser(description='Model, paper and support workflow. No external administration.')
    parser.add_argument('task',choices=['doctor','selftest','ingest','run','prepare','build','package','check','release'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    root=args.root.resolve();cfg=settings(root)
    if Path(sys.executable).resolve()!=Path(cfg['python']).resolve():
        raise ValueError('Use the Python interpreter configured in project_config.json')
    if args.task in {'doctor','selftest'}:
        from selftest import run
        functional=run()
        result={'functional':functional}
        if args.task=='doctor':
            names={'statsmodels':'statsmodels','scikit-learn':'sklearn','ortools':'ortools','cvxpy':'cvxpy','networkx':'networkx','simpy':'simpy','SALib':'SALib','xlrd':'xlrd','pdfplumber':'pdfplumber','python-docx':'docx','shapely':'shapely','pymoo':'pymoo'}
            for module in names.values():importlib.import_module(module)
            check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
            if check.returncode:raise RuntimeError(check.stdout+check.stderr)
            for name in ['xelatex.exe','latexmk.exe','biber.exe']:
                subprocess.run([str(Path(cfg['tex_bin'])/name),'--version'],check=True,capture_output=True)
            result['packages']={p:importlib.metadata.version(p) for p in names}
            result['pip_check']=check.stdout.strip()
        write_json(root/'records/healthcheck.json',result)
    elif args.task=='ingest':
        from ingest import ingest
        result=ingest(root)
    elif args.task=='run':
        from production import run
        result=run(root)
    elif args.task=='prepare':
        from materials import ai_details,appendix,stage
        manifest=json.loads((root/'submission-manifest.json').read_text(encoding='utf-8'))
        if manifest.get('ai_used'):ai_details(root,cfg['tex_bin'])
        appendix(root)
        stage(root)
        result={'prepared':True,'note':'Candidate only. Build and check next.'}
    elif args.task=='build':
        result=build(root/'paper/main.tex',cfg['tex_bin'])
    elif args.task=='package':
        from materials import package
        result=package(root,cfg['tex_bin'])
    elif args.task=='check':
        from precheck import check
        result=check(root)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if result['technical_pass'] else 2
    else:
        from materials import release
        result=release(root)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        print(str(exc),file=sys.stderr);sys.exit(1)

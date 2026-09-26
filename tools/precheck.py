"""Fail-closed technical checks of the actual paper and support candidate."""
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import zipfile
from pypdf import PdfReader
from common import digest, local_path, write_json

MAX_BYTES = 20_000_000  # conservative decimal MB, independent per artifact
PLACEHOLDER = re.compile(r'待填写|待根据|待补充|TODO|PLACEHOLDER|尚未开始|不可直接提交|草稿骨架|〖|〗', re.I)
TEXT_EXT = {'.py','.m','.r','.cpp','.c','.h','.jl','.json','.csv','.txt','.tex','.bib','.md','.yaml','.yml','.toml','.sty','.cls','.cfg','.def','.tikz','.pgf'}


def identity_terms(root):
    terms=[os.environ.get('USERNAME',''),Path.home().name]
    identity_path=Path(root)/'.local/identity.json'
    if identity_path.exists():
        terms += json.loads(identity_path.read_text(encoding='utf-8')).get('terms',[])
    return terms


def identity_findings(text, terms=()):
    patterns = [r'[A-Za-z]:[\\/]+Users[\\/]+[^\s\\/"{}]+',
                r'/home/[^\s/]+', r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}',
                r'(?:姓名|学号|参赛学校|所属赛区)\s*[:：]\s*\S+', r'\b(?:Author|Committer):\s*.+']
    found = [f'pattern:{i}' for i,p in enumerate(patterns) if re.search(p,text,re.I)]
    for term in terms:
        if term and re.search(r'(?<![\w])'+re.escape(term)+r'(?![\w])' if term.isdigit() else re.escape(term), text, re.I):
            found.append('configured_identity')
    return sorted(set(found))


def read_content(name, data):
    suffix = Path(name).suffix.lower()
    if suffix == '.pdf':
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError('Encrypted PDF cannot be inspected')
        return '\n'.join(p.extract_text() or '' for p in reader.pages)+'\n'+str(reader.metadata)
    if suffix in {'.xlsx','.docx'}:
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if sum(x.file_size for x in z.infolist()) > 100_000_000:
                raise ValueError('Office archive exceeds inspection budget')
            pieces=[]
            for n in z.namelist():
                if n.endswith('.xml'):
                    content=z.read(n).decode('utf-8',errors='replace')
                    pieces += [content,''.join(ET.fromstring(content).itertext())]
            return '\n'.join(pieces)
    if suffix == '.xls':
        import xlrd
        book = xlrd.open_workbook(file_contents=data)
        return '\n'.join(str(sheet.row_values(i)) for sheet in book.sheets() for i in range(sheet.nrows))
    if suffix in {'.png','.jpg','.jpeg'}:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as img:
            return str(img.info)+'\n'+str(img.getexif())
    if suffix in TEXT_EXT:
        for enc in ['utf-8-sig','gb18030']:
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                pass
        raise ValueError('Unrecognized text encoding')
    raise ValueError('Unsupported final file type: '+suffix)


def scan_file(name,data,terms=(),placeholders=False):
    findings = identity_findings(name,terms)
    try:
        content = read_content(name,data)
        findings += identity_findings(content,terms)
        if placeholders and PLACEHOLDER.search(content):
            findings.append('unfinished_placeholder')
    except Exception as exc:
        findings.append('unscannable:'+str(exc))
    return sorted(set(findings))


def check(root,paper=None,support=None):
    root = Path(root).resolve()
    paper = Path(paper) if paper else root/'paper/build/main.pdf'
    support = Path(support) if support else root/'.local/candidate/support.zip'
    errors=[]
    terms=identity_terms(root)
    cfg_path=root/'submission-manifest.json'
    cfg=json.loads(cfg_path.read_text(encoding='utf-8')) if cfg_path.exists() else {}
    if not cfg:
        errors.append('No submission-manifest.json')
    manifest=root/'records/run-manifest.json'
    if not manifest.exists():
        errors.append('No verified production run')
    else:
        run=json.loads(manifest.read_text(encoding='utf-8'))
        if run.get('status')!='success':
            errors.append('Latest model run did not succeed')
        if not (root/'model-job.json').is_file() or digest(root/'model-job.json')!=run.get('job_sha256'):
            errors.append('Model configuration changed since run')
        if digest(root/'requirements-lock.txt')!=run.get('requirements_sha256'):
            errors.append('Dependency lock changed since run')
        for group in ['input_sha256','source_sha256','output_sha256']:
            if not run.get(group): errors.append('Missing '+group)
            for name,expected in run.get(group,{}).items():
                path=local_path(root,name)
                if not path.is_file() or digest(path)!=expected:
                    errors.append('Stale model evidence: '+name)
    if not paper.is_file():
        errors.append('Paper PDF is missing')
    else:
        if paper.stat().st_size>MAX_BYTES: errors.append('Paper exceeds 20 MB')
        errors += ['Paper: '+x for x in scan_file(paper.name,paper.read_bytes(),terms,True)]
        reader=PdfReader(paper)
        pages=[p.extract_text() or '' for p in reader.pages]
        full=re.sub(r'\s+','', '\n'.join(pages))
        if not pages or '摘要' not in pages[0]: errors.append('First PDF page is not an abstract page')
        if pages and not re.search(r'\b1\s*$',pages[0]): errors.append('Abstract page footer is not numbered 1')
        if '目录' in full: errors.append('Contents heading found; inspect and remove contents page')
        if '承诺书' in full or '编号专用页' in full: errors.append('Identity cover pages found')
        for page in reader.pages:
            if abs(float(page.mediabox.width)-595.276)>2 or abs(float(page.mediabox.height)-841.89)>2:
                errors.append('Non-A4 page'); break
        aux=paper.parent/'main.aux'
        text=aux.read_text(encoding='utf-8',errors='replace') if aux.exists() else ''
        end=re.search(r'\\newlabel\{body:end\}\{\{[^}]*\}\{(\d+)\}',text)
        if not end: errors.append('No compiled body:end page boundary')
        elif not 1 <= int(end.group(1))-1 <= 30: errors.append('Main body exceeds 30 pages or is empty')
        title=full.find('AI工具使用声明')
        refs=full.find('参考文献')
        if title<0 or refs<0 or title>refs: errors.append('AI declaration must precede references')
        if cfg.get('ai_used') is True:
            if not re.search('本参赛队在竞赛过程中使用了AI工具，主要用于.+?，详细使用情况见支撑材料。',full):
                errors.append('AI-use declaration wording is incomplete')
        elif cfg.get('ai_used') is False:
            if '本参赛队在竞赛过程中未使用任何AI工具。' not in full: errors.append('No-AI declaration missing')
        else: errors.append('AI usage state is unset')
        br=paper.parent/'main.build-report.json'
        if not br.is_file(): errors.append('Current build report missing')
        else:
            b=json.loads(br.read_text(encoding='utf-8'))
            if not b.get('success') or b.get('pdf_sha256')!=digest(paper): errors.append('PDF does not match successful build')
            for name,expected_hash in b.get('source_sha256',{}).items():
                current=local_path(root/'paper',name)
                if not current.is_file() or digest(current)!=expected_hash: errors.append('Paper source changed since build: '+name)
            if b.get('warnings'): errors.append('Build warnings require resolution')
    expected=set(cfg.get('support_files',[]))
    if cfg.get('ai_used'):
        ai_manifest=root/'records/ai-details-manifest.json'
        if not ai_manifest.is_file():errors.append('AI details provenance is missing')
        else:
            ai=json.loads(ai_manifest.read_text(encoding='utf-8'))
            ai_pdf=root/'AI工具使用详情.pdf';ai_log=root/'records/AI使用日志.csv'
            if not ai_pdf.is_file() or not ai_log.is_file() or ai.get('pdf_sha256')!=digest(ai_pdf) or ai.get('log_sha256')!=digest(ai_log) or ai.get('scope')!=cfg.get('ai_log_scope','contest') or ai.get('records',0)<1:
                errors.append('AI details are stale or inconsistent with the actual log')
    if not expected: errors.append('No explicit support files')
    job=root/'model-job.json'
    if job.exists():
        j=json.loads(job.read_text(encoding='utf-8'))
        required={j.get('entry',''),'model-job.json','requirements-lock.txt',*j.get('outputs',[]),*j.get('inputs',[])}
        if not required.issubset(expected): errors.append('Support list omits model entry/config/outputs/lock')
        all_code={p.relative_to(root).as_posix() for p in (root/'src').rglob('*') if p.is_file() and p.suffix.lower() in {'.py','.m','.r','.cpp','.jl'}}
        if not all_code.issubset(expected): errors.append('Support list omits source code')
    appendix=root/'paper/generated/support-index.json'
    if not appendix.exists() or set(json.loads(appendix.read_text(encoding='utf-8')).get('files',[]))!=expected:
        errors.append('Appendix support list is missing or out of date')
    else:
        index=json.loads(appendix.read_text(encoding='utf-8'))
        for name,item in index.get('code',{}).items():
            original=local_path(root,name);copied=local_path(root/'paper',item['copy'])
            if not original.is_file() or not copied.is_file() or digest(original)!=item['sha256'] or digest(copied)!=item['sha256']:
                errors.append('Appendix source is stale: '+name)
        expected_code={n for n in expected if Path(n).suffix.lower() in {'.py','.m','.r','.cpp','.c','.h','.jl'}}
        if set(index.get('code',{}))!=expected_code:errors.append('Appendix source listing is incomplete')
    if not support.is_file(): errors.append('Support ZIP is missing')
    else:
        if support.stat().st_size>MAX_BYTES: errors.append('Support ZIP exceeds 20 MB')
        with zipfile.ZipFile(support) as z:
            if z.testzip(): errors.append('Support ZIP is corrupt')
            names=z.namelist()
            if len(names)!=len(set(names)): errors.append('Duplicate archive paths')
            if set(names)!=expected: errors.append('Archive differs from explicit support list')
            if sum(i.file_size for i in z.infolist())>100_000_000: errors.append('Archive exceeds inspection budget')
            else:
                for name in names:
                    parts=PurePosixPath(name).parts
                    if '..' in parts or name.startswith(('/','\\')) or ':' in name or '\\' in name:
                        errors.append('Unsafe archive path');continue
                    if any(x in parts for x in ['.git','.local','work','audit','demo','__pycache__','build']): errors.append('Private/temporary file in support: '+name)
                    data=z.read(name)
                    errors += [name+': '+f for f in scan_file(name,data,terms)]
                    local=local_path(root,name)
                    if not local.is_file() or digest(local)!=__import__('hashlib').sha256(data).hexdigest(): errors.append('Archive source changed: '+name)
                if cfg.get('ai_used') and 'AI工具使用详情.pdf' not in names: errors.append('AI details PDF missing')
    review=cfg.get('human_review',{})
    reviewed=all(review.get(k) is True for k in ['model_and_results','anonymity_including_images','layout_and_sources','ai_details'])
    current_hashes={'paper':digest(paper) if paper.is_file() else None,'support':digest(support) if support.is_file() else None}
    reviewed=reviewed and cfg.get('review_artifact_sha256')==current_hashes
    report={'technical_pass':not errors,'release_ready':not errors and reviewed,'errors':sorted(set(errors)),
            'human_review_recorded':reviewed,'candidate_sha256':current_hashes,'scope':'Paper and supporting material production only',
            'limits':['Text and metadata scan cannot recognize every unconfigured name or identity drawn in images.',
                      'A4 is automated; margins, scientific validity and visual layout require actual review.']}
    write_json(root/'records/precheck.json',report)
    return report

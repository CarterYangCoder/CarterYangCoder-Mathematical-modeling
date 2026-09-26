import hashlib
import json
import os
from pathlib import Path
import sys


def settings(root):
    cfg = json.loads((Path(root)/'project_config.json').read_text(encoding='utf-8'))
    def expand(value):
        return os.path.expandvars(value.replace('${USERPROFILE}', str(Path.home())))
    return {key: expand(value) if isinstance(value,str) else value for key,value in cfg.items()}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def local_path(root, relative):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes project: '+str(relative))
    return path

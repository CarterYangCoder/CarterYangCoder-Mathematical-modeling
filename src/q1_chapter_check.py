"""Check Q1 LaTeX/PDF against frozen numerical authority. Never solve the PDE."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
REC = ROOT / "records/q1-chapter-20260911"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed(value):
    return str(Decimal.from_float(float(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Render pages for subsequent visual review.")
    args = parser.parse_args()
    assets = json.loads((REC/"assets.json").read_text(encoding="utf-8"))
    assert assets["status"] == "CHAPTER_ASSETS_PASS"
    assert assets["source_sha256"] == sha(ROOT/"src/q1_chapter_assets.py")
    for rel, expected in assets["inputs_sha256"].items():
        assert sha(ROOT/rel) == expected, f"Changed frozen input: {rel}"
    for rel, expected in assets["output_files"].items():
        assert sha(ROOT/rel) == expected, f"Changed generated asset: {rel}"
    build = json.loads((PAPER/"build/q1-preview.build-report.json").read_text(encoding="utf-8"))
    assert build["success"] and not build["warnings"]
    # The shared helper also inventories non-input artifacts (including a previous
    # delivered PDF). Bind this check to the actual TeX/bibliography dependencies.
    compile_inputs = ["q1-preview.tex", "sections/q1.tex", "references.bib",
                      "generated/q1-numbers.tex", "generated/q1-table-temperature.tex",
                      "generated/q1-table-moisture.tex"]
    for rel in compile_inputs:
        expected = build["source_sha256"][rel]
        assert sha(PAPER/rel) == expected, f"PDF source is stale: {rel}"
    pdf = PAPER/"build/q1-preview.pdf"
    assert sha(pdf) == build["pdf_sha256"]

    reader = PdfReader(pdf)
    page_texts = [page.extract_text() for page in reader.pages]
    assert all(len(t.strip())>200 for t in page_texts), "Unexpected blank/near-blank page"
    assert sum(len(page.images) for page in reader.pages) == 0, "Unexpected embedded PDF image"
    assert all(abs(float(p.mediabox.width)-595.28)<1 and abs(float(p.mediabox.height)-841.89)<1 for p in reader.pages)
    fulltext = "\n".join(page_texts)
    compact = re.sub(r"\s+", "", fulltext)
    checked_rows = []
    times = [100,300,600,900,1200,1500,1800]
    columns = [0,5,10,15,20]
    with np.load(ROOT/"results/q1/q1-authority.npz", allow_pickle=False) as z:
        for key, basename, table_number in [
            ("T_C", "q1-table-temperature.tex", 1),
            ("C_kg_kg", "q1-table-moisture.tex", 2)]:
            source = (PAPER/"generated"/basename).read_text(encoding="utf-8")
            rows = re.findall(r"^(\d+)\s*&\s*([\d. &]+)\\\\", source, flags=re.M)
            assert len(rows) == 7
            for ti, (time_token, cells) in enumerate(rows):
                time = times[ti]
                expected = [fixed(z[key][time,j]) for j in columns]
                actual = [c.strip() for c in cells.split("&")]
                assert int(time_token) == time and actual == expected
                needle = str(time)+"".join(expected)
                assert needle in compact, f"PDF row missing or damaged: table {table_number}, {time}"
                checked_rows.append({"table":table_number, "time_s":time, "values":expected})
            assert f"表{table_number}:30分钟内药材的" in compact
    assert len(checked_rows)*5 == 70

    tex_paths = [PAPER/"sections/q1.tex", PAPER/"q1-preview.tex", *sorted((PAPER/"generated").glob("q1-*.tex"))]
    tex = "\n".join(p.read_text(encoding="utf-8") for p in tex_paths)
    uncommented = re.sub(r"(?m)%[^\n]*", "", tex)
    forbidden = [r"\\begin\{figure", r"\\includegraphics", r"\\captionof\{figure", r"\\ref\{fig:",
                 "如图所示", "待补图", "图片占位", "待填写", "待补充", "官方答案一致", "保证获奖"]
    assert not any(re.search(x, uncommented) for x in forbidden)
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", tex))
    assert refs <= labels
    citations = set(re.findall(r"\\cite\{([^}]+)\}", tex))
    bib = (PAPER/"references.bib").read_text(encoding="utf-8")
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
    assert citations == {"q1-dlmf", "q1-radau", "q1-fao", "q1-evaporation"}
    assert citations <= bibkeys
    assert len(re.findall(r"\\begin\{table\}", tex)) == 2
    assert len(re.findall(r"\\begin\{equation\}", tex)) == 16

    REC.mkdir(exist_ok=True)
    (REC/"pdf-text.txt").write_text("\n\n".join(f"PAGE {i+1}\n{text}" for i,text in enumerate(page_texts)),encoding="utf-8",newline="\n")
    if args.render:
        import pypdfium2
        out = REC/"render"
        out.mkdir(exist_ok=True)
        document = pypdfium2.PdfDocument(str(pdf))
        for i in range(len(document)):
            document[i].render(scale=1.5).to_pil().save(out/f"page-{i+1}.png")
        document.close()
    delivery = PAPER/"q1-chapter.pdf"
    shutil.copyfile(pdf, delivery)
    assert sha(delivery) == sha(pdf)
    status = {
        "status":"Q1_CHAPTER_AUTOMATED_CHECK_PASS",
        "time_utc":datetime.now(timezone.utc).isoformat(),
        "command":[sys.executable,*sys.argv], "source_sha256":sha(__file__),
        "numerical_inputs_unchanged":True, "PDE_solved":False,
        "paper_cells_checked_against_authority_and_PDF":70,
        "preceding_XLSX_and_evidence_checks":75600,
        "source_equations":16, "tables":2, "citations":sorted(citations),
        "pages":len(reader.pages), "embedded_images":0, "unresolved_references":0,
        "build_warnings":build["warnings"], "PDF_sha256":sha(delivery),
        "delivery":str(delivery.relative_to(ROOT)), "table_rows":checked_rows,
        "source_files_sha256":{str(p.relative_to(ROOT)):sha(p) for p in [*tex_paths,PAPER/"references.bib"]},
        "dependencies":{name:version(name) for name in ["numpy","pypdf","openpyxl","pypdfium2"]},
        "visual_review":"Not inferred by this script; recorded separately after actual page inspection.",
        "human_review_performed":False, "full_contest_submission_ready":False,
    }
    (REC/"chapter-check.json").write_text(json.dumps(status,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8",newline="\n")
    print(json.dumps({k:status[k] for k in ["status","pages","paper_cells_checked_against_authority_and_PDF","embedded_images","numerical_inputs_unchanged","PDF_sha256"]},ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Check the editable Q2 chapter and its PDF against frozen accepted outputs."""
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
REC = ROOT / "records/q2-chapter-20260911"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed(value):
    return str(Decimal.from_float(float(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    assets = json.loads((REC/"assets.json").read_text(encoding="utf-8"))
    assert assets["status"] == "Q2_CHAPTER_ASSETS_PASS"
    assert assets["source_sha256"] == sha(ROOT/"src/q2_chapter_assets.py")
    for rel, expected in assets["inputs_sha256"].items():
        assert sha(ROOT/rel) == expected, f"Frozen paper/numerical file changed: {rel}"
    for path, expected in assets["preserved_evidence_sha256"].items():
        assert sha(path) == expected, f"Accepted evidence changed: {path}"
    for rel, expected in assets["output_files"].items():
        assert sha(ROOT/rel) == expected, f"Changed generated asset: {rel}"
    build = json.loads((PAPER/"build/q2-preview.build-report.json").read_text(encoding="utf-8"))
    assert build["success"] and not build["warnings"], build["warnings"]
    dependencies = [
        "q2-preview.tex", "sections/q2.tex", "references.bib", "q2-references.bib",
        "generated/q2-numbers.tex", "generated/q2-table-temperature.tex", "generated/q2-table-moisture.tex",
    ]
    for rel in dependencies:
        assert sha(PAPER/rel) == build["source_sha256"][rel], f"Uncompiled source: {rel}"
    pdf = PAPER/"build/q2-preview.pdf"
    assert sha(pdf) == build["pdf_sha256"]
    reader = PdfReader(pdf)
    page_texts = [p.extract_text() for p in reader.pages]
    assert all(len(t.strip()) > 200 for t in page_texts), "Blank/near-blank page"
    images = sum(len(p.images) for p in reader.pages)
    assert images == 0
    assert all(abs(float(p.mediabox.width)-595.28)<1 and
               abs(float(p.mediabox.height)-841.89)<1 for p in reader.pages)
    fulltext = "\n".join(page_texts)
    compact = re.sub(r"\s+", "", fulltext)
    times = [1800, 3600, 5400, 7200, 9000, 10800]
    columns = [0,5,10,15,20]
    checked = []
    with np.load(ROOT/"results/q2/q2-authority.npz", allow_pickle=False) as z:
        for field, basename, number in [
            ("T_C","q2-table-temperature.tex",3),
            ("C_kg_kg","q2-table-moisture.tex",4),
        ]:
            source = (PAPER/"generated"/basename).read_text(encoding="utf-8")
            rows = re.findall(r"^([\d.]+)\s*&\s*([\d. &]+)\\\\", source, flags=re.M)
            assert len(rows) == 6
            for i, (hour, rawcells) in enumerate(rows):
                expected = [fixed(z[field][times[i],j]) for j in columns]
                actual = [v.strip() for v in rawcells.split("&")]
                assert Decimal(hour)*3600 == Decimal(times[i])
                assert actual == expected
                assert hour+"".join(expected) in compact, f"Broken PDF row {number}/{hour}"
                checked.append(dict(table=number, time_s=times[i], hour=hour, values=expected))
            assert f"表{number}:3小时内药材的" in compact
    texpaths = [PAPER/p for p in dependencies if p.endswith(".tex")]
    tex = "\n".join(p.read_text(encoding="utf-8") for p in texpaths)
    uncommented = re.sub(r"(?m)%[^\n]*", "", tex)
    forbidden = [r"\\begin\{figure", r"\\includegraphics", r"\\captionof\{figure", r"\\ref\{fig:",
                 "如图所示", "待补图", "图片占位", "待填写", "待补充", "官方答案一致", "保证获奖",
                 r"[A-Z]:[/\\]", "records/", "results/q2/", "src/q2"]
    assert not any(re.search(s, uncommented) for s in forbidden)
    labels_list = re.findall(r"\\label\{([^}]+)\}", tex)
    labels = set(labels_list)
    assert len(labels) == len(labels_list), "Duplicate label"
    refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", tex))
    assert refs <= labels, refs-labels
    citations = set()
    for item in re.findall(r"\\cite\{([^}]+)\}", tex):
        citations.update(item.split(","))
    bib = "\n".join((PAPER/p).read_text(encoding="utf-8") for p in ["references.bib","q2-references.bib"])
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
    assert citations == {"q1-radau", "q2-spatial-convergence"} and citations <= bibkeys
    assert len(re.findall(r"\\begin\{table\}", tex)) == 2
    assert len(re.findall(r"\\begin\{equation\}", tex)) == 17
    # Every chapter numeric macro is generated, traceable and actually compiled.
    used = set(re.findall(r"\\(QTwo[A-Za-z]+)", (PAPER/"sections/q2.tex").read_text(encoding="utf-8")))
    assert used <= set(assets["numeric_macros"])
    for name in used:
        rendered = assets["numeric_macros"][name]["rendered"]
        if r"\times" not in rendered:
            assert rendered in compact, f"Numeric macro absent from PDF: {name}"
    # Protect Q1 label namespace on eventual integration; preview counters are local.
    q1labels = set(re.findall(r"\\label\{([^}]+)\}", (PAPER/"sections/q1.tex").read_text(encoding="utf-8")))
    assert labels.isdisjoint(q1labels)
    fonts = set()
    for page in reader.pages:
        resources = page.get("/Resources")
        if resources:
            for ref in resources.get_object().get("/Font", {}).values():
                fonts.add(str(ref.get_object().get("/BaseFont", "")))
    (REC/"pdf-text.txt").write_text(
        "\n\n".join(f"PAGE {i+1}\n{text}" for i,text in enumerate(page_texts)),
        encoding="utf-8", newline="\n")
    if args.render:
        import pypdfium2
        out = REC/"render"
        out.mkdir(exist_ok=True)
        document = pypdfium2.PdfDocument(str(pdf))
        for i in range(len(document)):
            document[i].render(scale=1.5).to_pil().save(out/f"page-{i+1}.png")
        document.close()
    delivery = PAPER/"q2-chapter.pdf"
    shutil.copyfile(pdf, delivery)
    assert sha(delivery) == sha(pdf)
    status = dict(
        status="Q2_CHAPTER_AUTOMATED_CHECK_PASS",
        created_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable,*sys.argv], source_sha256=sha(__file__),
        PDE_solved=False, accepted_results_and_evidence_unchanged=True,
        checked_evidence_files=len(assets["preserved_evidence_sha256"]),
        paper_cells_checked_against_authority_and_PDF=60,
        preceding_XLSX_values_checked=assets["source_excel_values_checked"],
        pages=len(reader.pages), embedded_images=images, source_equations=17,
        tables=2, citations=sorted(citations), generated_macros_used=len(used),
        build_warnings=build["warnings"], unresolved_references=0,
        q1_label_collisions=0, PDF_sha256=sha(delivery), PDF_fonts=sorted(fonts),
        source_files_sha256={str((PAPER/p).relative_to(ROOT)):sha(PAPER/p) for p in dependencies},
        table_rows=checked, delivery=str(delivery.relative_to(ROOT)),
        dependencies={name:version(name) for name in ["numpy","pypdf","pypdfium2","openpyxl"]},
        visual_review="Separate actual page inspection required; not inferred by this script.",
        human_review_performed=False, entire_paper_ready=False)
    (REC/"chapter-check.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),
                                         encoding="utf-8",newline="\n")
    print(json.dumps({k:status[k] for k in [
        "status","pages","paper_cells_checked_against_authority_and_PDF","embedded_images",
        "build_warnings","accepted_results_and_evidence_unchanged","PDF_sha256"]},ensure_ascii=False))


if __name__ == "__main__":
    main()

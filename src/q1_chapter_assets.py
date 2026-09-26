"""Generate Q1 paper tables and numeric macros from frozen results; no PDE solve."""
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import openpyxl
from q1_verify_delivery import main as verify_delivery

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/generated"
RECORD = ROOT / "records/q1-chapter-20260911"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed(v, places=4):
    return str(Decimal.from_float(float(v)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def scientific(v, digits=5):
    a, b = f"{float(v):.{digits}e}".split("e")
    return a + r"\times10^{" + str(int(b)) + "}"


def main():
    verify_delivery()
    OUT.mkdir(exist_ok=True)
    RECORD.mkdir(exist_ok=True)
    authority = ROOT / "results/q1/q1-authority.npz"
    workbook = ROOT / "results/q1/result1.xlsx"
    inputs = [authority, workbook, ROOT/"config/q1-accepted.json",
              ROOT/"records/q1-reaudit-20260911/data-and-point-audit.json",
              ROOT/"records/q1-reaudit-20260911/paper-70-evidence.csv",
              ROOT/"records/q1-reaudit-20260911/compare-refinement.json",
              ROOT/"records/q1-full-20260911/final-candidate-analysis.json"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    with np.load(authority, allow_pickle=False) as z:
        ts, rr = z["time_s"].copy(), z["radius_m"].copy()
        fields = [z["T_C"].copy(), z["C_kg_kg"].copy()]
        budgets = [z["T_empirical_budget_C"].copy(), z["C_empirical_budget_kg_kg"].copy()]
    assert np.array_equal(ts, np.arange(1801)) and np.array_equal(rr, np.arange(21)*.001)
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    selected_times = [100, 300, 600, 900, 1200, 1500, 1800]
    selected_columns = [0, 5, 10, 15, 20]
    checked = []
    with inputs[4].open(encoding="utf-8-sig", newline="") as stream:
        evidence = list(csv.DictReader(stream))
    with (ROOT/"records/q1-reaudit-20260911/data-and-point-audit.json").open(encoding="utf-8") as stream:
        audit = json.load(stream)
    for k, (sheet, basename, caption, label) in enumerate([
        ("温度", "q1-table-temperature.tex", "30分钟内药材的温度（单位：℃）", "tab:q1-temperature"),
        ("水分浓度", "q1-table-moisture.tex", "30分钟内药材的水分浓度（干基，单位：kg/kg）", "tab:q1-moisture")]):
        grid = list(wb[sheet].iter_rows(values_only=True))
        with (ROOT/f"results/q1/table{k+1}.csv").open(encoding="utf-8-sig", newline="") as stream:
            old_table = list(csv.reader(stream))
        lines = ["% Generated from frozen q1-authority.npz; do not hand-edit numeric cells.",
                 "% SHA-256: " + before[str(authority.relative_to(ROOT))],
                 r"\begin{table}[!htbp]", r"\centering", r"\small",
                 r"\caption{"+caption+"}", r"\label{"+label+"}",
                 r"\setlength{\tabcolsep}{9pt}", r"\renewcommand{\arraystretch}{1.12}",
                 r"\begin{tabular*}{0.86\textwidth}{@{\extracolsep{\fill}}rccccc}", r"\toprule",
                 r"时间/s & \multicolumn{5}{c}{到药材中心的距离/cm} \\",
                 r"\cmidrule(lr){2-6}", r" & 0 & 0.5 & 1 & 1.5 & 2 \\", r"\midrule"]
        for row_i, t in enumerate(selected_times, 1):
            row = []
            for col_i, j in enumerate(selected_columns, 1):
                value = fixed(fields[k][t,j])
                assert Decimal(str(grid[t][j+1])) == Decimal(value)
                assert Decimal(old_table[row_i][col_i]) == Decimal(value)
                certificate = next(x for x in evidence if x["quantity"] == sheet and int(x["time_s"]) == t and abs(float(x["radius_cm"])-rr[j]*100)<1e-12)
                assert certificate["published_4dp"] == value
                row.append(value)
                checked.append({"field":sheet,"time_s":t,"radius_cm":float(rr[j]*100),"value":value})
            lines.append(str(t)+" & "+" & ".join(row)+r" \\")
        lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
        (OUT/basename).write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
    wb.close()
    m = audit["mechanistic_scales"]
    nums = {
        "QOneTCentre":fixed(fields[0][-1,0]), "QOneTSurface":fixed(fields[0][-1,-1]),
        "QOneTGap":fixed(fields[0][-1,-1]-fields[0][-1,0]),
        "QOneTEnvironment":fixed(m["final_Ta_degC"],3),
        "QOneCCentre":fixed(fields[1][-1,0]), "QOneCSurface":fixed(fields[1][-1,-1]),
        "QOneCCentreFine":fixed(fields[1][-1,0],9),
        "QOneCeEnd":fixed(m["final_Ce_effective_kg_kg"],5),
        "QOneAlpha":scientific(m["alpha_m2_s"]), "QOneDInitial":scientific(m["D_initial_m2_s"]),
        "QOneBiT":fixed(m["Bi_T_radius"],4),
        "QOneHeatTime":fixed(m["radial_thermal_diffusion_time_s"],1),
        "QOneMassTime":fixed(m["radial_initial_moisture_diffusion_time_s"],1),
        "QOneHeatLength":fixed(m["sqrt_alpha_t_at1800_m"]*1000,2),
        "QOneMassLength":fixed(m["sqrt_Dinitial_t_at1800_m"]*1000,2),
        "QOneDLastSurface":scientific(m["final_surface_D_m2_s"]),
        "QOneTEstimate":scientific(budgets[0][1:].max()),
        "QOneCEstimate":scientific(budgets[1][1:].max()),
        "QOneTIndependent":scientific(audit["fields"][0]["independent_method_maximum_difference"]["absolute_value"]),
        "QOneCIndependent":scientific(audit["fields"][1]["independent_method_maximum_difference"]["absolute_value"]),
    }
    (OUT/"q1-numbers.tex").write_text(
        "% Values derived from accepted Q1 authority and bound audit data.\n"+
        "\n".join("\\newcommand{\\"+key+"}{"+value+"}" for key,value in nums.items())+"\n",
        encoding="utf-8", newline="\n")
    for path, expected in before.items():
        assert sha(ROOT/path) == expected
    result = {"status":"CHAPTER_ASSETS_PASS", "created_utc":datetime.now(timezone.utc).isoformat(),
        "command":[sys.executable,*sys.argv], "source_sha256":sha(Path(__file__)),
        "inputs_sha256":before, "PDE_solved":False, "checked_paper_values":checked,
        "output_files":{str(p.relative_to(ROOT)):sha(p) for p in OUT.glob("q1-*.tex")},
        "input_and_accepted_results_unchanged":True}
    (RECORD/"assets.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8",newline="\n")
    print(json.dumps({"status":result["status"],"checked_paper_values":len(checked),"PDE_solved":False},ensure_ascii=False))


if __name__ == "__main__":
    main()

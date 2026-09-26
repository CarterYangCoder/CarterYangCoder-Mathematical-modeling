"""Summarize the completed Q1 re-audit without changing accepted results."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "records/q1-reaudit-20260911"
REPRO = ROOT / "records/q1-full-20260911/reproductions/20260911-041932-830558"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    replay = load(REPRO/"run.json")
    audit = load(OUT/"data-and-point-audit.json")
    refine = load(OUT/"compare-refinement.json")
    assert replay["status"] == "REPRODUCTION_COMPARISON_PASS"
    assert audit["status"] == "READBACK_AND_POINT_CERTIFICATE_PASS"
    assert refine["status"] == "ADDITIONAL_REFINEMENT_REVIEW_COMPLETE"
    assert all(count == 0 for count in refine["rounding"]["candidate_to_each_grid_mismatch_count"].values())
    authority = ROOT/"results/q1/q1-authority.npz"
    assert digest(authority) == replay["authority_sha256"] == audit["authority_sha256"]
    assert digest(ROOT/"results/q1/result1.xlsx") == audit["excel_sha256"]
    with np.load(authority, allow_pickle=False) as old, np.load(REPRO/"reproduced.npz", allow_pickle=False) as new:
        repeats = {}
        for key in ["T_C", "C_kg_kg"]:
            difference = new[key]-old[key]
            i, j = np.unravel_index(np.argmax(abs(difference)), difference.shape)
            repeats[key] = {"maximum_absolute_difference": float(abs(difference[i,j])),
                "time_s": float(old["time_s"][i]), "radius_cm": float(old["radius_m"][j]*100)}
    status = {"status": "Q1_REAUDIT_COMPLETE_NO_NUMERICAL_CORRECTION_REQUIRED",
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
        "full_replay_completed": True, "full_replay_elapsed_s": replay["elapsed_s"],
        "full_replay_maximum_differences": repeats, "full_replay_rounding_mismatches": 0,
        "additional_65536_from_zero_completed": True,
        "additional_65536_elapsed_s": refine["run_settings"]["elapsed_s"],
        "additional_65536_to_authority_maximum": refine["candidate_to_new_reference"],
        "additional_16k_32k_64k_maxnorm_order": refine["spatial_refinement"]["maximum_norm_order"],
        "checked_original_input_files": len(audit["original_input_checks"]),
        "checked_export_values": sum(f["checked_excel_values"] for f in audit["fields"]),
        "checked_paper_values": sum(f["checked_paper_values"] for f in audit["fields"]),
        "unresolved_4dp_under_frozen_empirical_budget": sum(f["unresolved_4dp_under_declared_budget"] for f in audit["fields"]),
        "weak_rounding_positions": [{"quantity": "C", "time_s": 1696, "radius_cm": 1.1,
            "authority": 2.532149997413533, "additional_65536": 2.5321499976953858,
            "rounded": "2.5321", "original_margin_over_budget": 1.079381283888725,
            "warning": "Support strengthened by another refinement; original empirical allowance unchanged, no strict bound."}],
        "accepted_authority_unchanged": True, "accepted_budget_unchanged": True,
        "original_inputs_unchanged": True, "formal_result_still_model_conditional": True,
        "physical_accuracy_quantified": False, "official_reference_answer_comparison_performed": False,
        "human_technical_review_performed": False, "result_figures_created": 0,
        "source_evidence_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [
            Path(__file__), ROOT/"src/q1_reaudit.py", ROOT/"src/q1_reaudit_refinement.py",
            REPRO/"run.json", REPRO/"reproduced.npz", OUT/"data-and-point-audit.json",
            OUT/"paper-70-evidence.csv", OUT/"physics-review.md", OUT/"physics-review.json",
            OUT/"numerics-review.md", OUT/"numerics-review.json", OUT/"numerics_review.py",
            OUT/"compare-refinement.py", OUT/"compare-refinement.json", OUT/"compare-refinement.md",
            OUT/"BDF-N65536/run.json", OUT/"BDF-N65536/solution.npz",
            OUT/"论证与来源.md", OUT/"图表制作指示.md", authority, ROOT/"results/q1/result1.xlsx"]}}
    with (OUT/"status.json").open("x", encoding="utf-8") as stream:
        json.dump(status, stream, ensure_ascii=False, indent=2, allow_nan=False)
    report = f"""# 问题1再次复算与审查结果

本轮按用户要求重新读取原始题目、附录、环境和模板，完整复算并对最敏感末位增加一层空间核验。**没有发现需要修改现有答案的确定性数值错误。当前结论仍限于已批准的R1有效模型，不能替代真实药材测量或未取得的官方参考答案。**

## 本轮实际完成

- 同配置从t=0重算到1800 s，耗时{replay['elapsed_s']:.3f} s。温度最大复现差{repeats['T_C']['maximum_absolute_difference']:.16g} ℃（1742 s，0.6 cm），水分{repeats['C_kg_kg']['maximum_absolute_difference']:.16g} kg/kg（339 s，1.8 cm）；75,600个四位值一致。
- 追加65536环BDF从原始初值到1800 s，耗时{refine['run_settings']['elapsed_s']:.3f} s。保留同时间控制、原边界与局部D，不从粗网格状态起算。与正式32768环Radau最大差1.1136459177052416e−7 kg/kg（1 s，表面）；全部37,800个水分值及35个论文水分值的四位舍入均一致。
- 原7份文件指纹不变，附件1全部241条记录无缺失/重复/不规则时间；问题1使用31个真实节点。最终两工作表75,600值、两论文表70值再次逐项回读一致；没有重建或修改正式Excel。
- 新增70项逐值证据表及机理论证、图表指示；没有生成结果图。原题第3页的渲染是原件核查副本，不是仿真图。

## 数值精度的准确口径

| 指标 | 温度/℃ | 水分/(kg/kg) |
|---|---:|---:|
| 正式全精度解的最大数值预算 | 2.383075783943827e−8 | 1.5046286200409508e−7 |
| 两张论文表各35项中的最大预算 | 2.383075783943827e−8 | 1.2993285245082992e−8 |
| 与不同空间实现的最大差 | 3.068920335635994e−7 | 2.442598070473423e−7 |

水分最大预算与独立差均在1 s表面；温度独立差在1800 s、r=1.9 cm。温度最大预算来自统一的中心尾界，不能把预算最大位置解释为真实误差最大位置。这里的数值预算不含物理/输入不确定性；温度解析尾界只覆盖模态截断，其算术余量仍是经验值。独立方法差包含参照自身误差。

水分16k→32k网格差4.4515162755942583e−7、32k→64k差1.1128560561957102e−7，最大范数阶2.000029833。该新证据支持原32768解及其预算；没有改用65536层作权威值，也没有缩小原2e−9经验余量。

四位小数在原预算下**未确认0项**，但1696 s、r=1.1 cm的C仍是小裕度点：原值2.532149997413533，新65536参照2.5321499976953858，均舍入2.5321。原界距/预算仅1.07938；把经验余量提高10%的压力试验会使此项失去原预算支持。新层增加了对舍入方向的证据，并未产生严格数学保证。不能将“未确认0项”写成“每个末位绝对保证”。

四位舍入自身最多改变约5e−5；Excel保留的是题定展示精度，不能声称它保留了全精度解的1e−7或1e−8误差尺度。同配置重算到机器精度、守恒小残差、多方法接近、严格模态尾界和物理真实性是不同层次。

## 机理、模型适配与创新

现有R1在题给参数和已声明闭合下成立：保留圆柱径向几何、有限Robin阻力、局部非线性D以及非稳态环境。热扩散时间2368.9 s、水分初始扩散时间81010.1 s，解释1800 s时内部已明显升温而中心水分几乎不变。1800 s的中心/表面温度33.5753/36.7856 ℃，中心/表面C为2.5500/1.5102；中心C实际为2.549992404139774。

原题没有证明烘房列等于药材平衡含水率，也没有量化忽略蒸发冷却的误差；一维中部截面亦是有尺度依据的近似。未取得官方数值基准，不能验证与标准答案一致。若参考采用同一方程、初边值和插值，数值差应由双方求解误差解释；不同闭合可能产生模型差异。

可写的特点是“半解析传热与非线性守恒传质组合，独立表面状态，启动误差定位和输出末位验证”。成熟算法不能包装为原创物理或双向强耦合。国赛合理性、创造性、正确性和表达均需证据，单凭问题1不能认定获奖等级。具体来源和论证见《论证与来源.md》及physics-review.md。

## 成果与复现

- 正式结果维持 `results/q1/result1.xlsx`、`table1/2.csv/.md/.tex` 和 `q1-authority.npz`，不覆盖原附件。
- 每个论文数值：`paper-70-evidence.csv`；新增网格比较：`compare-refinement.json/md`。
- 图表只给《图表制作指示.md》：两个结果组合图、一个验证组合图和两张题定表；源数据/坐标/单位/禁用误导方式均已明确。
- 只读核验入口 `src/q1_verify_delivery.py`；本轮实际完整复算入口 `src/q1_reproduce.py --run`。完整新状态位于 `records/q1-full-20260911/reproductions/20260911-041932-830558`。原README中的包装器未执行说明为前次历史，本轮已补上端到端实际执行证据。
- `src/q1_reaudit.py` 为本轮数据与逐值检查，`src/q1_reaudit_refinement.py` 为本轮追加空间参照；新输出目录拒绝覆盖。再次执行时应使用新的版本目录，不删除已有证据来重复利用同名路径。
- 本轮实际读取技能及软件来源（包括已核实的Scientific Agent Skills来源）记录于《论证与来源.md》。本次AI复核不能冒称队员人工技术审核；问题1之后暂停，不扩展其他小问或提交。
"""
    with (OUT/"本轮复核结论.md").open("x", encoding="utf-8") as stream:
        stream.write(report)
    agent = ROOT/"AGENT.md"
    text = agent.read_text(encoding="utf-8")
    marker = "## D007：用户要求再次全量复算与物理审查"
    assert marker not in text
    insert = f"""{marker}（2026-09-11，已完成）

- 用户明确要求再次核对所有数据、验算、解释精度与机理，并只给图表指示。本轮完整复算及审查完成，继续暂停在问题1，不扩展或提交。
- q1_reproduce.py --run本次已实际从0至1800 s运行{replay['elapsed_s']:.3f} s；T/C最大复现差1.4210854715202004e-14 ℃、8.881784197001252e-16 kg/kg，75600值舍入一致。原“该包装器未运行”是D006历史，本条为当前状态。
- 为1696 s/r1.1 cm小裕度尾数增加65536环BDF完整从0参照，耗时{refine['run_settings']['elapsed_s']:.3f} s；32→64层最大差1.1128560561957102e-7，16/32/64最大范数阶2.000029833；原Radau32到新参照最大差1.1136459177052416e-7（1 s表面），37800水分舍入差0。该点新值2.5321499976953858，仍2.5321；原预算不缩小、权威值不替换，仍无严格全局界。
- 原7文件、241条环境及Q1的31节点重新核对；75600导出和70论文值再读一致。新增records/q1-reaudit-20260911/paper-70-evidence.csv为每个论文值保留数值依据。
- 机理复审无新增题给参数冲突；等效Ce与省略蒸发冷却仍属未量化物理假设，没有官方参考值/内部实测可验证。不得称物理完全正确、双向强耦合或保证标准答案/国奖。创新表述为成熟方法的组合适配及数值验证特点。
- 新记录records/q1-reaudit-20260911/status.json、本轮复核结论.md、论证与来源.md、图表制作指示.md；未画结果图，未冒充人工审核。正式结果保持原authority SHA 8aab6a875752f60bb798519d60e2532a54585874b92e18183adb861571d320a4。

"""
    text = text.replace("## 当前阶段（D006完整验收已完成，2026-09-11）", insert+"## D006完整验收（历史定稿，正式数值仍生效）", 1)
    text = text.replace("## 原题公式（已核实转录；仅Q1完成短时试算）", "## 原题公式（已核实转录；Q1已完成全时段验收及再次复算）")
    agent.write_text(text, encoding="utf-8")
    readme = ROOT/"records/q1-full-20260911/README.md"
    old = readme.read_text(encoding="utf-8")
    update = "\n> 2026-09-11后续复审更新：本页所述q1_reproduce.py --run已在用户再次核验授权下实际运行，419.098 s从0至1800 s复算通过，75600舍入一致。新证据位于reproductions/20260911-041932-830558/run.json及records/q1-reaudit-20260911。下文未执行包装器的文字属于此前D006阶段历史，不能作为当前状态。正式结果与原验收预算未改。\n"
    assert update not in old
    readme.write_text(old.replace("# 问题1计算、核验与复现\n", "# 问题1计算、核验与复现\n"+update, 1), encoding="utf-8")
    with (ROOT/"records/AI使用日志.csv").open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(["2026-09-11", "Q1-reaudit-20260911", "Codex主代理及两个AI复核子代理；GPT-6系列，精确版本未知",
            "再次核对原件、完整复算、物理与创新审查、图表指示", "用户要求不编造且解释每个答案；读取grilling、paper-methods/quality/ab-validation及critical-thinking/visualization；检索国赛官方、FAO、COMSOL、NIST与技能来源",
            "完整32768Radau与1024热模态从0复算；另65536BDF从0参照；70项证据与75600回读；权威/预算/原件不变；新增审查脚本、报告、AGENT；未画结果图",
            "用户已授权复核；AI审查不等于队员人工审核；物理误差与标准答案未验证；无获奖保证；不扩展其他小问"])
    with (ROOT/"records/逐问验证.csv").open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(["Q1-再次全量复核", "原正式值维持；复算与增补空间参照通过", "records/q1-reaudit-20260911/status.json；paper-70-evidence.csv",
            "冻结32768Radau/1024模态及原独立参照", "R1有效Ce和显热近似不变；所有参数/单位不变",
            "0—1800同设置复算；65536BDF空间加密；原件/75600导出/70纸表逐值审计；物理与创新来源审查",
            "原数值目标/预算不变；全部四位舍入保持；不声称严格物理精度",
            "T复现差1.42109e-14℃；C复现差8.88178e-16；主解对64k最大差1.1136459177052416e-7(1s,2cm)；75600舍入差0",
            "1696s/1.1cm小裕度仍需限定；65536无同网格Radau对照；Ce/潜热物理偏差未量化；无标准答案",
            "主AI与两个AI子代理；无队员人工核验", "问题1复核完成并暂停"])
    print(json.dumps({k: status[k] for k in ["status", "full_replay_completed", "additional_65536_from_zero_completed", "checked_export_values", "checked_paper_values", "unresolved_4dp_under_frozen_empirical_budget"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

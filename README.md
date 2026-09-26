# 2026 全国大学生数学建模竞赛 A 题项目

本仓库收录项目源码、图表、关键计算数据及完整论文。文件按用户提供的 `支撑材料/` 与 `最终材料/` 整理。

## 内容

- `src/`、`support/source/`、`tools/`：模型、求解、绘图、结果导出及项目工具源码。`support/source/` 保留支撑材料目录中的 Q1–Q4 最终源文件。
- `config/`：模型参数和运行/输出配置。
- `results/`：Q1/Q2 全精度 NPZ 数据与论文结果表。
- `support/data/environment-scenarios.csv`：环境情景数据。
- `support/results/result1.xlsx` 至 `result4.xlsx`：四问结果工作簿。
- `figures/`：最终材料目录中的 Q1–Q4 图表及灵敏度/支撑图。
- `paper/论文.pdf`：完整四问论文 PDF。`paper/sections/`、`paper/generated/` 与两个章节预览保留可编辑的 Q1/Q2 LaTeX 章节材料。
- `requirements-lock.txt`：依赖版本记录。

## 范围和说明

原题 PDF、原始附件、身份照片、提交截图、MD5 截图和临时/历史文件未纳入仓库。Q3/Q4 源码、结果和图表按用户指定目录收录；此次 GitHub 整理没有重新运行计算或替代原项目的技术验收。

AI 工具使用报告暂未纳入：该 PDF 含个人/人工审核声明，而项目当前 `submission-manifest.json` 未记录人工复核完成。待该声明与真实复核记录核对一致后再添加。

Q1/Q2 的计算结果及误差适用范围见相应章节与 `results/README.md`。模型计算值不等于材料内部实测值；上传论文和文件不构成额外的科学复核或正式提交状态声明。

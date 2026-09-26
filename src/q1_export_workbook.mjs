// Template import, bounded previews and export for Q1. No PDE calculation here.
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const args = {};
for (let i = 2; i < process.argv.length; i += 2) args[process.argv[i].replace(/^--/, "")] = process.argv[i + 1];
if (!args.runtime || !args.template || !args.preview) throw new Error("runtime, template and preview are required");
const requireRuntime = createRequire(path.join(path.resolve(args.runtime), "runtime-entry.js"));
const { FileBlob, SpreadsheetFile } = await import(pathToFileURL(requireRuntime.resolve("@oai/artifact-tool")).href);
await fs.mkdir(args.preview, { recursive: true });
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(args.template));
const names = ["温度", "水分浓度"];
const a1 = names.map(name => workbook.worksheets.getItem(name).getRange("A1").values[0][0]);
if (a1.some(value => value !== "时间\\到药材中心的距离")) throw new Error("Unexpected A1 text");
const summary = await workbook.inspect({ kind: "workbook,sheet,table", maxChars: 4500, tableMaxRows: 5, tableMaxCols: 6 });
await fs.writeFile(path.join(args.preview, "template-inspection.ndjson"), summary.ndjson);

if (args.payload) {
  if (!args.output) throw new Error("output required for payload");
  const payload = JSON.parse(await fs.readFile(args.payload, "utf8"));
  if (payload.times.length !== 1800 || payload.radius_cm.length !== 21) throw new Error("Bad axis shape");
  for (let i = 0; i < names.length; i++) {
    const sheet = workbook.worksheets.getItem(names[i]);
    const matrix = payload.values[i];
    if (matrix.length !== 1800 || matrix.some(row => row.length !== 21 || row.some(x => !Number.isFinite(x)))) throw new Error("Bad field matrix");
    // Extend the nearby template style and dimensions; preserve the A1 content.
    sheet.getRange("A2:A1801").copyFrom(sheet.getRange("A2"), "all");
    sheet.getRange("B1:V1").copyFrom(sheet.getRange("B1"), "all");
    sheet.getRange("B2:V1801").copyFrom(sheet.getRange("B2"), "all");
    sheet.getRange("A2:A1801").values = payload.times.map(t => [t]);
    sheet.getRange("B1:V1").values = [payload.radius_cm];
    sheet.getRange("B2:V1801").values = matrix;
    sheet.getRange("A2:A1801").setNumberFormat("0");
    sheet.getRange("B1:V1").setNumberFormat("0.0");
    sheet.getRange("B2:V1801").setNumberFormat("0.0000");
    sheet.getRange("B1:V1801").format.columnWidth = 9.625;
    sheet.getRange("A2:V1801").format.rowHeight = 14.1;
    if (sheet.getRange("A1").values[0][0] !== a1[i]) throw new Error("A1 unexpectedly changed");
  }
  workbook.recalculate(); // The task writes typed values; the template has no formulas.
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 20 }, summary: "Q1 export error scan" });
  await fs.writeFile(path.join(args.preview, "formula-error-scan.ndjson"), errors.ndjson);
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(args.output);
}
for (let i = 0; i < names.length; i++) {
  const range = args.payload ? "A1:V8" : "A1:F5";
  const blob = await workbook.render({ sheetName: names[i], range, scale: 1.5, format: "png" });
  await fs.writeFile(path.join(args.preview, args.payload ? `result-sheet-${i + 1}.png` : `template-sheet-${i + 1}.png`), new Uint8Array(await blob.arrayBuffer()));
  if (args.payload) {
    const tail = await workbook.render({ sheetName: names[i], range: "A1795:V1801", scale: 1.5, format: "png" });
    await fs.writeFile(path.join(args.preview, `result-tail-${i + 1}.png`), new Uint8Array(await tail.arrayBuffer()));
  }
}
console.log(JSON.stringify({ template: args.template, output: args.output || null, preview: args.preview, exported: !!args.payload, a1 }));

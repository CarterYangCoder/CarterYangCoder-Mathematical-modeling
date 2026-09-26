// Read the final XLSX for previews; no workbook value/style writes or export.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const root=process.cwd();
const runtime=path.join(root,'records/q1-full-20260911/artifact-preview');
const require=createRequire(path.join(runtime,'runtime-entry.js'));
const {FileBlob,SpreadsheetFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const file=path.join(root,'results/q1/result1.xlsx');
const workbook=await SpreadsheetFile.importXlsx(await FileBlob.load(file));
const out=path.join(root,'records/q1-full-20260911/export-v2/final-preview');
await fs.mkdir(out,{recursive:true});
for(const [index,name] of ['温度','水分浓度'].entries()){
  for(const [part,range] of [['head','A1:V8'],['tail','A1795:V1801']]){
    const image=await workbook.render({sheetName:name,range,scale:1.5,format:'png'});
    await fs.writeFile(path.join(out,`${part}-${index+1}.png`),new Uint8Array(await image.arrayBuffer()));
  }
}
console.log(JSON.stringify({source:file,preview:out,read_only:true}));

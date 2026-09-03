import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const releaseRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const releaseDir = path.resolve(process.env.CGC_TABLE_OUTPUT_DIR || releaseRoot);
const sourceDir = path.join(releaseRoot, "supplementary_table_sources");
const previewDir = path.resolve(process.env.CGC_TABLE_PREVIEW_DIR || path.join(releaseDir, "table_previews"));
await fs.mkdir(releaseDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const configs = [
  {
    csv: "Supplementary_Table_1_Datasets_and_Accessions.csv",
    xlsx: "Supplementary_Table_1_Datasets_and_Accessions.xlsx",
    title: "Supplementary Table 1 | Datasets, public accessions and analyzed representations",
    widths: [230, 165, 110, 125, 240, 150, 300, 170, 250, 300],
    wrap: true,
    rowHeight: 105,
  },
  {
    csv: "Supplementary_Table_2_Information_Sets_and_Splits.csv",
    xlsx: "Supplementary_Table_2_Information_Sets_and_Splits.xlsx",
    title: "Supplementary Table 2 | Prediction tasks, inference-time information and held-out splits",
    widths: [220, 245, 150, 170, 310, 260, 180, 270, 210, 190],
    wrap: true,
    rowHeight: 115,
  },
  {
    csv: "Supplementary_Table_3_Statistics_and_Inference.csv",
    xlsx: "Supplementary_Table_3_Statistics_and_Inference.xlsx",
    title: "Supplementary Table 3 | Statistical estimands, resampling, nulls and multiplicity",
    widths: [210, 245, 210, 210, 150, 235, 235, 245, 310],
    wrap: true,
    rowHeight: 115,
  },
  {
    csv: "Supplementary_Table_4_PROGENy_Weights.csv",
    xlsx: "Supplementary_Table_4_PROGENy_Weights.xlsx",
    title: "Supplementary Table 4 | Fixed PROGENy pathway definitions and gene weights",
    widths: [120, 150, 120, 105, 280, 180, 105, 175],
    wrap: false,
  },
  {
    csv: "Supplementary_Table_5_Experimental_Compression_Grid.csv",
    xlsx: "Supplementary_Table_5_Experimental_Compression_Grid.xlsx",
    title: "Supplementary Table 5 | Experimental Compression support grid and evaluated budgets",
    widths: [150, 175, 145, 140, 105, 235, 155, 210],
    wrap: true,
    rowHeight: 26,
    formulas: true,
  },
  {
    csv: "Supplementary_Table_6_Multimodal_Panel_Inventory.csv",
    xlsx: "Supplementary_Table_6_Multimodal_Panel_Inventory.xlsx",
    title: "Supplementary Table 6 | Mechanism-aligned multimodal panel inventory",
    widths: [190, 290, 175, 125, 145, 125, 230, 300, 235, 225, 310],
    wrap: true,
    rowHeight: 130,
  },
];

function columnName(index) {
  let result = "";
  let value = index + 1;
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

function countRows(csvText) {
  const trimmed = csvText.trimEnd();
  if (!trimmed) return 0;
  return trimmed.split(/\r?\n/).length - 1;
}

function csvRecords(text) {
  const records = []; let row = [], cell = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      if (quoted && text[i + 1] === '"') { cell += '"'; i++; }
      else quoted = !quoted;
    } else if (ch === ',' && !quoted) { row.push(cell); cell = ""; }
    else if (ch === '\n' && !quoted) { row.push(cell.replace(/\r$/, "")); records.push(row); row = []; cell = ""; }
    else cell += ch;
  }
  if (cell || row.length) { row.push(cell); records.push(row); }
  return records;
}

function wrappedLines(value, width) {
  const capacity = Math.max(8, Math.floor((width - 12) / 5.6));
  let lines = 1, used = 0;
  for (const word of String(value).split(/\s+/)) {
    if (word.length > capacity) {
      if (used) { lines++; used = 0; }
      lines += Math.floor(word.length / capacity); used = word.length % capacity;
    } else if (used + word.length + 1 > capacity) { lines++; used = word.length; }
    else used += word.length + 1;
  }
  return lines;
}

async function buildOne(config) {
  const csvPath = path.join(sourceDir, config.csv);
  const csvText = await fs.readFile(csvPath, "utf8");
  const rowCount = countRows(csvText);
  const headers = csvText.slice(0, csvText.indexOf("\n")).replace(/\r$/, "").split(",");
  const lastColumn = columnName(headers.length - 1);
  while (config.widths.length < headers.length) config.widths.push(310);
  const lastRow = rowCount + 1;
  const isLarge = rowCount > 100000;

  process.stderr.write(`[${config.xlsx}] source loaded (${rowCount} rows)\n`);

  if (isLarge) {
    const previewCsv = csvText.split(/\r?\n/).slice(0, 13).join("\n") + "\n";
    const previewWorkbook = await Workbook.fromCSV(previewCsv, { sheetName: "Data" });
    const previewSheet = previewWorkbook.worksheets.getItem("Data");
    previewSheet.showGridLines = false;
    previewSheet.getRange(`A1:${lastColumn}1`).format = {
      fill: "#0B5563",
      font: { name: "Aptos", bold: true, color: "#FFFFFF", size: 10 },
      wrapText: true,
      rowHeightPx: 40,
    };
    config.widths.forEach((width, index) => {
      previewSheet.getRange(`${columnName(index)}1`).format.columnWidthPx = width;
    });
    previewSheet.getRange("C2:C12").format.numberFormat = "0.0000000000";
    const previewInspect = await previewWorkbook.inspect({
      kind: "sheet,table",
      include: "id,name,text",
      maxChars: 4000,
    });
    await fs.writeFile(
      path.join(previewDir, `${path.parse(config.xlsx).name}_inspect.ndjson`),
      previewInspect.ndjson + "\n",
      "utf8",
    );
    const previewBlob = await previewWorkbook.render({
      sheetName: "Data",
      range: `A1:${lastColumn}12`,
      format: "png",
      scale: 1,
    });
    await fs.writeFile(
      path.join(previewDir, `${path.parse(config.xlsx).name}.png`),
      new Uint8Array(await previewBlob.arrayBuffer()),
    );
    process.stderr.write(`[${config.xlsx}] bounded preview verified\n`);
  }

  const workbook = await Workbook.fromCSV(csvText, { sheetName: "Data" });
  process.stderr.write(`[${config.xlsx}] workbook imported\n`);
  workbook.setColorScheme({
    name: "CGC Publication",
    themeColors: {
      accent1: "#0B5563",
      accent2: "#C66B3D",
      bg1: "#FFFFFF",
      tx1: "#17212B",
    },
  });

  const dataSheet = workbook.worksheets.getItem("Data");
  dataSheet.showGridLines = false;
  dataSheet.freezePanes.freezeRows(1);
  dataSheet.getRange(`A1:${lastColumn}1`).format = {
    fill: "#0B5563",
    font: { name: "Aptos", bold: true, color: "#FFFFFF", size: 10 },
    horizontalAlignment: "left",
    verticalAlignment: "middle",
    wrapText: true,
    rowHeightPx: 40,
    borders: { bottom: { style: "medium", color: "#073B44" } },
  };

  if (rowCount > 0 && !isLarge) {
    dataSheet.getRange(`A2:${lastColumn}${lastRow}`).format = {
      font: { name: "Aptos", color: "#17212B", size: 10 },
      verticalAlignment: "top",
      wrapText: config.wrap,
      rowHeightPx: config.wrap ? (config.rowHeight ?? 48) : 19,
    };
  }
  config.widths.forEach((width, index) => {
    const column = columnName(index);
    dataSheet.getRange(`${column}1`).format.columnWidthPx = width;
  });

  if (config.wrap && !isLarge) {
    const records = csvRecords(csvText);
    records.slice(1).forEach((record, index) => {
      const lines = Math.max(...record.map((value, col) => wrappedLines(value, config.widths[col])));
      dataSheet.getRange(`A${index+2}:${lastColumn}${index+2}`).format.rowHeightPx = Math.max(config.rowHeight || 48, lines * 18 + 24);
    });
  }

  if (rowCount <= 1000) {
    for (let row = 2; row <= lastRow; row += 2) {
      dataSheet.getRange(`A${row}:${lastColumn}${row}`).format.fill = "#F2F7F8";
    }
  }

  const notes = workbook.worksheets.add("Notes");
  notes.showGridLines = false;
  notes.getRange("A1:D1").merge();
  notes.getRange("A1").values = [[config.title]];
  notes.getRange("A1:D1").format = {
    fill: "#0B5563",
    font: { name: "Aptos Display", bold: true, color: "#FFFFFF", size: 16 },
    verticalAlignment: "middle",
    rowHeightPx: 34,
  };
  notes.getRange("A3:B6").values = [
    ["Workbook", config.xlsx],
    ["Data rows", rowCount],
    ["Source table", `supplementary_table_sources/${config.csv}`],
    ["Scope", "Final manuscript-facing evidence only; no model refitting"],
  ];
  notes.getRange("A3:A6").format = {
    fill: "#DCECEF",
    font: { name: "Aptos", bold: true, color: "#073B44", size: 10 },
  };
  notes.getRange("B3:B6").format = {
    font: { name: "Aptos", color: "#17212B", size: 10 },
    wrapText: true,
  };
  notes.getRange("A1:A10").format.columnWidthPx = 250;
  notes.getRange("B1:B10").format.columnWidthPx = 540;
  notes.getRange("C1:D10").format.columnWidthPx = 160;

  if (config.formulas && rowCount > 0) {
    notes.getRange("A8:C10").values = [
      ["Formula parameter", "Value", "Meaning"],
      ["Interventions per complete reference context", 93, "Complete intervention axis"],
      ["Complete response matrix entries", 4650, "50 contexts × 93 interventions"],
    ];
    notes.getRange("A8:C8").format = {
      fill: "#C66B3D",
      font: { name: "Aptos", bold: true, color: "#FFFFFF", size: 10 },
    };
    dataSheet.getRange("C2").formulas = [["=Notes!$B$9*A2+B2"]];
    dataSheet.getRange(`C2:C${lastRow}`).fillDown();
    dataSheet.getRange("D2").formulas = [["=C2/Notes!$B$10"]];
    dataSheet.getRange(`D2:D${lastRow}`).fillDown();
  }

  if (config.csv.includes("PROGENy")) {
    dataSheet.getRange("C2:C50").format.numberFormat = "0.0000000000";
    dataSheet.getRange("H2:H50").format.horizontalAlignment = "center";
  }
  if (config.csv.includes("Experimental_Compression")) {
    dataSheet.getRange(`A2:C${lastRow}`).format.numberFormat = "0";
    dataSheet.getRange(`D2:D${lastRow}`).format.numberFormat = "0.00%";
    dataSheet.getRange(`G2:H${lastRow}`).format.numberFormat = "0.0000";
  }
  if (config.csv.includes("Multimodal")) {
    dataSheet.getRange(`D2:F${lastRow}`).format.numberFormat = "#,##0";
  }

  if (config.formulas) {
    const formulaErrors = ["#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"];
    for (const term of formulaErrors) {
      const matches = workbook.findCells({
        searchTerm: term,
        sheetId: "Data",
        options: { maxResults: 5, matchFormulas: true },
      });
      if (matches.totalCount && matches.totalCount > 0) {
        throw new Error(`${config.xlsx}: formula error ${term}`);
      }
    }
  }

  if (!isLarge) {
    const inspect = await workbook.inspect({
      kind: "sheet,formula",
      include: "id,name,text",
      maxChars: 4000,
    });
    await fs.writeFile(
      path.join(previewDir, `${path.parse(config.xlsx).name}_inspect.ndjson`),
      inspect.ndjson + "\n",
      "utf8",
    );

    const preview = await workbook.render({
      sheetName: "Data",
      range: `A1:${lastColumn}${Math.min(lastRow, 12)}`,
      format: "png",
      scale: 1,
    });
    await fs.writeFile(
      path.join(previewDir, `${path.parse(config.xlsx).name}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
    if (rowCount > 12 && rowCount < 1000) {
      const tail = await workbook.render({
        sheetName: "Data", range: `A${Math.max(2,lastRow-4)}:${lastColumn}${lastRow}`,
        format: "png", scale: 1,
      });
      await fs.writeFile(path.join(previewDir, `${path.parse(config.xlsx).name}_tail.png`), new Uint8Array(await tail.arrayBuffer()));
    }
  }

  process.stderr.write(`[${config.xlsx}] exporting XLSX\n`);
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  const outputPath = path.join(releaseDir, config.xlsx);
  await xlsx.save(outputPath);
  const sidecar = `${outputPath}.inspect.ndjson`;
  try {
    await fs.rename(sidecar, path.join(previewDir, `${path.parse(config.xlsx).name}_export_inspect.ndjson`));
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  process.stderr.write(`[${config.xlsx}] saved\n`);
  return { workbook: config.xlsx, rows: rowCount, columns: headers.length, outputPath };
}

const outputs = [];
const requested = new Set(process.argv.slice(2));
const selectedConfigs = requested.size
  ? configs.filter((_, index) => requested.has(String(index + 1)))
  : configs;
for (const config of selectedConfigs) {
  outputs.push(await buildOne(config));
}
await fs.writeFile(
  path.join(previewDir, `workbook_build_summary_${selectedConfigs.map(c => c.csv.match(/Table_(\d)_/)[1]).join("-")}.json`),
  JSON.stringify(outputs, null, 2) + "\n",
  "utf8",
);
process.stdout.write(JSON.stringify(outputs, null, 2) + "\n");

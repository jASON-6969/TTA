import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const here = path.dirname(fileURLToPath(import.meta.url));
const workspace = path.dirname(here);
const skillDir = "C:/Users/jason/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations";
const finalPath = path.join(workspace, "output", "benchmark_cxr_base_model_vs_paper.pptx");
const staging = path.join(here, "staging"), previews = path.join(here, "previews");
const font = "Arial", W = 1280, H = 720;
process.env.RUNTIME_NODE_MODULES = "C:/Users/jason/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
const paper = "C:/Users/jason/Documents/cource/y4sem1/fyp/TTA for medical image segmentation/TTA for medical image segmentation/Benchmark for Test Time Adaptation.pdf";
const summary = "C:/Users/jason/Documents/cource/y4sem1/fyp/test_v2/base_model/benchmark_cxr_base_no_source_aug/summary.json";
const audit = "C:/Users/jason/Documents/cource/y4sem1/fyp/test_v2/base_model/benchmark_cxr_base_no_source_aug/protocol_audit.json";
const log = "C:/Users/jason/Documents/cource/y4sem1/fyp/test_v2/base_model/benchmark_cxr_base_no_source_aug/execution.log";
const C = { bg: "#F5F7F6", white: "#FFFFFF", ink: "#1B2A2F", muted: "#58686D", soft: "#829094", rule: "#D9E1DF", teal: "#087F72", tealLight: "#D9EEE9", coral: "#C85F49", gold: "#C18A24" };

function box(s, x, y, w, h, fill) {
  return s.shapes.add({ geometry: "rect", position: { left: x, top: y, width: w, height: h }, fill, line: { style: "solid", fill: "none", width: 0 } });
}
function txt(s, value, x, y, w, h, o = {}) {
  const a = s.shapes.add({ geometry: "textbox", position: { left: x, top: y, width: w, height: h }, fill: "none", line: { style: "solid", fill: "none", width: 0 } });
  a.text = value;
  a.text.style = { typeface: font, fontSize: o.size ?? 18, bold: o.bold ?? false, color: o.color ?? C.ink, alignment: o.align ?? "left", verticalAlignment: o.valign ?? "middle", autoFit: "none", wrap: "square", insets: 0 };
  return a;
}
function rule(s, x, y, w, color = C.rule, h = 1) { box(s, x, y, w, h, color); }
function footer(s, n, dark = false) {
  rule(s, 64, 668, 1152, dark ? "#3A4B50" : C.rule);
  txt(s, "CXR BASELINE COMPARISON  |  SOURCE-ONLY", 64, 677, 800, 20, { size: 11, bold: true, color: dark ? "#AFC0C0" : C.soft });
  txt(s, String(n).padStart(2, "0"), 1165, 677, 50, 20, { size: 11, bold: true, color: dark ? "#AFC0C0" : C.soft, align: "right" });
}
function heading(s, title, subtitle) {
  box(s, 64, 38, 46, 4, C.teal);
  txt(s, title, 64, 48, 1140, 54, { size: 35, bold: true });
  txt(s, subtitle, 64, 108, 1120, 42, { size: 17, color: C.muted });
}
function newSlide(deck, title, subtitle) {
  const s = deck.slides.add(); s.background.fill = C.bg; heading(s, title, subtitle); return s;
}
function cite(s, lines) { s.speakerNotes.text = lines; }

const deck = Presentation.create({ slideSize: { width: W, height: H } });

// 1. Cover and headline Dice comparison.
{
  const s = deck.slides.add(); s.background.fill = C.ink; box(s, 0, 0, 14, H, C.teal); box(s, 14, 0, 6, H, C.coral);
  txt(s, "MEDSEG-TTA  /  CHEST X-RAY LUNG SEGMENTATION", 72, 52, 850, 28, { size: 14, bold: true, color: "#A8C7C0" });
  txt(s, "CXR base-model\nbenchmark", 72, 116, 735, 144, { size: 54, bold: true, color: C.white });
  txt(s, "Local source-only run compared with the paper's no-TTA baseline", 76, 277, 890, 34, { size: 20, color: "#D1DBD9" });
  txt(s, "PAPER BASELINE", 82, 374, 420, 24, { size: 13, bold: true, color: "#F1B7A8" });
  txt(s, "LOCAL BASE MODEL", 670, 374, 420, 24, { size: 13, bold: true, color: "#8DD0C1" });
  txt(s, "0.9093", 82, 406, 430, 82, { size: 64, bold: true, color: C.white });
  txt(s, "0.9396", 670, 406, 430, 82, { size: 64, bold: true, color: C.white });
  txt(s, "Dice mean | 138 Montgomery cases", 86, 493, 420, 26, { size: 15, color: "#C1CECC" });
  txt(s, "Dice mean | 138 Montgomery cases", 674, 493, 440, 26, { size: 15, color: "#C1CECC" });
  box(s, 84, 540, 450, 8, "#445258"); box(s, 84, 540, 409.2, 8, C.coral);
  box(s, 670, 540, 450, 8, "#445258"); box(s, 670, 540, 422.8, 8, C.teal);
  box(s, 1004, 94, 206, 110, "#263A3F");
  txt(s, "+0.0303", 1020, 104, 174, 48, { size: 32, bold: true, color: "#9FE0D1", align: "center" });
  txt(s, "+3.03 percentage points\ndescriptive difference", 1020, 153, 174, 43, { size: 13, color: "#D1DBD9", align: "center" });
  txt(s, "Same target cohort size; source-mask coverage and full training equivalence differ.", 84, 592, 1090, 28, { size: 16, color: "#D1DBD9" });
  footer(s, 1, true);
  cite(s, [`Paper baseline: ${paper}, Table 9, PDF p. 28.`, `Local metrics: ${summary}.`, "Local minus paper = 0.9395699204 - 0.9093 = 0.0302699204, rounded to 0.0303 (3.03 pp). Descriptive only; not a significance test or superiority claim."]);
}

// 2. Protocol and source-data comparison.
{
  const s = newSlide(deck, "Target benchmark aligned; source coverage differs", "Paper values are from the specified MedSeg-TTA PDF; local values are from the audited source-only run.");
  const x = [64, 300, 744], cw = [220, 420, 472];
  box(s, 64, 168, 1152, 42, C.ink);
  ["BENCHMARK ITEM", "PAPER", "LOCAL RUN"].forEach((v, i) => txt(s, v, x[i] + 12, 176, cw[i] - 24, 25, { size: 13, bold: true, color: C.white }));
  const rows = [
    ["Domain direction", "SZ-CXR source -> Montgomery target", "SZ-CXR source -> Montgomery target"],
    ["Target cohort", "138 posteroanterior chest radiographs", "Same 138 Montgomery cases"],
    ["Backbone", "UNet2D; 256 x 256; 1 input / 2 output; 5 blocks; C0=32", "Same listed UNet2D configuration"],
    ["Source labels", "662 SZ-CXR images with lung masks", "566 exact pairs; 96 unpaired images excluded"],
    ["Source split", "8:2 source train / validation protocol", "453 train / 113 validation from 566 pairs; seed 42"],
    ["Base condition", "Target-domain baseline without TTA", "Source-only; no target-time parameter update"],
  ];
  rows.forEach((r, i) => {
    const y = 214 + i * 62; if (i % 2 === 0) box(s, 64, y, 1152, 62, C.white);
    txt(s, r[0], x[0] + 12, y + 7, cw[0] - 24, 48, { size: 16, bold: true, color: C.muted });
    txt(s, r[1], x[1] + 12, y + 6, cw[1] - 24, 50, { size: 16 });
    txt(s, r[2], x[2] + 12, y + 6, cw[2] - 24, 50, { size: 16 });
    rule(s, 64, y + 61, 1152);
  });
  box(s, 64, 590, 6, 48, C.gold);
  txt(s, "Both scores are measured on the 138-case Montgomery target set before TTA.", 84, 594, 1100, 38, { size: 17, bold: true });
  footer(s, 2);
  cite(s, [`Paper dataset protocol: ${paper}, Section 3.1.7, PDF pp. 15-16.`, `Paper UNet setup and source split: ${paper}, Table 2 and Section 3.2, PDF pp. 18-19.`, `Local source coverage and split: ${audit}.`, "The paper reports 662 SZ-CXR images with lung-field masks and 138 Montgomery targets. The local run had 566 exact source image-mask pairs and excluded 96 unpaired images."]);
}

// 3. Primary endpoint as an editable error-bar plot.
{
  const s = newSlide(deck, "Local mean Dice is 3.03 points higher", "Means are over the same 138-case Montgomery cohort; whiskers show mean +/- 1 reported SD.");
  box(s, 885, 151, 331, 74, C.tealLight);
  txt(s, "+0.0303", 906, 159, 150, 50, { size: 35, bold: true, color: C.teal });
  txt(s, "+3.03 pp\nlocal minus paper", 1060, 161, 140, 46, { size: 15, bold: true });
  const x0 = 256, x1 = 1004, width = x1 - x0, yTop = 276, yBottom = 526;
  const scale = (v) => x0 + ((v - 0.8) / 0.2) * width;
  [0.8, 0.85, 0.9, 0.95, 1].forEach((v) => {
    const x = scale(v); box(s, x, yTop, 1, yBottom - yTop, C.rule);
    txt(s, v.toFixed(2), x - 28, 534, 56, 22, { size: 13, color: C.muted, align: "center" });
  });
  const points = [{ label: "PAPER", mean: 0.9093, sd: 0.0695, y: 350, color: C.coral }, { label: "LOCAL", mean: 0.9395699204, sd: 0.0537603167, y: 450, color: C.teal }];
  points.forEach((p) => {
    const lo = scale(Math.max(0.8, p.mean - p.sd)), hi = scale(Math.min(1, p.mean + p.sd)), mx = scale(p.mean);
    txt(s, p.label, 76, p.y - 17, 150, 30, { size: 15, bold: true, color: p.color });
    box(s, lo, p.y - 3, hi - lo, 6, p.color); box(s, lo, p.y - 13, 2, 26, p.color); box(s, hi - 2, p.y - 13, 2, 26, p.color);
    s.shapes.add({ geometry: "ellipse", position: { left: mx - 10, top: p.y - 10, width: 20, height: 20 }, fill: p.color, line: { style: "solid", fill: C.white, width: 2 } });
    txt(s, `${p.mean.toFixed(4)} +/- ${p.sd.toFixed(4)}`, 1018, p.y - 20, 194, 40, { size: 16, bold: true, color: p.color });
  });
  rule(s, x0, yBottom, width, C.soft, 2);
  box(s, 64, 583, 1152, 56, C.white);
  txt(s, "Interpretation", 82, 594, 145, 32, { size: 14, bold: true, color: C.teal });
  txt(s, "The local point estimate is higher; overlapping SD ranges are not a significance test.", 233, 591, 950, 38, { size: 15 });
  footer(s, 3);
  cite(s, [`Paper Dice baseline: ${paper}, Table 9, PDF p. 28.`, `Local Dice mean and population SD: ${summary}.`, "Whiskers show mean +/- one reported SD, clipped to the displayed 0.80-1.00 axis. They are descriptive distributions, not confidence intervals."]);
}

// 4. Secondary metrics and the HD95 caveat.
{
  const s = newSlide(deck, "Other overlap metrics are also higher locally", "Mean +/- SD across 138 target cases. HD95 is listed for context, not ranked across implementations.");
  const x = [76, 340, 644, 960], cw = [250, 290, 300, 250];
  box(s, 64, 178, 1152, 46, C.ink);
  ["METRIC", "PAPER", "LOCAL", "LOCAL - PAPER"].forEach((v, i) => txt(s, v, x[i] + 10, 187, cw[i] - 20, 26, { size: 13, bold: true, color: C.white, align: i === 0 ? "left" : "center" }));
  const rows = [
    ["Dice", "0.9093 +/- 0.0695", "0.9396 +/- 0.0538", "+0.0303"],
    ["Jaccard index", "0.8414 +/- 0.1038", "0.8903 +/- 0.0855", "+0.0489"],
    ["Sensitivity", "0.8534 +/- 0.1059", "0.8971 +/- 0.0843", "+0.0437"],
    ["Precision (PPV)", "0.9846 +/- 0.0272", "0.9920 +/- 0.0265", "+0.0074"],
    ["HD95 (pixels)", "5.15 +/- 9.11", "2.6404 +/- 6.8043", "Not compared"],
  ];
  rows.forEach((r, i) => {
    const y = 224 + i * 61; if (i % 2 === 0) box(s, 64, y, 1152, 61, C.white);
    r.forEach((v, j) => txt(s, v, x[j] + 10, y + 8, cw[j] - 20, 44, { size: j === 0 ? 17 : 16, bold: j === 0 || j === 3, color: j === 3 ? (i === 4 ? C.gold : C.teal) : C.ink, align: j === 0 ? "left" : "center" }));
    rule(s, 64, y + 60, 1152);
  });
  box(s, 64, 551, 6, 88, C.gold);
  txt(s, "HD95 is implementation-sensitive", 84, 554, 400, 28, { size: 16, bold: true, color: C.gold });
  txt(s, "Local audit uses an all-foreground-pixel nearest-distance statistic in resized pixel units. Metric equivalence with the paper has not been established.", 84, 582, 1100, 50, { size: 15, color: C.muted });
  footer(s, 4);
  cite(s, [`Paper Table 9 metrics: ${paper}, PDF p. 28.`, `Local target metrics: ${summary}.`, `Local HD95 policy: ${audit}.`, "Local-minus-paper differences are arithmetic on displayed rounded means, not paired improvements or statistical evidence. Local HD95 is reported in resized 256x256 pixel units using the audited all-foreground nearest-distance implementation."]);
}

// 5. Limits on interpretation.
{
  const s = newSlide(deck, "The score gap cannot be assigned to model quality alone", "The task is closely matched, but source training evidence is not identical.");
  const items = [
    [174, C.teal, "MATCHED", "Target task and model family", "SZ-CXR -> Montgomery; 138 targets; source-only inference; same listed five-block UNet2D width and I/O channels."],
    [273, C.coral, "DIFFERS", "Source supervision coverage", "Paper reports 662 source images with masks. Local training used 566 exact pairs and excluded 96 unpaired images."],
    [372, C.gold, "NOT FULLY ALIGNED", "Training and preprocessing detail", "Local run records z-score, no augmentation, Adam, batch 2, and one source seed. The paper main text does not fully specify every setting."],
    [471, C.coral, "NOT ESTABLISHED", "Metric and statistical equivalence", "HD95 implementation equivalence is unverified. Local evidence is one seed; no matched per-case cross-paper test is available."],
  ];
  items.forEach(([y, color, label, title, body]) => {
    box(s, 64, y, 5, 82, color); txt(s, label, 86, y + 2, 190, 22, { size: 12, bold: true, color });
    txt(s, title, 86, y + 26, 342, 40, { size: 20, bold: true }); txt(s, body, 448, y + 4, 740, 72, { size: 16, color: C.muted });
    rule(s, 86, y + 88, 1100);
  });
  box(s, 64, 592, 1152, 48, C.ink);
  txt(s, "Report +3.03 pp as a descriptive result, not proof of superiority or significance.", 82, 600, 1110, 32, { size: 16, bold: true, color: C.white });
  footer(s, 5);
  cite(s, [`Paper dataset protocol: ${paper}, Section 3.1.7, PDF pp. 15-16.`, `Paper backbone and implementation: ${paper}, Table 2 and Section 3.2, PDF pp. 18-19.`, `Local source coverage, recipe, and metric policy: ${audit}.`, `Local single-seed execution: ${log} and ${summary}.`]);
}

// 6. Conclusion and source trail.
{
  const s = newSlide(deck, "Baseline comparison: evidence and next steps", "A local reproduction milestone, with a clear boundary on what the numbers establish.");
  txt(s, "WHAT THE EVIDENCE SUPPORTS", 72, 177, 525, 28, { size: 13, bold: true, color: C.teal }); rule(s, 72, 211, 500, C.teal, 2);
  txt(s, "The local model reaches 0.9396 mean Dice on all 138 Montgomery cases before TTA.\n\nThe paper's Table 9 baseline is 0.9093 mean Dice on 138 targets.\n\nThe +0.0303 gap is descriptive; source coverage and full protocol matching remain incomplete.", 88, 229, 490, 222, { size: 16 });
  txt(s, "TO STRENGTHEN THE REPRODUCTION", 650, 177, 558, 28, { size: 13, bold: true, color: C.coral }); rule(s, 650, 211, 558, C.coral, 2);
  txt(s, "Recover or document the missing 96 source masks.\n\nLock source split, preprocessing, optimizer, and checkpoint selection to paper evidence.\n\nUse one metric implementation and repeat across seeds before making comparative claims.", 666, 229, 510, 222, { size: 16 });
  box(s, 64, 478, 1152, 154, C.white);
  txt(s, "SOURCE TRAIL", 84, 489, 170, 24, { size: 12, bold: true, color: C.soft });
  txt(s, "Yu et al., A Large Scale Benchmark for Test Time Adaptation Methods in Medical Image Segmentation.\nCXR source/target protocol: Section 3.1.7, pp. 15-16. UNet2D setup: Table 2, p. 18.\nCXR source-only baseline: Table 9, p. 28 (Dice 0.9093 +/- 0.0695).", 84, 517, 1090, 72, { size: 14 });
  txt(s, "Local evidence: base_model/benchmark_cxr_base_no_source_aug/summary.json and protocol_audit.json", 84, 600, 1090, 22, { size: 12, color: C.muted });
  footer(s, 6);
  cite(s, [`Paper: ${paper}; Section 3.1.7 (PDF pp. 15-16), Table 2 (p. 18), Table 9 (p. 28).`, `Local target metrics/checkpoint: ${summary}.`, `Local protocol/source coverage: ${audit}.`, `Local execution log: ${log}.`]);
}

await fs.mkdir(staging, { recursive: true });
await fs.mkdir(previews, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });
for (const [i, slide] of deck.slides.items.entries()) {
  const png = await deck.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(previews, `slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
}
const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);
const candidatePath = path.join(staging, "candidate.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
const result = await finalizePresentation({
  workspaceDir: workspace, candidatePath, finalPath,
  pythonExecutable: "C:/Users/jason/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit"],
  requiredNativeTableOwnerSlides: [], requiredNativeChartOwnerSlides: [], explicitTotalSlideCount: 6,
  fontPolicy: { basis: "design", families: [font] }, verifyArtifactToolImport: true,
  receiptPath: path.join(staging, "benchmark_cxr_base_model_vs_paper.validation.json"),
});
console.log(JSON.stringify({ finalPath, slideCount: deck.slides.items.length, result }, null, 2));

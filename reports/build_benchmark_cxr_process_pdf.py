from __future__ import annotations

import json
import csv
from statistics import pstdev
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "benchmark_cxr_reproduction_process.pdf"
BASE_NEW = ROOT / "output" / "experiments" / "benchmark_cxr_base_no_source_aug" / "summary.json"
METHODS_NEW = ROOT / "output" / "experiments" / "benchmark_cxr_methods_no_source_aug" / "summary.json"
AUDIT_NEW = ROOT / "output" / "experiments" / "benchmark_cxr_methods_no_source_aug" / "protocol_audit.json"
DATA_AUDIT = ROOT / "output" / "experiments" / "benchmark_cxr_source_data_audit.json"
BASE_OLD = ROOT / "output" / "experiments" / "benchmark_cxr_base" / "summary.json"
METRICS_OLD = ROOT / "output" / "experiments" / "benchmark_cxr_base" / "metrics.csv"

INK = colors.HexColor("#202B33")
MUTED = colors.HexColor("#53636F")
TEAL = colors.HexColor("#117C83")
BLUE = colors.HexColor("#356AA0")
PALE = colors.HexColor("#EAF2F4")
PALE_BLUE = colors.HexColor("#EDF2F8")
PALE_GOLD = colors.HexColor("#F8F1DE")
LINE = colors.HexColor("#D6E0E4")
WHITE = colors.white


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ProcessFlow(Flowable):
    def __init__(self, width: float):
        super().__init__()
        self.width = width
        self.height = 202

    def draw(self) -> None:
        steps = [
            ("1", "Fix the CXR protocol", "SZ-CXR source -> Montgomery target; 2D UNet, 256 x 256."),
            ("2", "Audit paired data", "662 source images; 566 masks; 96 unpaired images excluded."),
            ("3", "Train source-only base", "Split paired source cases 80:20; select checkpoint by source validation Dice."),
            ("4", "Adapt without target masks", "Run TENT, TestFit and GraTa on the ordered target stream."),
            ("5", "Evaluate after adaptation", "Load target masks only after predictions are saved; score all 138 cases."),
        ]
        box_h = 34
        gap = 7
        y = self.height - box_h
        for index, (number, title, detail) in enumerate(steps):
            self.canv.setFillColor(PALE if index % 2 == 0 else PALE_BLUE)
            self.canv.setStrokeColor(LINE)
            self.canv.roundRect(0, y, self.width, box_h, 4, fill=1, stroke=1)
            self.canv.setFillColor(TEAL)
            self.canv.circle(18, y + box_h / 2, 10, fill=1, stroke=0)
            self.canv.setFillColor(WHITE)
            self.canv.setFont("Helvetica-Bold", 9)
            self.canv.drawCentredString(18, y + box_h / 2 - 3, number)
            self.canv.setFillColor(INK)
            self.canv.setFont("Helvetica-Bold", 9)
            self.canv.drawString(36, y + 20, title)
            self.canv.setFillColor(MUTED)
            self.canv.setFont("Helvetica", 7.5)
            self.canv.drawString(36, y + 8, detail)
            if index < len(steps) - 1:
                self.canv.setStrokeColor(TEAL)
                self.canv.setLineWidth(1.1)
                self.canv.line(18, y - 1, 18, y - gap + 2)
                self.canv.line(18, y - gap + 2, 15, y - gap + 5)
                self.canv.line(18, y - gap + 2, 21, y - gap + 5)
            y -= box_h + gap


class PairedBars(Flowable):
    def __init__(self, width: float, height: float, labels: list[str], old: list[float], new: list[float]):
        super().__init__()
        self.width = width
        self.height = height
        self.labels = labels
        self.old = old
        self.new = new

    def draw(self) -> None:
        left = 96
        right = self.width - 16
        top = self.height - 25
        row_h = 44
        plot_w = right - left
        self.canv.setFillColor(MUTED)
        self.canv.setFont("Helvetica", 7.5)
        self.canv.drawString(left, self.height - 11, "Metric value (0 to 1)")
        self.canv.setFillColor(BLUE)
        self.canv.rect(right - 150, self.height - 14, 7, 7, fill=1, stroke=0)
        self.canv.setFillColor(MUTED)
        self.canv.drawString(right - 139, self.height - 13, "Initial pipeline")
        self.canv.setFillColor(TEAL)
        self.canv.rect(right - 66, self.height - 14, 7, 7, fill=1, stroke=0)
        self.canv.setFillColor(MUTED)
        self.canv.drawString(right - 55, self.height - 13, "Adjusted pipeline")
        for i, label in enumerate(self.labels):
            y = top - i * row_h
            self.canv.setFillColor(INK)
            self.canv.setFont("Helvetica-Bold", 8)
            self.canv.drawRightString(left - 9, y - 2, label)
            self.canv.setFillColor(colors.HexColor("#E6ECEF"))
            self.canv.roundRect(left, y - 1, plot_w, 8, 2, fill=1, stroke=0)
            self.canv.roundRect(left, y - 13, plot_w, 8, 2, fill=1, stroke=0)
            self.canv.setFillColor(BLUE)
            self.canv.roundRect(left, y - 1, plot_w * self.old[i], 8, 2, fill=1, stroke=0)
            self.canv.setFillColor(TEAL)
            self.canv.roundRect(left, y - 13, plot_w * self.new[i], 8, 2, fill=1, stroke=0)
            self.canv.setFillColor(MUTED)
            self.canv.setFont("Helvetica", 7)
            self.canv.drawString(min(left + plot_w * self.old[i] + 4, right - 28), y, f"{self.old[i]:.3f}")
            self.canv.drawString(min(left + plot_w * self.new[i] + 4, right - 28), y - 12, f"{self.new[i]:.3f}")
        base_y = top - len(self.labels) * row_h + 9
        self.canv.setStrokeColor(LINE)
        self.canv.line(left, base_y, right, base_y)
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = left + plot_w * tick
            self.canv.setStrokeColor(LINE)
            self.canv.line(x, base_y, x, base_y - 4)
            self.canv.setFillColor(MUTED)
            self.canv.setFont("Helvetica", 6.5)
            self.canv.drawCentredString(x, base_y - 13, f"{tick:.2f}")


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleCustom", parent=base["Title"], fontName="Helvetica-Bold", fontSize=23, leading=27, textColor=INK, alignment=TA_LEFT, spaceAfter=8),
        "subtitle": ParagraphStyle("SubtitleCustom", parent=base["Normal"], fontName="Helvetica", fontSize=10, leading=14, textColor=MUTED, spaceAfter=12),
        "h1": ParagraphStyle("H1Custom", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=INK, spaceBefore=4, spaceAfter=8),
        "h2": ParagraphStyle("H2Custom", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=TEAL, spaceBefore=7, spaceAfter=4),
        "body": ParagraphStyle("BodyCustom", parent=base["BodyText"], fontName="Helvetica", fontSize=8.7, leading=12.2, textColor=INK, spaceAfter=5),
        "small": ParagraphStyle("SmallCustom", parent=base["BodyText"], fontName="Helvetica", fontSize=7.2, leading=9.2, textColor=MUTED, spaceAfter=3),
        "table": ParagraphStyle("TableCustom", parent=base["BodyText"], fontName="Helvetica", fontSize=7.2, leading=9, textColor=INK),
        "table_head": ParagraphStyle("TableHeadCustom", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.1, leading=8.6, textColor=WHITE),
        "callout": ParagraphStyle("CalloutCustom", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=9, leading=12.5, textColor=INK),
        "metric": ParagraphStyle("MetricCustom", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=16, leading=18, textColor=TEAL, alignment=TA_CENTER),
        "metric_label": ParagraphStyle("MetricLabelCustom", parent=base["BodyText"], fontName="Helvetica", fontSize=7, leading=9, textColor=MUTED, alignment=TA_CENTER),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def standard_table(data: list[list], widths: list[float], header: bool = True) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
    ]
    if header:
        commands.extend([("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), WHITE)])
        if len(data) > 1:
            commands.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#F4F7F8")]))
    table.setStyle(TableStyle(commands))
    return table


def metric_cards(summary: dict, width: float, s: dict) -> Table:
    dice = summary["target_metrics"]["Dice_mean"]
    val = summary["source_training"]["best_val_dice"]
    cards = []
    for number, label in [(f"{dice:.4f}", "Source-only target Dice, n=138"), (f"{val:.4f}", "Best source validation Dice"), ("566 / 662", "Paired source masks / paper images")]:
        cards.append([
            [para(number, s["metric"])],
            [para(label, s["metric_label"])],
        ])
    widths = [width / 3] * 3
    table = Table([[Table(card, colWidths=[width / 3 - 14]) for card in cards]], colWidths=widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, WHITE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build_pdf() -> None:
    base = read_json(BASE_NEW)
    methods = read_json(METHODS_NEW)
    audit = read_json(AUDIT_NEW)
    old = read_json(BASE_OLD)
    data_audit = read_json(DATA_AUDIT)
    if data_audit.get("complete_public_662_mask_archive_found"):
        raise RuntimeError("Data audit changed: complete source mask archive found; refresh report wording.")
    with METRICS_OLD.open("r", encoding="utf-8", newline="") as stream:
        old_rows = list(csv.DictReader(stream))
    old_population_sd = {
        metric: pstdev(float(row[metric]) for row in old_rows)
        for metric in ("Dice", "Sensitivity", "PPV")
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    s = styles()
    page_w, page_h = A4
    left = 18 * mm
    right = 18 * mm
    top = 18 * mm
    bottom = 17 * mm
    usable = page_w - left - right
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=left,
        rightMargin=right,
        topMargin=top,
        bottomMargin=bottom,
        title="CXR Benchmark Reproduction Process",
        author="Reproduction record",
        subject="MedSeg-TTA CXR source-only and target-time adaptation process",
    )

    def page_chrome(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.5)
        canvas.line(left, page_h - 12 * mm, page_w - right, page_h - 12 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(left, 9 * mm, "MedSeg-TTA CXR reproduction process | local evidence record")
        canvas.drawRightString(page_w - right, 9 * mm, f"{document.page}")
        canvas.restoreState()

    story: list[Flowable] = []

    # Page 1: result and process map.
    story += [
        para("CXR Benchmark<br/>Reproduction Process", s["title"]),
        para("A process record for SZ-CXR to Montgomery lung segmentation | Single source seed 42", s["subtitle"]),
        metric_cards(base, usable, s),
        Spacer(1, 10),
        para("Executive conclusion", s["h1"]),
        para(
            "The local run reproduces the benchmark's CXR task structure and completes source-only evaluation plus three target-only adaptation runs. The adjusted preprocessing/training pipeline reached <b>0.9592 Source-only Dice</b> on all 138 Montgomery cases. TENT, TestFit and GraTa changed Dice only slightly in this single-seed run. This is a local reproduction record, not a full reproduction of the paper's released results or official method wrappers.",
            s["body"],
        ),
        para("Process map", s["h2"]),
        ProcessFlow(usable),
        Spacer(1, 4),
        para(
            "Target masks were isolated from adaptation inputs and used only after the prediction streams completed. The reported target metrics are per-image means over the full Montgomery set.",
            s["small"],
        ),
        PageBreak(),
    ]

    # Page 2: protocol and implementation.
    story += [
        para("1. Protocol and run sequence", s["h1"]),
        para("Paper protocol used as the reproduction target", s["h2"]),
    ]
    protocol_rows = [
        [para("Item", s["table_head"]), para("Benchmark specification", s["table_head"]), para("Local run", s["table_head"])],
        [para("Dataset pair", s["table"]), para("SZ-CXR source to Montgomery target", s["table"]), para("Same direction; 566 paired source masks available", s["table"])],
        [para("Backbone", s["table"]), para("2D UNet; 256 x 256; 1 input; 2 output; 5 encoder blocks; C0=32", s["table"]), para("Five-level UNet, grayscale, binary output, base width 32", s["table"])],
        [para("Partition", s["table"]), para("Source split 8:2; entire target used for out-of-domain testing", s["table"]), para("453 source train / 113 source validation; 138 target", s["table"])],
        [para("TTA access", s["table"]), para("No source images or labels during target adaptation", s["table"]), para("Image-only adaptation records; target masks loaded after predictions", s["table"])],
        [para("Metrics", s["table"]), para("Dice, HD95, JI, Sensitivity, PPV", s["table"]), para("All five reported per image; HD95 in resized pixel units", s["table"])],
    ]
    story += [standard_table(protocol_rows, [90, 194, usable - 284]), Spacer(1, 7)]
    story += [para("Local execution settings", s["h2"])]
    settings = [
        [para("Stage", s["table_head"]), para("Implementation", s["table_head"]), para("Audit note", s["table_head"])],
        [para("Source preprocessing", s["table"]), para("Resize to 256 x 256; per-image z-score; no source augmentation", s["table"]), para("The paper's main text does not provide a complete preprocessing recipe", s["table"])],
        [para("Source training", s["table"]), para("Adam, lr 1e-3, batch 2, max 25 epochs, patience 5; best validation checkpoint", s["table"]), para(f"Best source validation Dice {base['source_training']['best_val_dice']:.4f} at epoch {base['source_training']['best_epoch']}", s["table"])],
        [para("Target stream", s["table"]), para("Sorted target images; one update, then one prediction per image; state persists across stream", s["table"]), para("138 adaptation steps per method; target masks are unavailable to adapters", s["table"])],
        [para("Method scope", s["table"]), para("TENT entropy / BN-affine adapter; TestFit pseudo-label and feature adapter; GraTa noisy-view and feature consistency adapter", s["table"]), para("Local adapters; not line-by-line official wrappers", s["table"])],
    ]
    story += [standard_table(settings, [86, 204, usable - 290]), Spacer(1, 5)]
    story += [
        para(
            "The paper specifies the shared architecture and high-level protocol, but the main text does not provide every optimizer, epoch, batch-size and adaptation learning-rate value used here. Local choices are therefore recorded explicitly instead of being presented as paper defaults.",
            s["small"],
        ),
        PageBreak(),
    ]

    # Page 3: preprocessing diagnosis.
    old_m = old["target_metrics"]
    new_m = base["target_metrics"]
    story += [
        para("2. Preprocessing and base-model diagnosis", s["h1"]),
        para("Observed change across two local runs", s["h2"]),
        para(
            "Both local base runs used no source augmentation. The initial run used per-image 1st/99th-percentile min-max scaling to [0,1], while the formal reproduction run uses per-image z-score. The initial predictions had low sensitivity and very high precision, a pattern consistent with under-segmenting the lung fields. This comparison gives a cleaner normalization diagnosis, although it does not prove that normalization is the only implementation difference from the paper.",
            s["body"],
        ),
        PairedBars(usable, 178, ["Dice", "Sensitivity", "PPV"], [old_m["Dice_mean"], old_m["Sensitivity_mean"], old_m["PPV_mean"]], [new_m["Dice_mean"], new_m["Sensitivity_mean"], new_m["PPV_mean"]]),
        Spacer(1, 4),
    ]
    before_after = [
        [para("Metric", s["table_head"]), para("Initial pipeline", s["table_head"]), para("Adjusted pipeline", s["table_head"]), para("Interpretation", s["table_head"])],
        [para("Dice", s["table"]), para(f"{old_m['Dice_mean']:.4f} +/- {old_population_sd['Dice']:.4f}", s["table"]), para(f"{new_m['Dice_mean']:.4f} +/- {new_m['Dice_std']:.4f}", s["table"]), para("Overlap improved substantially", s["table"])],
        [para("Sensitivity", s["table"]), para(f"{old_m['Sensitivity_mean']:.4f}", s["table"]), para(f"{new_m['Sensitivity_mean']:.4f}", s["table"]), para("More target lung pixels recovered", s["table"])],
        [para("PPV", s["table"]), para(f"{old_m['PPV_mean']:.4f}", s["table"]), para(f"{new_m['PPV_mean']:.4f}", s["table"]), para("Precision remained high", s["table"])],
    ]
    story += [standard_table(before_after, [75, 105, 112, usable - 292]), Spacer(1, 8)]
    story += [
        para("What this diagnostic establishes", s["h2"]),
        para(
            "The initial result should not be used to judge TTA: the base prediction was already severely under-segmented. The no-augmentation z-score pipeline restored much of the source-to-target performance. It is evidence that preprocessing fidelity matters; it is not proof that every remaining difference from the paper has been eliminated.",
            s["body"],
        ),
        para("HD95 caution", s["h2"]),
        para(
            "The initial run used MedPy surface HD95 (mean 71.96 pixels). The adjusted run reports the vendor helper's all-foreground-pixel nearest-distance 95th percentile (mean 1.39 pixels). These HD95 values use different definitions and must not be treated as a before/after improvement. The adjusted statistic is measured on 256 x 256 masks without physical spacing.",
            s["body"],
        ),
        PageBreak(),
    ]

    # Page 4: paper comparison and conclusion.
    paper = {
        "Source-only": {"dice": .9093, "sd": .0695, "hd95": 5.15, "hd_sd": 9.11, "delta": 0.0},
        "TENT": {"dice": .9325, "sd": .0630, "hd95": 3.69, "hd_sd": 10.20, "delta": .0232},
        "TestFit": {"dice": .9120, "sd": .0849, "hd95": 5.59, "hd_sd": 11.62, "delta": .0027},
        "GraTa": {"dice": .9212, "sd": .0733, "hd95": 4.73, "hd_sd": 11.46, "delta": .0119},
    }
    names = ["Source-only", "TENT", "TestFit", "GraTa"]
    local_base_dice = methods["methods"]["Source-only"]["Dice_mean"]
    local_deltas = {
        name: methods["methods"][name]["Dice_mean"] - local_base_dice
        for name in ("TENT", "TestFit", "GraTa")
    }
    local_best = max(local_deltas, key=local_deltas.get)
    table_data = [[
        para("Method", s["table_head"]),
        para("Paradigm", s["table_head"]),
        para("Paper Dice<br/>mean +/- SD", s["table_head"]),
        para("Paper HD95<br/>mean +/- SD", s["table_head"]),
        para("Local Dice<br/>mean +/- SD", s["table_head"]),
        para("Local HD95*", s["table_head"]),
        para("Local Dice<br/>delta vs base", s["table_head"]),
    ]]
    for name in names:
        ref = paper[name]
        local = methods["methods"][name]
        paradigm = "Baseline" if name == "Source-only" else ("Output-level" if name == "TENT" else "Feature-level")
        delta = 0.0 if name == "Source-only" else local["Dice_mean"] - methods["methods"]["Source-only"]["Dice_mean"]
        table_data.append([
            para(name, s["table"]),
            para(paradigm, s["table"]),
            para(f"{ref['dice']:.4f} +/- {ref['sd']:.4f}", s["table"]),
            para(f"{ref['hd95']:.2f} +/- {ref['hd_sd']:.2f}", s["table"]),
            para(f"{local['Dice_mean']:.4f} +/- {local['Dice_std']:.4f}", s["table"]),
            para(f"{local['HD95_mean']:.2f}", s["table"]),
            para("-" if name == "Source-only" else f"{delta:+.4f}", s["table"]),
        ])
    widths = [67, 70, 79, 79, 91, 52, usable - 438]
    story += [
        para("3. Comparison with the benchmark paper", s["h1"]),
        para("CXR results from Table 9 and the adjusted local run", s["h2"]),
        standard_table(table_data, widths),
        Spacer(1, 6),
        para(
            "Paper baseline is named Target-Domain (w/o TTA); it is displayed beside local Source-only for task-level context. * Local HD95 uses the all-foreground-pixel nearest-distance implementation and is listed for traceability, not as a guaranteed like-for-like comparison with the paper's generated result. Paper values are reported as mean +/- SD; local SD is population SD across 138 target images. Local results use one source seed.",
            s["small"],
        ),
        para("Interpretation", s["h2"]),
        para(
            f"In the paper, TENT has the largest Dice gain among these three methods (+0.0232), while TestFit's Dice gain is small (+0.0027) and its reported HD95 is worse than the no-TTA row. In the no-source-augmentation local run, the Dice deltas are TENT {local_deltas['TENT']:+.4f}, TestFit {local_deltas['TestFit']:+.4f}, and GraTa {local_deltas['GraTa']:+.4f}. {local_best} is numerically highest in this single-seed run, but this does not establish a reliable advantage or contradict the paper because the source mask coverage, source training recipe and method implementations differ.",
            s["body"],
        ),
    ]
    limitation_rows = [
        [para("Limit", s["table_head"]), para("Effect on interpretation", s["table_head"])],
        [para("Source-mask coverage", s["table"]), para("The paper uses 662 source images with masks; local supervised training could use 566 exact pairs and excluded 96.", s["table"])],
        [para("Complete source data", s["table"]), para("Four public candidate manifests were audited; none supplied masks for all 662 CHNCXR IDs. The two relevant mask archives both stop at 566 exact pairs.", s["table"])],
        [para("Method fidelity", s["table"]), para("Local TENT, TestFit and GraTa adapters are not the official wrappers or their exact update schedules/losses.", s["table"])],
        [para("Statistical evidence", s["table"]), para("One source seed; per-image SD is not a multi-seed confidence interval and does not establish significance.", s["table"])],
        [para("HD95", s["table"]), para("Pixel-space implementation and full evaluation pipeline differ; do not infer equivalence from the displayed numbers.", s["table"])],
    ]
    story += [Spacer(1, 5), standard_table(limitation_rows, [105, usable - 105]), Spacer(1, 5)]
    story += [
        para("Conclusion", s["h2"]),
        para(
            "This work establishes a reproducible local CXR process and shows that replacing percentile scaling with per-image z-score, while keeping source augmentation disabled, substantially improves source-only target Dice. Under this base setup, TTA adds a modest gain, with TENT highest in this single-seed run. The evidence supports a local reproduction milestone, not full reproduction of MedSeg-TTA. A closer paper replication requires the complete source annotations and official source-training and TTA protocols, followed by multi-seed evaluation.",
            s["body"],
        ),
        para("Source trail", s["h2"]),
        para(
            "Benchmark paper: Yu et al., <i>A Large Scale Benchmark for Test Time Adaptation Methods in Medical Image Segmentation</i>, Table 2 and Sections 3.2/4.3.5 (local PDF pp. 18-19, 27-28; CXR results in Table 9). Local evidence: benchmark_cxr_base_no_source_aug/summary.json; benchmark_cxr_methods_no_source_aug/summary.json and protocol_audit.json; benchmark_cxr_source_data_audit.json; experiments/feature_alignment_2d_pilot/benchmark_config.py, data.py, run_base_model.py and run_selected_methods.py.",
            s["small"],
        ),
    ]

    doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    print(OUTPUT)


if __name__ == "__main__":
    build_pdf()

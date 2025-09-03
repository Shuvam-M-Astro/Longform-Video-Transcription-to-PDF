from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Callable
import json

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Preformatted


def _add_images(flow: List, title: str, image_paths: List[Path], max_width: float):
    if not image_paths:
        return
    flow.append(Paragraph(title, getSampleStyleSheet()["Heading2"]))
    flow.append(Spacer(1, 0.2 * inch))
    for img_path in image_paths:
        try:
            img = Image(str(img_path))
            # scale to width
            w, h = img.wrap(0, 0)
            if w > max_width:
                scale = max_width / w
                img.drawWidth = w * scale
                img.drawHeight = h * scale
            flow.append(img)
            flow.append(Spacer(1, 0.2 * inch))
        except Exception:
            continue


def build_pdf_report(
    output_pdf_path: Path,
    transcript_txt_path: Path,
    segments_json_path: Path,
    classification_result: Dict[str, List[Path]],
    output_dir: Path,
    *,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Path:
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(output_pdf_path), pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    heading1 = styles["Heading1"]
    code_style = ParagraphStyle(
        name="Code",
        parent=body,
        fontName="Courier",
        fontSize=9,
        leading=11,
    )
    timestamp_style = ParagraphStyle(
        name="Timestamp",
        parent=body,
        fontSize=8,
        leading=10,
        textColor=colors.grey,
        spaceBefore=4,
        spaceAfter=2,
    )

    def _format_hms(seconds: float) -> str:
        try:
            total = int(max(0, round(seconds)))
            h = total // 3600
            m = (total % 3600) // 60
            s = total % 60
            return f"{h:02d}:{m:02d}:{s:02d}"
        except Exception:
            return "00:00:00"

    flow = []
    flow.append(Paragraph("Video Transcript Report", heading1))
    flow.append(Spacer(1, 0.25 * inch))

    # Load structured segments if available
    segments: List[Dict] = []
    try:
        with open(segments_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                segments = [s for s in data if isinstance(s, dict) and {"start", "end", "text"} <= set(s.keys())]
    except Exception:
        segments = []

    # Overview
    flow.append(Paragraph("Overview", styles["Heading2"]))
    flow.append(Spacer(1, 0.1 * inch))
    total_duration = 0.0
    if segments:
        try:
            total_duration = max(float(s.get("end", 0.0)) for s in segments)
        except Exception:
            total_duration = 0.0
    overview_lines = [
        f"Segments: {len(segments)}" if segments else "Segments: N/A",
        f"Duration: {_format_hms(total_duration)}" if total_duration > 0 else "Duration: N/A",
    ]
    for line in overview_lines:
        flow.append(Paragraph(line, body))
    flow.append(Spacer(1, 0.2 * inch))

    # Transcript
    flow.append(Paragraph("Transcript", styles["Heading2"]))
    flow.append(Spacer(1, 0.15 * inch))

    if segments:
        for s in segments:
            try:
                start = float(s.get("start", 0.0))
                end = float(s.get("end", 0.0))
                text = str(s.get("text", "")).strip()
            except Exception:
                start, end, text = 0.0, 0.0, ""
            ts = f"[{_format_hms(start)} - {_format_hms(end)}]"
            flow.append(Paragraph(ts, timestamp_style))
            if text:
                flow.append(Paragraph(text, body))
            flow.append(Spacer(1, 0.1 * inch))
    else:
        # Fallback: include raw transcript text
        try:
            transcript_text = Path(transcript_txt_path).read_text(encoding="utf-8")
        except Exception:
            transcript_text = ""
        if transcript_text:
            flow.append(Preformatted(transcript_text, body, dedent=0))

    flow.append(PageBreak())

    # Code snippets
    snippets_dir = Path(output_dir) / "snippets" / "code"
    code_snippets = sorted(snippets_dir.glob("*.txt"))
    if code_snippets:
        flow.append(Paragraph("Extracted Code", styles["Heading2"]))
        flow.append(Spacer(1, 0.15 * inch))
        for snip in code_snippets:
            try:
                txt = snip.read_text(encoding="utf-8")
                flow.append(Preformatted(txt, code_style, dedent=0))
                flow.append(Spacer(1, 0.15 * inch))
            except Exception:
                continue
        flow.append(PageBreak())

    max_img_width = A4[0] - (doc.leftMargin + doc.rightMargin)
    _add_images(flow, "Plots and Diagrams", classification_result.get("plots", []), max_img_width)
    flow.append(PageBreak())
    _add_images(flow, "Images", classification_result.get("images", []), max_img_width)

    if progress_cb:
        progress_cb(50.0)
    doc.build(flow)
    if progress_cb:
        progress_cb(100.0)
    return output_pdf_path

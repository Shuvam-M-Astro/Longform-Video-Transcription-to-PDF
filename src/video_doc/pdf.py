from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
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

    flow = []
    flow.append(Paragraph("Video Documentation", heading1))
    flow.append(Spacer(1, 0.25 * inch))

    # Transcript
    flow.append(Paragraph("Transcript", styles["Heading2"]))
    flow.append(Spacer(1, 0.15 * inch))
    try:
        transcript_text = Path(transcript_txt_path).read_text(encoding="utf-8")
    except Exception:
        transcript_text = ""
    if transcript_text:
        # Keep it manageable: add as preformatted
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

    doc.build(flow)
    return output_pdf_path

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Callable
import json
import re
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Preformatted
from reportlab.platypus.tableofcontents import TableOfContents


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
    report_style: str = "minimal",
    video_title: str = "Video Report",
) -> Path:
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(output_pdf_path), pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    code_style = ParagraphStyle(
        name="Code",
        parent=body,
        fontName="Courier",
        fontSize=9,
        leading=11,
    )

    flow: List = []

    # Load structured segments if available
    segments: List[Dict] = []
    try:
        with open(segments_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                segments = [s for s in data if isinstance(s, dict) and {"start", "end", "text"} <= set(s.keys())]
    except Exception:
        segments = []

    # Helpers for text processing
    def _normalize_text(items: List[Dict]) -> str:
        joined: List[str] = []
        for it in items:
            txt = str(it.get("text", "")).strip()
            if txt:
                joined.append(txt)
        text = " ".join(joined)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _split_sentences(text: str) -> List[str]:
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s and len(s.strip()) > 1]

    def _summarize(sentences: List[str], max_sentences: int = 5) -> List[str]:
        if not sentences:
            return []
        stop = set([
            "the","a","an","and","or","but","if","then","so","of","to","in","on","for","with","as","is","are","was","were","be","been","it","this","that","these","those","at","by","from","about","into","over","after","before","between","out","up","down","off","above","under","again","further","once","do","does","did","doing","have","has","had","having","i","we","you","he","she","they","them","my","your","our","their","me","him","her","us","ours","yours","theirs",
        ])
        def words(s: str) -> List[str]:
            return re.findall(r"[a-zA-Z']+", s.lower())
        word_freq: Dict[str, int] = {}
        for s in sentences:
            for w in words(s):
                if w in stop:
                    continue
                word_freq[w] = word_freq.get(w, 0) + 1
        if not word_freq:
            return sentences[:max_sentences]
        max_freq = max(word_freq.values())
        scored: List[tuple] = []
        for i, s in enumerate(sentences):
            score = 0.0
            for w in words(s):
                if w in stop:
                    continue
                score += word_freq.get(w, 0) / max_freq
            length = len(s)
            if length < 40:
                score *= 0.7
            elif length > 300:
                score *= 0.8
            scored.append((i, score))
        top = sorted(scored, key=lambda t: t[1], reverse=True)[:max_sentences]
        idxs = sorted(i for i, _ in top)
        return [sentences[i] for i in idxs]

    def _to_paragraphs(items: List[Dict], target_chars: int = 800) -> List[str]:
        text = _normalize_text(items)
        if not text:
            return []
        sentences = _split_sentences(text)
        paragraphs: List[str] = []
        buf: List[str] = []
        count = 0
        for sent in sentences:
            buf.append(sent)
            count += len(sent)
            if count >= target_chars:
                paragraphs.append(" ".join(buf).strip())
                buf = []
                count = 0
        if buf:
            paragraphs.append(" ".join(buf).strip())
        return paragraphs

    def _chapterize(items: List[Dict]) -> List[Dict]:
        if not items:
            return []
        try:
            total_duration = max(float(s.get("end", 0.0)) for s in items)
        except Exception:
            total_duration = 0.0
        if total_duration <= 0:
            n = 3
            size = max(1, len(items) // n)
            chapters = []
            for i in range(n):
                chunk = items[i * size : (i + 1) * size]
                if not chunk:
                    continue
                chapters.append({
                    "index": i + 1,
                    "start": chunk[0].get("start", 0.0),
                    "end": chunk[-1].get("end", 0.0),
                    "segments": chunk,
                })
            return chapters
        target_chapters = 4 if total_duration < 20 * 60 else 5
        if total_duration < 10 * 60:
            target_chapters = 3
        chapter_len = max(180.0, min(600.0, total_duration / target_chapters))
        chapters: List[Dict] = []
        cur: List[Dict] = []
        cur_start = float(items[0].get("start", 0.0))
        idx = 1
        for s in items:
            cur.append(s)
            if float(s.get("end", 0.0)) - cur_start >= chapter_len:
                chapters.append({
                    "index": idx,
                    "start": cur_start,
                    "end": float(s.get("end", 0.0)),
                    "segments": cur,
                })
                idx += 1
                cur = []
                cur_start = float(s.get("end", 0.0))
        if cur:
            chapters.append({
                "index": idx,
                "start": cur_start,
                "end": float(cur[-1].get("end", 0.0)),
                "segments": cur,
            })
        return chapters

    # Builders
    def _build_minimal():
        # Minimal layout: just paragraphs
        if segments:
            paragraphs = _to_paragraphs(segments)
            for para in paragraphs:
                flow.append(Paragraph(xml_escape(para), body))
                flow.append(Spacer(1, 0.12 * inch))
        else:
            try:
                transcript_text = Path(transcript_txt_path).read_text(encoding="utf-8").strip()
            except Exception:
                transcript_text = ""
            if transcript_text:
                chunks = [c.strip() for c in re.split(r"\n\n+", transcript_text) if c.strip()]
                if not chunks:
                    chunks = [transcript_text]
                for ch in chunks:
                    flow.append(Paragraph(xml_escape(ch), body))
                    flow.append(Spacer(1, 0.12 * inch))

        # Optional sections
        snippets_dir = Path(output_dir) / "snippets" / "code"
        code_snippets = sorted(snippets_dir.glob("*.txt"))
        plots_list = classification_result.get("plots", [])
        images_list = classification_result.get("images", [])
        if code_snippets or plots_list or images_list:
            flow.append(PageBreak())
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
            if plots_list or images_list:
                flow.append(PageBreak())
        max_img_width = A4[0] - (doc.leftMargin + doc.rightMargin)
        if plots_list:
            _add_images(flow, "Plots and Diagrams", plots_list, max_img_width)
            if images_list:
                flow.append(PageBreak())
        if images_list:
            _add_images(flow, "Images", images_list, max_img_width)

    def _build_book():
        # Title page
        title_style = styles["Title"]
        flow.append(Paragraph(xml_escape(video_title or "Video Report"), title_style))
        flow.append(Spacer(1, 0.3 * inch))
        # Abstract
        h1 = styles["Heading1"]
        flow.append(Paragraph("Abstract", h1))
        flow.append(Spacer(1, 0.12 * inch))
        full_text = _normalize_text(segments) if segments else (
            Path(transcript_txt_path).read_text(encoding="utf-8").strip() if Path(transcript_txt_path).exists() else ""
        )
        abs_sentences = _split_sentences(full_text)
        abstract = _summarize(abs_sentences, max_sentences=6)
        if abstract:
            for s in abstract:
                flow.append(Paragraph(xml_escape(s), body))
                flow.append(Spacer(1, 0.08 * inch))
        else:
            flow.append(Paragraph("No abstract available.", body))
        flow.append(PageBreak())
        # Table of contents
        flow.append(Paragraph("Contents", h1))
        flow.append(Spacer(1, 0.15 * inch))
        toc = TableOfContents()
        toc.levelStyles = [
            ParagraphStyle(name='TOCHeading0', parent=body, fontSize=11, leftIndent=20, firstLineIndent=-10, spaceBefore=4, leading=13),
            ParagraphStyle(name='TOCHeading1', parent=body, fontSize=10, leftIndent=30, firstLineIndent=-10, spaceBefore=2, leading=12),
            ParagraphStyle(name='TOCHeading2', parent=body, fontSize=9,  leftIndent=40, firstLineIndent=-10, spaceBefore=0, leading=11),
        ]
        flow.append(toc)
        flow.append(PageBreak())

        # afterFlowable: capture headings for TOC and bookmarks
        heading_seq = {"n": 0}
        def after_flowable(flowable):
            if isinstance(flowable, Paragraph):
                text = flowable.getPlainText()
                style_name = getattr(flowable.style, 'name', '')
                if style_name in ("Title", "Heading1", "Heading2"):
                    level = 0 if style_name == "Title" else (1 if style_name == "Heading1" else 2)
                    key = f"h{heading_seq['n']}"
                    heading_seq['n'] += 1
                    doc.canv.bookmarkPage(key)
                    doc.canv.addOutlineEntry(text, key, level=level, closed=False)
                    doc.notify('TOCEntry', (level, text, doc.page))
        doc.afterFlowable = after_flowable

        # Chapters
        chapters = _chapterize(segments) if segments else []
        if not chapters:
            flow.append(Paragraph("Transcript", h1))
            flow.append(Spacer(1, 0.1 * inch))
            paragraphs = _to_paragraphs(segments) if segments else []
            for para in paragraphs:
                flow.append(Paragraph(xml_escape(para), body))
                flow.append(Spacer(1, 0.1 * inch))
        else:
            for ch in chapters:
                idx = ch["index"]
                # Generate a content-based chapter title
                ch_text = _normalize_text(ch["segments"]) if ch.get("segments") else ""
                # Heuristic: first meaningful sentence as chapter title,
                # falling back to a keyword-based short title
                def _chapter_title(text: str) -> str:
                    sents = _split_sentences(text)
                    for s in sents:
                        if len(s) >= 20:
                            return s[:120] + ("…" if len(s) > 120 else "")
                    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']+", text)
                    if not words:
                        return f"Chapter {idx}"
                    # pick top keywords
                    freq: Dict[str, int] = {}
                    for w in words:
                        lw = w.lower()
                        if lw in {"the","a","an","and","or","but","if","then","so","of","to","in","on","for","with","as","is","are","was","were","be","been","it","this","that","these","those","at","by","from","about","into","over","after","before","between","out","up","down","off","above","under","again","further","once"}:
                            continue
                        freq[lw] = freq.get(lw, 0) + 1
                    top = sorted(freq.items(), key=lambda t: t[1], reverse=True)[:4]
                    key = " ".join(w for w, _ in top)
                    return (key.title() if key else f"Chapter {idx}")
                title_text = _chapter_title(ch_text)
                flow.append(Paragraph(xml_escape(f"Chapter {idx}: {title_text}"), h1))
                flow.append(Spacer(1, 0.08 * inch))
                # Chapter summary
                ch_sents = _split_sentences(ch_text)
                ch_summary = _summarize(ch_sents, max_sentences=3)
                if ch_summary:
                    flow.append(Paragraph("Summary", styles["Heading2"]))
                    for s in ch_summary:
                        flow.append(Paragraph(xml_escape(s), body))
                        flow.append(Spacer(1, 0.05 * inch))
                # Chapter body as prose paragraphs (no timestamps)
                paras = _to_paragraphs(ch.get("segments", []), target_chars=900)
                for para in paras:
                    flow.append(Paragraph(xml_escape(para), body))
                    flow.append(Spacer(1, 0.08 * inch))

        # Optional sections
        snippets_dir = Path(output_dir) / "snippets" / "code"
        code_snippets = sorted(snippets_dir.glob("*.txt"))
        plots_list = classification_result.get("plots", [])
        images_list = classification_result.get("images", [])
        if code_snippets or plots_list or images_list:
            flow.append(PageBreak())
        if code_snippets:
            flow.append(Paragraph("Extracted Code", styles["Heading1"]))
            flow.append(Spacer(1, 0.12 * inch))
            for snip in code_snippets:
                try:
                    txt = snip.read_text(encoding="utf-8")
                    flow.append(Preformatted(txt, code_style, dedent=0))
                    flow.append(Spacer(1, 0.12 * inch))
                except Exception:
                    continue
        max_img_width = A4[0] - (doc.leftMargin + doc.rightMargin)
        if plots_list:
            flow.append(PageBreak())
            _add_images(flow, "Plots and Diagrams", plots_list, max_img_width)
        if images_list:
            flow.append(PageBreak())
            _add_images(flow, "Images", images_list, max_img_width)

    # Dispatch
    if report_style == "book":
        _build_book()
    else:
        _build_minimal()

    if progress_cb:
        progress_cb(50.0)
    doc.build(flow)
    if progress_cb:
        progress_cb(100.0)
    return output_pdf_path

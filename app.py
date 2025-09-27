from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from src.video_doc.progress import PipelineProgress, make_console_progress_printer
from src.video_doc.stream import (
    resolve_stream_urls,
    stream_extract_audio,
    stream_extract_keyframes,
    fallback_download_audio_via_ytdlp,
    fallback_download_small_video,
)
from src.video_doc.download import download_video
from src.video_doc.audio import extract_audio_wav
from src.video_doc.transcribe import transcribe_audio
from src.video_doc.frames import extract_keyframes, build_contact_sheet
from src.video_doc.classify import classify_frames
from src.video_doc.pdf import build_pdf_report


st.set_page_config(page_title="Longform Video → PDF", layout="wide")


def _ensure_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transcript").mkdir(parents=True, exist_ok=True)
    (output_dir / "frames" / "keyframes").mkdir(parents=True, exist_ok=True)
    (output_dir / "classified" / "code").mkdir(parents=True, exist_ok=True)
    (output_dir / "classified" / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "classified" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "snippets" / "code").mkdir(parents=True, exist_ok=True)


def _progress_cb_factory(label: str):
    ph = st.empty()

    def _cb(p: float) -> None:
        try:
            ph.progress(int(max(0.0, min(100.0, p))), text=label)
        except Exception:
            pass

    return _cb


def _load_segments(path: Path) -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [s for s in data if isinstance(s, dict) and {"start", "end", "text"} <= set(s.keys())]
    except Exception:
        pass
    return []


def _save_segments(path: Path, segs: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(segs, f, ensure_ascii=False, indent=2)


st.title("Longform Video → PDF (Interactive)")
st.caption("Review transcript, pick visuals, and export a clean PDF")

with st.sidebar:
    st.header("Input")
    url = st.text_input("Video URL (YouTube, etc.)", value="")
    local_video = st.file_uploader("Or upload a local video", type=["mp4", "mkv", "mov", "webm", "mp3", "m4a", "wav", "flac"])  # type: ignore[arg-type]
    out_dir_str = st.text_input("Output directory", value=str(Path("outputs") / "run_ui"))
    out_dir = Path(out_dir_str)
    colA, colB = st.columns(2)
    with colA:
        pipeline_mode = st.selectbox("Pipeline", ["transcribe-only", "full"], index=0)
    with colB:
        report_style = st.selectbox("Report style", ["book", "minimal"], index=0)

    st.divider()
    st.header("Transcription")
    language = st.text_input("Language (code or 'auto')", value="auto")
    model_size = st.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large-v3"], index=3)
    beam_size = st.slider("Beam size", 1, 10, 5)
    export_srt = st.checkbox("Export SRT", value=True)

    st.divider()
    st.header("Keyframes")
    transcribe_only = st.checkbox("Transcript only (skip visuals)", value=(pipeline_mode == "transcribe-only"))
    kf_method = st.selectbox("Method", ["scene", "iframe", "interval"], index=0)
    max_fps = st.slider("Max FPS (scene)", 0.05, 3.0, 1.0, 0.05)
    min_scene_diff = st.slider("Min scene diff", 0.1, 1.0, 0.45, 0.05)
    kf_interval_sec = st.slider("Interval seconds", 1, 60, 5)
    frame_format = st.selectbox("Frame format", ["jpg", "png", "webp"], index=0)
    frame_quality = st.slider("Frame quality", 10, 100, 90)
    frame_max_width = st.number_input("Frame max width", value=1280)
    frame_max_frames = st.number_input("Frame max frames (0=all)", value=0)
    skip_dark = st.checkbox("Skip mostly dark frames", value=False)
    dark_value = st.slider("Dark pixel value", 0, 255, 16)
    dark_ratio = st.slider("Dark pixel ratio", 0.5, 1.0, 0.98, 0.01)
    dedupe = st.checkbox("Dedupe near-duplicates", value=False)
    dedupe_sim = st.slider("Dedupe similarity (corr)", 0.90, 0.999, 0.995, 0.001)
    contact_sheet = st.checkbox("Generate contact sheet", value=True)

    st.divider()
    st.header("Classification")
    ocr_langs = st.text_input("OCR languages (comma separated)", value="en")
    skip_blurry = st.checkbox("Skip blurry frames", value=True)
    blurry_threshold = st.slider("Blurry threshold", 10.0, 200.0, 60.0, 1.0)
    max_per_category = st.number_input("Max per category (0=all)", value=0)

    st.divider()
    run_btn = st.button("Run / Update", type="primary")


_ensure_dirs(out_dir)
transcript_txt = out_dir / "transcript" / "transcript.txt"
segments_json = out_dir / "transcript" / "segments.json"
frames_dir = out_dir / "frames" / "keyframes"
classified_dir = out_dir / "classified"
snippets_dir = out_dir / "snippets" / "code"
report_pdf_path = out_dir / "report.pdf"


def _write_transcript_txt_from_segments(segs: List[Dict]) -> None:
    lines: List[str] = []
    for s in segs:
        try:
            start = float(s.get("start", 0.0))
            end = float(s.get("end", 0.0))
            text = str(s.get("text", "")).strip()
            lines.append(f"[{start:7.2f} -> {end:7.2f}] {text}")
        except Exception:
            continue
    transcript_txt.parent.mkdir(parents=True, exist_ok=True)
    transcript_txt.write_text("\n".join(lines), encoding="utf-8")


left, right = st.columns([0.6, 0.4])

with left:
    st.subheader("Transcript Editor")
    segs_state_key = f"segs::{out_dir_str}"
    if segs_state_key not in st.session_state:
        st.session_state[segs_state_key] = _load_segments(segments_json)
    editable = st.data_editor(
        st.session_state[segs_state_key] or [],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "start": st.column_config.NumberColumn("Start", format="%.2f"),
            "end": st.column_config.NumberColumn("End", format="%.2f"),
            "text": st.column_config.TextColumn("Text"),
        },
        key=segs_state_key,
    )
    colx, coly, colz = st.columns(3)
    with colx:
        if st.button("Save transcript edits"):
            try:
                _save_segments(segments_json, editable)
                _write_transcript_txt_from_segments(editable)
                st.success("Transcript saved.")
            except Exception as e:
                st.error(f"Failed to save: {e}")
    with coly:
        if st.button("Load existing transcript"):
            st.session_state[segs_state_key] = _load_segments(segments_json)
            st.rerun()
    with colz:
        if st.button("Clear transcript"):
            st.session_state[segs_state_key] = []
            _save_segments(segments_json, [])
            _write_transcript_txt_from_segments([])
            st.rerun()

with right:
    st.subheader("Keyframes & Preview")
    grid = st.container()
    if frames_dir.exists():
        imgs = sorted(frames_dir.glob("frame_*.jpg")) + sorted(frames_dir.glob("frame_*.png")) + sorted(frames_dir.glob("frame_*.webp"))
    else:
        imgs = []
    if imgs:
        cols = st.columns(4)
        for idx, img in enumerate(imgs[:40]):
            with cols[idx % 4]:
                st.image(str(img), use_container_width=True)
    else:
        st.info("No keyframes yet.")

    st.subheader("PDF Preview")
    if report_pdf_path.exists():
        try:
            data = report_pdf_path.read_bytes()
            st.download_button("Download PDF", data=data, file_name=report_pdf_path.name, mime="application/pdf")
            st.components.v1.html(
                f"""
                <iframe src="data:application/pdf;base64,{data.hex()}" width="100%" height="600px"></iframe>
                """,
                height=620,
            )
        except Exception:
            st.info("PDF available. Use Download to view.")
    else:
        st.caption("PDF will appear here after build")


if run_btn:
    try:
        _ensure_dirs(out_dir)

        # Determine video source
        video_path: Optional[Path]
        if local_video is not None:
            dst = out_dir / "video.mp4"
            with open(dst, "wb") as f:
                f.write(local_video.getbuffer())
            video_path = dst
        else:
            video_path = None

        audio_path = out_dir / "audio.wav"
        progress_printer = make_console_progress_printer()
        pp = PipelineProgress(total_weight=100.0, on_change=lambda p: progress_printer(p))

        # Acquire audio and keyframes
        if video_path is None and url:
            # Streaming path
            resolved = resolve_stream_urls(url)
            if resolved.get("audio_url"):
                st.write("Extracting audio (stream)...")
                stream_extract_audio(resolved["audio_url"], audio_path, headers=resolved.get("headers"), progress_cb=_progress_cb_factory("Audio"))
            else:
                st.write("Audio stream not available; downloading audio")
                fallback_download_audio_via_ytdlp(url, audio_path)
            if not transcribe_only:
                if resolved.get("video_url"):
                    st.write("Extracting keyframes (stream)...")
                    stream_extract_keyframes(
                        resolved["video_url"],
                        frames_dir,
                        max_fps=float(max_fps),
                        scene_threshold=float(min_scene_diff),
                        headers=resolved.get("headers"),
                        output_format=str(frame_format),
                        jpeg_quality=int(frame_quality),
                        max_width=int(frame_max_width),
                        max_frames=(int(frame_max_frames) if int(frame_max_frames) > 0 else None),
                        progress_cb=_progress_cb_factory("Keyframes"),
                    )
                else:
                    st.write("Video stream not available; downloading tiny video for frames")
                    small_video = out_dir / "video_small.mp4"
                    fallback_download_small_video(url, small_video)
                    extract_keyframes(
                        video_path=small_video,
                        output_dir=frames_dir,
                        max_fps=float(max_fps),
                        scene_threshold=float(min_scene_diff),
                        method=str(kf_method),
                        interval_sec=float(kf_interval_sec),
                        output_format=str(frame_format),
                        jpeg_quality=int(frame_quality),
                        max_width=int(frame_max_width),
                        max_frames=(int(frame_max_frames) if int(frame_max_frames) > 0 else None),
                        skip_dark=bool(skip_dark),
                        dark_pixel_value=int(dark_value),
                        dark_ratio_threshold=float(dark_ratio),
                        dedupe=bool(dedupe),
                        dedupe_similarity=float(dedupe_sim),
                        progress_cb=_progress_cb_factory("Keyframes"),
                    )
        else:
            # Download or reuse local file
            if video_path is None and url:
                st.write("Downloading full video...")
                video_path = out_dir / "video.mp4"
                download_video(url, video_path, progress_cb=_progress_cb_factory("Download"))
            if video_path is not None:
                st.write("Extracting audio...")
                extract_audio_wav(video_path, audio_path, progress_cb=_progress_cb_factory("Audio"))
                if not transcribe_only:
                    st.write("Extracting keyframes...")
                    extract_keyframes(
                        video_path=video_path,
                        output_dir=frames_dir,
                        max_fps=float(max_fps),
                        scene_threshold=float(min_scene_diff),
                        method=str(kf_method),
                        interval_sec=float(kf_interval_sec),
                        output_format=str(frame_format),
                        jpeg_quality=int(frame_quality),
                        max_width=int(frame_max_width),
                        max_frames=(int(frame_max_frames) if int(frame_max_frames) > 0 else None),
                        skip_dark=bool(skip_dark),
                        dark_pixel_value=int(dark_value),
                        dark_ratio_threshold=float(dark_ratio),
                        dedupe=bool(dedupe),
                        dedupe_similarity=float(dedupe_sim),
                        progress_cb=_progress_cb_factory("Keyframes"),
                    )

        # Transcribe
        st.write("Transcribing...")
        segs = transcribe_audio(
            audio_path=audio_path,
            transcript_txt_path=transcript_txt,
            segments_json_path=segments_json,
            language=str(language),
            beam_size=int(beam_size),
            model_size=str(model_size),
            progress_cb=_progress_cb_factory("Transcribe"),
            srt_path=(transcript_txt.parent / "transcript.srt") if bool(export_srt) else None,
        )
        # Merge any in-editor edits
        if st.session_state.get(segs_state_key):
            edits = st.session_state[segs_state_key]
            if isinstance(edits, list) and len(edits) > 0:
                _save_segments(segments_json, edits)
                _write_transcript_txt_from_segments(edits)

        # Classify
        if not transcribe_only:
            st.write("Classifying visuals...")
            classification_result = classify_frames(
                frame_paths=sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.webp")),
                classified_root=classified_dir,
                snippets_dir=snippets_dir,
                progress_cb=_progress_cb_factory("Classify"),
                ocr_languages=[l.strip() for l in str(ocr_langs).split(',') if l.strip()],
                skip_blurry=bool(skip_blurry),
                blurry_threshold=float(blurry_threshold),
                max_per_category=(int(max_per_category) if int(max_per_category) > 0 else None),
            )
        else:
            classification_result = {"code": [], "plots": [], "images": []}

        # Contact sheet (optional)
        contact_sheet_path = None
        if not transcribe_only and bool(contact_sheet):
            all_imgs = (
                sorted(frames_dir.glob("*.jpg")) +
                sorted(frames_dir.glob("*.png")) +
                sorted(frames_dir.glob("*.webp"))
            )
            if all_imgs:
                st.write("Building contact sheet...")
                cs_path = frames_dir.parent / "contact_sheet.jpg"
                try:
                    build_contact_sheet(all_imgs, cs_path)
                    contact_sheet_path = cs_path
                except Exception:
                    contact_sheet_path = None

        # Build PDF
        st.write("Building PDF...")
        build_pdf_report(
            output_pdf_path=report_pdf_path,
            transcript_txt_path=transcript_txt,
            segments_json_path=segments_json,
            classification_result=classification_result,
            output_dir=out_dir,
            progress_cb=_progress_cb_factory("PDF"),
            report_style=str(report_style),
            video_title="Video Report",
            contact_sheet_path=contact_sheet_path,
        )
        st.success(f"Report ready: {report_pdf_path}")
        st.rerun()
    except Exception as e:
        st.error(f"Run failed: {e}")



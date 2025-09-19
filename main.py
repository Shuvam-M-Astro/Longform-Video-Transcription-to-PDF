import argparse
import json
import os
import shutil
from pathlib import Path

from src.video_doc.download import download_video
from src.video_doc.audio import extract_audio_wav
from src.video_doc.transcribe import transcribe_audio
from src.video_doc.frames import extract_keyframes, build_contact_sheet
from src.video_doc.classify import classify_frames
from src.video_doc.pdf import build_pdf_report
from src.video_doc.stream import (
    resolve_stream_urls,
    stream_extract_audio,
    stream_extract_keyframes,
    fallback_download_audio_via_ytdlp,
    fallback_download_small_video,
)
from src.video_doc.progress import PipelineProgress, make_console_progress_printer
from yt_dlp import YoutubeDL


def ensure_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transcript").mkdir(parents=True, exist_ok=True)
    (output_dir / "frames" / "keyframes").mkdir(parents=True, exist_ok=True)
    (output_dir / "classified" / "code").mkdir(parents=True, exist_ok=True)
    (output_dir / "classified" / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "classified" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "snippets" / "code").mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PDF document from a long-form video")
    parser.add_argument("--url", type=str, required=False, help="Video URL")
    parser.add_argument("--video", type=str, default=None, help="Path to a local video file (e.g., .mp4, .mkv)")
    parser.add_argument("--out", type=str, default="./outputs/run", help="Output directory")
    parser.add_argument("--language", type=str, default="auto", help="Language code or 'auto'")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--whisper-model", type=str, default="medium", help="faster-whisper model size")
    parser.add_argument("--whisper-cpu-threads", type=int, default=0, help="Override faster-whisper CPU threads; 0=auto")
    parser.add_argument("--whisper-num-workers", type=int, default=1, help="Number of CPU decoder workers (>=1)")
    parser.add_argument("--max-fps", type=float, default=1.0, help="Max FPS for keyframe detection (scene method)")
    parser.add_argument("--min-scene-diff", type=float, default=0.45, help="Scene change threshold [0-1]")
    parser.add_argument("--kf-method", type=str, choices=["scene", "iframe", "interval"], default="scene", help="Keyframe extraction method")
    parser.add_argument("--kf-interval-sec", type=float, default=5.0, help="Interval seconds for interval method")
    # New keyframe output options
    parser.add_argument("--frame-format", type=str, choices=["jpg", "png", "webp"], default="jpg", help="Output image format for frames")
    parser.add_argument("--frame-quality", type=int, default=90, help="Quality for JPG/WEBP (1-100)")
    parser.add_argument("--frame-max-width", type=int, default=1280, help="Resize frames to max width (pixels); <=0 to disable")
    parser.add_argument("--frame-max-frames", type=int, default=0, help="Cap total saved frames; 0 to disable")
    parser.add_argument("--skip-dark", action="store_true", help="Skip mostly dark frames (OpenCV path)")
    parser.add_argument("--dark-value", type=int, default=16, help="Dark pixel V threshold [0-255]")
    parser.add_argument("--dark-ratio", type=float, default=0.98, help="Dark pixel ratio threshold [0-1]")
    parser.add_argument("--dedupe", action="store_true", help="Skip near-duplicate frames by histogram similarity (OpenCV path)")
    parser.add_argument("--dedupe-sim", type=float, default=0.995, help="Histogram correlation threshold to treat as duplicate [0-1]")
    # Contact sheet
    parser.add_argument("--contact-sheet", action="store_true", help="Generate a contact sheet of keyframes")
    parser.add_argument("--cs-columns", type=int, default=6, help="Contact sheet columns")
    parser.add_argument("--cs-thumb-width", type=int, default=300, help="Contact sheet thumbnail width (px)")
    parser.add_argument("--cs-padding", type=int, default=8, help="Contact sheet padding (px)")
    parser.add_argument("--cs-title", type=str, default=None, help="Contact sheet title text")
    parser.add_argument("--cs-title-height", type=int, default=40, help="Contact sheet title area height (px)")
    # Pipeline mode and IO
    parser.add_argument("--skip-download", action="store_true", help="Skip download if video exists")
    parser.add_argument("--streaming", action="store_true", help="Process via streaming without saving full video")
    parser.add_argument("--pipeline-mode", type=str, choices=["transcribe-only", "full"], default="transcribe-only", help="Pipeline mode")
    parser.add_argument("--transcribe-only", action="store_true", help="Skip frames/classification; transcript-only PDF (compat)")
    parser.add_argument("--export-srt", action="store_true", help="Also export transcript in SubRip (.srt)")
    parser.add_argument("--resume", action="store_true", help="Resume: do not clean output, reuse existing artifacts")
    # Download auth/bypass options
    parser.add_argument("--cookies-from-browser", type=str, default=None, help="Browser to read cookies from (chrome|edge|brave|firefox)")
    parser.add_argument("--browser-profile", type=str, default=None, help="Specific browser profile name, e.g., 'Default' or 'Profile 1'")
    parser.add_argument("--cookies-file", type=str, default=None, help="Path to cookies.txt file")
    parser.add_argument("--use-android-client", action="store_true", help="Use YouTube Android client fallback")
    parser.add_argument("--report-style", type=str, choices=["minimal", "book"], default="book", help="PDF layout style")
    # Audio extraction controls
    parser.add_argument("--trim-start", type=float, default=0.0, help="Trim start seconds of audio")
    parser.add_argument("--trim-end", type=float, default=0.0, help="Trim end seconds of audio (from end); 0 for none")
    parser.add_argument("--volume-gain", type=float, default=0.0, help="Audio volume gain in dB (0 for none)")
    # Classification controls
    parser.add_argument("--ocr-langs", type=str, default="en", help="Comma-separated OCR languages, e.g., 'en,fr'")
    parser.add_argument("--skip-blurry", action="store_true", help="Skip classifying frames detected as blurry")
    parser.add_argument("--blurry-threshold", type=float, default=60.0, help="Blurriness threshold (variance of Laplacian)")
    parser.add_argument("--max-per-category", type=int, default=0, help="Max items saved per category; 0 for unlimited")
    args = parser.parse_args()
    if not args.url and not args.video:
        parser.error("You must provide either --url or --video")
    return args


def main() -> None:
    args = parse_args()
    # Pipeline mode: transcribe-only (default) or full (compat flag supported)
    args.transcribe_only = args.transcribe_only or (args.pipeline_mode == "transcribe-only")
    # Local file implies no streaming mode
    if getattr(args, "video", None):
        args.streaming = False
    output_dir = Path(args.out)
    # Clean output directory before running, unless resume is requested
    if output_dir.exists() and not getattr(args, "resume", False):
        print(f"Cleaning output directory: {output_dir}", flush=True)
        shutil.rmtree(output_dir, ignore_errors=True)
    ensure_dirs(output_dir)

    # Try to extract a human-readable video title for the report
    def _extract_video_title() -> str:
        # Prefer local filename when provided
        if getattr(args, "video", None):
            try:
                stem = Path(args.video).stem
                title = stem.replace("-", " ").replace("_", " ").strip()
                return title.title() if title else "Video Report"
            except Exception:
                return "Video Report"
        def try_opts(use_cookies: bool, use_android: bool):
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
            }
            if use_cookies and args.cookies_from_browser:
                tup = (args.cookies_from_browser,) if not args.browser_profile else (args.cookies_from_browser, args.browser_profile)
                ydl_opts["cookiesfrombrowser"] = tup
            if use_cookies and args.cookies_file:
                ydl_opts["cookiefile"] = str(Path(args.cookies_file))
            if use_android:
                ydl_opts["extractor_args"] = {"youtube": {"player_client": ["android"]}}
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(args.url, download=False)
                return str(info.get("title") or "")
        # Attempt with cookies and current client
        try:
            title = try_opts(use_cookies=True, use_android=args.use_android_client)
            if title:
                return title
        except Exception:
            pass
        # Attempt without cookies
        try:
            title = try_opts(use_cookies=False, use_android=args.use_android_client)
            if title:
                return title
        except Exception:
            pass
        # Attempt Android client without cookies
        try:
            title = try_opts(use_cookies=False, use_android=True)
            if title:
                return title
        except Exception:
            pass
        # Fallback: derive from URL slug
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(args.url)
            if parsed.netloc.lower().endswith("youtube.com"):
                vid = parse_qs(parsed.query).get("v", [""])[0]
                if vid:
                    return f"YouTube Video {vid}"
            if parsed.path:
                slug = Path(parsed.path).name.replace("-", " ").replace("_", " ")
                slug = slug.strip() or "Video Report"
                return slug.title()
        except Exception:
            pass
        return "Video Report"

    video_title = _extract_video_title()

    # Determine source video path
    if getattr(args, "video", None):
        src_video_path = Path(args.video)
        if not src_video_path.exists():
            raise FileNotFoundError(f"Local video not found: {src_video_path}")
        video_path = src_video_path
    else:
        video_path = output_dir / "video.mp4"
    audio_path = output_dir / "audio.wav"
    transcript_dir = output_dir / "transcript"
    frames_dir = output_dir / "frames" / "keyframes"
    classified_dir = output_dir / "classified"
    snippets_dir = output_dir / "snippets" / "code"

    print(
        f"Mode: {'streaming' if args.streaming else ('local-file' if getattr(args, 'video', None) else 'download')}; "
        f"transcribe_only={args.transcribe_only}",
        flush=True,
    )

    # Overall pipeline progress (weights roughly proportional to average time)
    # Download 20, Audio 10, Frames 20, Transcribe 40, Classify 5, PDF 5
    progress = PipelineProgress(
        total_weight=100.0,
        on_change=make_console_progress_printer(),
    )

    keyframe_paths = []

    if args.streaming:
        print("Resolving stream URLs...", flush=True)
        resolved = resolve_stream_urls(
            args.url,
            cookies_from_browser=args.cookies_from_browser,
            browser_profile=args.browser_profile,
            cookies_file=Path(args.cookies_file) if args.cookies_file else None,
            use_android_client=args.use_android_client,
        )
        # Audio
        try:
            if not resolved.get("audio_url"):
                raise RuntimeError("No audio_url")
            print("Extracting audio from stream...", flush=True)
            progress.start_step("Audio (stream)", 10)
            stream_extract_audio(resolved["audio_url"], audio_path, headers=resolved.get("headers"), progress_cb=lambda p: progress.update(p))
            progress.end_step()
            print(f"Audio saved: {audio_path}", flush=True)
        except Exception:
            print("Stream audio failed; falling back to yt-dlp audio-only...", flush=True)
            progress.start_step("Audio fallback download", 10)
            audio_path = fallback_download_audio_via_ytdlp(
                args.url,
                audio_path,
                cookies_from_browser=args.cookies_from_browser,
                browser_profile=args.browser_profile,
                cookies_file=Path(args.cookies_file) if args.cookies_file else None,
                use_android_client=args.use_android_client,
            )
            progress.end_step()
            print(f"Audio saved (fallback): {audio_path}", flush=True)
        # Frames (skip if transcribe-only)
        if not args.transcribe_only:
            try:
                if not resolved.get("video_url"):
                    raise RuntimeError("No video_url")
                print("Extracting keyframes from stream...", flush=True)
                progress.start_step("Keyframes (stream)", 20)
                keyframe_paths = stream_extract_keyframes(
                    resolved["video_url"],
                    frames_dir,
                    max_fps=args.max_fps,
                    scene_threshold=args.min_scene_diff,
                    headers=resolved.get("headers"),
                    output_format=args.frame_format,
                    jpeg_quality=args.frame_quality,
                    max_width=args.frame_max_width,
                    max_frames=(args.frame_max_frames if args.frame_max_frames > 0 else None),
                    progress_cb=lambda p: progress.update(p),
                )
                progress.end_step()
                print(f"Keyframes saved: {len(keyframe_paths)}", flush=True)
            except Exception:
                print("Stream keyframes failed; downloading tiny MP4 for keyframes...", flush=True)
                small_video = output_dir / "video_small.mp4"
                progress.start_step("Download tiny video", 10)
                fallback_download_small_video(
                    args.url,
                    small_video,
                    cookies_from_browser=args.cookies_from_browser,
                    browser_profile=args.browser_profile,
                    cookies_file=Path(args.cookies_file) if args.cookies_file else None,
                    use_android_client=args.use_android_client,
                )
                progress.end_step()
                print("Extracting keyframes from tiny MP4...", flush=True)
                progress.start_step("Keyframes (tiny)", 20)
                keyframe_paths = extract_keyframes(
                    video_path=small_video,
                    output_dir=frames_dir,
                    max_fps=args.max_fps,
                    scene_threshold=args.min_scene_diff,
                    method=args.kf_method,
                    interval_sec=args.kf_interval_sec,
                    progress_cb=lambda p: progress.update(p),
                )
                progress.end_step()
                print(f"Keyframes saved: {len(keyframe_paths)}", flush=True)
    else:
        if getattr(args, "video", None):
            print(f"Using local video file: {video_path}", flush=True)
        else:
            if not args.skip_download and video_path.exists() and getattr(args, "resume", False):
                print(f"Reusing existing video: {video_path}", flush=True)
            elif not args.skip_download or not video_path.exists():
                print("Downloading full video (mp4)...", flush=True)
                progress.start_step("Download video", 20)
                download_video(
                    args.url,
                    video_path,
                    cookies_from_browser=args.cookies_from_browser,
                    browser_profile=args.browser_profile,
                    cookies_file=Path(args.cookies_file) if args.cookies_file else None,
                    use_android_client=args.use_android_client,
                    progress_cb=lambda p: progress.update(p),
                )
                progress.end_step()
        if audio_path.exists() and getattr(args, "resume", False):
            print(f"Reusing existing audio: {audio_path}", flush=True)
        else:
            print("Extracting audio from file...", flush=True)
            progress.start_step("Extract audio", 10)
            extract_audio_wav(
                video_path,
                audio_path,
                progress_cb=lambda p: progress.update(p),
                start_time=args.trim_start,
                end_trim=args.trim_end,
                volume_gain_db=args.volume_gain,
            )
            progress.end_step()
            print(f"Audio saved: {audio_path}", flush=True)
        if not args.transcribe_only:
            print("Extracting keyframes from file...", flush=True)
            print(
                f"  - method={args.kf_method} max_fps={args.max_fps} "
                f"min_scene_diff={args.min_scene_diff} interval_sec={args.kf_interval_sec}",
                flush=True,
            )
            existing_frames = sorted(frames_dir.glob("frame_*.jpg"))
            if existing_frames and getattr(args, "resume", False):
                keyframe_paths = existing_frames
                print(f"Reusing existing keyframes: {len(keyframe_paths)}", flush=True)
            else:
                progress.start_step("Keyframes", 20)
                keyframe_paths = extract_keyframes(
                    video_path=video_path,
                    output_dir=frames_dir,
                    max_fps=args.max_fps,
                    scene_threshold=args.min_scene_diff,
                    method=args.kf_method,
                    interval_sec=args.kf_interval_sec,
                    output_format=args.frame_format,
                    jpeg_quality=args.frame_quality,
                    max_width=args.frame_max_width,
                    max_frames=(args.frame_max_frames if args.frame_max_frames > 0 else None),
                    skip_dark=args.skip_dark,
                    dark_pixel_value=args.dark_value,
                    dark_ratio_threshold=args.dark_ratio,
                    dedupe=args.dedupe,
                    dedupe_similarity=args.dedupe_sim,
                    progress_cb=lambda p: progress.update(p),
                )
                progress.end_step()
                print(f"Keyframes saved: {len(keyframe_paths)}", flush=True)
            # Contact sheet
            if args.contact_sheet and keyframe_paths:
                print("Building contact sheet...", flush=True)
                cs_path = frames_dir.parent / "contact_sheet.jpg"
                try:
                    if cs_path.exists() and getattr(args, "resume", False):
                        print(f"Reusing existing contact sheet: {cs_path}", flush=True)
                    else:
                        build_contact_sheet(
                            image_paths=keyframe_paths,
                            output_path=cs_path,
                            columns=args.cs_columns,
                            thumb_width=args.cs_thumb_width,
                            padding=args.cs_padding,
                            title=(args.cs_title or video_title),
                            title_height=args.cs_title_height,
                        )
                        print(f"Contact sheet saved: {cs_path}", flush=True)
                except Exception as e:
                    print(f"Contact sheet failed: {e}", flush=True)

    transcript_txt = transcript_dir / "transcript.txt"
    segments_json = transcript_dir / "segments.json"
    srt_path = transcript_dir / "transcript.srt" if args.export_srt else None
    if getattr(args, "resume", False) and transcript_txt.exists() and segments_json.exists() and (not args.export_srt or (srt_path and srt_path.exists())):
        print("Reusing existing transcript artifacts", flush=True)
        try:
            with open(segments_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            segments = [
                type("TS", (), {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": str(s.get("text", ""))})
                for s in data if isinstance(s, dict)
            ]
        except Exception:
            segments = []
    else:
        print(f"Transcribing audio with model={args.whisper_model}, beam_size={args.beam_size}...", flush=True)
        progress.start_step("Transcribe", 40)
        segments = transcribe_audio(
            audio_path=audio_path,
            transcript_txt_path=transcript_txt,
            segments_json_path=segments_json,
            language=args.language,
            beam_size=args.beam_size,
            model_size=args.whisper_model,
            progress_cb=lambda p: progress.update(p),
            cpu_threads=(args.whisper_cpu_threads if args.whisper_cpu_threads and args.whisper_cpu_threads > 0 else None),
            num_workers=(args.whisper_num_workers if args.whisper_num_workers and args.whisper_num_workers > 0 else None),
            srt_path=srt_path,
        )
        progress.end_step()
        print(f"Transcription done: {len(segments)} segments", flush=True)

    if args.transcribe_only:
        classification_result = {"code": [], "plots": [], "images": []}
    else:
        print("Classifying frames...", flush=True)
        manifest_path = classified_dir / "classification_result.json"
        if getattr(args, "resume", False) and manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    classification_result = json.load(f)
                # Normalize to lists of Paths
                for k in ("code", "plots", "images"):
                    classification_result[k] = [Path(p) for p in classification_result.get(k, [])]
                print("Reusing existing classification manifest", flush=True)
            except Exception:
                classification_result = {"code": [], "plots": [], "images": []}
        else:
            progress.start_step("Classify frames", 5)
            classification_result = classify_frames(
                frame_paths=keyframe_paths,
                classified_root=classified_dir,
                snippets_dir=snippets_dir,
                progress_cb=lambda p: progress.update(p),
                ocr_languages=[lang.strip() for lang in str(args.ocr_langs).split(',') if lang.strip()],
                skip_blurry=args.skip_blurry,
                blurry_threshold=args.blurry_threshold,
                max_per_category=(args.max_per_category if args.max_per_category > 0 else None),
            )
            progress.end_step()
            # Write manifest
            try:
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump({k: [str(p) for p in v] for k, v in classification_result.items()}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            print(
                f"Classified - code: {len(classification_result.get('code', []))}, "
                f"plots: {len(classification_result.get('plots', []))}, "
                f"images: {len(classification_result.get('images', []))}",
                flush=True,
            )

    print("Building PDF report...", flush=True)
    progress.start_step("Build PDF", 5)
    report_pdf_path = output_dir / "report.pdf"
    # Optionally detect contact sheet path to embed later
    contact_sheet_path = None
    try:
        possible_cs = (frames_dir.parent / "contact_sheet.jpg")
        contact_sheet_path = possible_cs if possible_cs.exists() else None
    except Exception:
        contact_sheet_path = None

    if getattr(args, "resume", False) and report_pdf_path.exists():
        print(f"Reusing existing report: {report_pdf_path}", flush=True)
        progress.update(100.0)
        progress.end_step()
    else:
        build_pdf_report(
            output_pdf_path=report_pdf_path,
            transcript_txt_path=transcript_txt,
            segments_json_path=segments_json,
            classification_result=classification_result,
            output_dir=output_dir,
            progress_cb=lambda p: progress.update(p),
            report_style=args.report_style,
            video_title=video_title,
            contact_sheet_path=contact_sheet_path,
        )
        progress.end_step()
        print(f"Report ready: {report_pdf_path}", flush=True)

    print(json.dumps({
        "audio": str(audio_path),
        "transcript_txt": str(transcript_txt),
        "segments_json": str(segments_json),
        "keyframes": len(keyframe_paths),
        "classified": {k: len(v) for k, v in classification_result.items()},
        "report_pdf": str(report_pdf_path),
        "mode": ("streaming" if args.streaming else ("local-file" if getattr(args, "video", None) else "download")),
        "transcribe_only": args.transcribe_only,
    }, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
import os
from pathlib import Path

from src.video_doc.download import download_video
from src.video_doc.audio import extract_audio_wav
from src.video_doc.transcribe import transcribe_audio
from src.video_doc.frames import extract_keyframes
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
    parser.add_argument("--url", type=str, required=True, help="Video URL")
    parser.add_argument("--out", type=str, default="./outputs/run", help="Output directory")
    parser.add_argument("--language", type=str, default="auto", help="Language code or 'auto'")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--whisper-model", type=str, default="medium", help="faster-whisper model size")
    parser.add_argument("--max-fps", type=float, default=1.0, help="Max FPS for keyframe detection (scene method)")
    parser.add_argument("--min-scene-diff", type=float, default=0.45, help="Scene change threshold [0-1]")
    parser.add_argument("--kf-method", type=str, choices=["scene", "iframe", "interval"], default="scene", help="Keyframe extraction method")
    parser.add_argument("--kf-interval-sec", type=float, default=5.0, help="Interval seconds for interval method")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if video exists")
    parser.add_argument("--streaming", action="store_true", help="Process via streaming without saving full video")
    parser.add_argument("--transcribe-only", action="store_true", help="Skip frames/classification; transcript-only PDF")
    # Download auth/bypass options
    parser.add_argument("--cookies-from-browser", type=str, default=None, help="Browser to read cookies from (chrome|edge|brave|firefox)")
    parser.add_argument("--browser-profile", type=str, default=None, help="Specific browser profile name, e.g., 'Default' or 'Profile 1'")
    parser.add_argument("--cookies-file", type=str, default=None, help="Path to cookies.txt file")
    parser.add_argument("--use-android-client", action="store_true", help="Use YouTube Android client fallback")
    parser.add_argument("--report-style", type=str, choices=["minimal", "book"], default="minimal", help="PDF layout style")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Force simple 3-step flow: download/stream -> transcribe -> build report
    # Always run in transcribe-only mode per user request
    args.transcribe_only = True
    output_dir = Path(args.out)
    ensure_dirs(output_dir)

    video_path = output_dir / "video.mp4"
    audio_path = output_dir / "audio.wav"
    transcript_dir = output_dir / "transcript"
    frames_dir = output_dir / "frames" / "keyframes"
    classified_dir = output_dir / "classified"
    snippets_dir = output_dir / "snippets" / "code"

    print(f"Mode: {'streaming' if args.streaming else 'download'}; transcribe_only={args.transcribe_only}", flush=True)

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
                stream_extract_keyframes(
                    resolved["video_url"],
                    frames_dir,
                    max_fps=args.max_fps,
                    scene_threshold=args.min_scene_diff,
                    headers=resolved.get("headers"),
                    progress_cb=lambda p: progress.update(p),
                )
                progress.end_step()
                keyframe_paths = sorted(frames_dir.glob("frame_*.jpg"))
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
        if not args.skip_download or not video_path.exists():
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
        print("Extracting audio from file...", flush=True)
        progress.start_step("Extract audio", 10)
        extract_audio_wav(video_path, audio_path, progress_cb=lambda p: progress.update(p))
        progress.end_step()
        print(f"Audio saved: {audio_path}", flush=True)
        if not args.transcribe_only:
            print("Extracting keyframes from file...", flush=True)
            print(
                f"  - method={args.kf_method} max_fps={args.max_fps} "
                f"min_scene_diff={args.min_scene_diff} interval_sec={args.kf_interval_sec}",
                flush=True,
            )
            progress.start_step("Keyframes", 20)
            keyframe_paths = extract_keyframes(
                video_path=video_path,
                output_dir=frames_dir,
                max_fps=args.max_fps,
                scene_threshold=args.min_scene_diff,
                method=args.kf_method,
                interval_sec=args.kf_interval_sec,
                progress_cb=lambda p: progress.update(p),
            )
            progress.end_step()
            print(f"Keyframes saved: {len(keyframe_paths)}", flush=True)

    transcript_txt = transcript_dir / "transcript.txt"
    segments_json = transcript_dir / "segments.json"
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
    )
    progress.end_step()
    print(f"Transcription done: {len(segments)} segments", flush=True)

    if args.transcribe_only:
        classification_result = {"code": [], "plots": [], "images": []}
    else:
        print("Classifying frames...", flush=True)
        progress.start_step("Classify frames", 5)
        classification_result = classify_frames(
            frame_paths=keyframe_paths,
            classified_root=classified_dir,
            snippets_dir=snippets_dir,
            progress_cb=lambda p: progress.update(p),
        )
        progress.end_step()
        print(
            f"Classified - code: {len(classification_result.get('code', []))}, "
            f"plots: {len(classification_result.get('plots', []))}, "
            f"images: {len(classification_result.get('images', []))}",
            flush=True,
        )

    print("Building PDF report...", flush=True)
    progress.start_step("Build PDF", 5)
    report_pdf_path = output_dir / "report.pdf"
    build_pdf_report(
        output_pdf_path=report_pdf_path,
        transcript_txt_path=transcript_txt,
        segments_json_path=segments_json,
        classification_result=classification_result,
        output_dir=output_dir,
        progress_cb=lambda p: progress.update(p),
        report_style=args.report_style,
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
        "mode": "streaming" if args.streaming else "download",
        "transcribe_only": args.transcribe_only,
    }, indent=2))


if __name__ == "__main__":
    main()
